import torch
import lietorch
import numpy as np

import matplotlib.pyplot as plt
from lietorch import SE3
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops


class V_FactorGraph:
    def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1, upsample=False):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl
        self.upsample = upsample

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)  #由两个48x64的矩阵组成[48,64,2]维度。第一个每一行从左往右0-47，第二个每一列从上到下0-63
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps) #【512，48，64】

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set( #建立一个eset集合，里面的数据是例如{(10,13),(11,14)}这样的对应边
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset  #如果ii,jj新建立的边不在eset中，则它们对应的keep为true

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0,2,3,4])
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    @torch.cuda.amp.autocast(enabled=True) #PyTorch 中的自动混合精度（Automatic Mixed Precision，AMP）运算
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges   去除和之前重复的边
        ii, jj = self.__filter_repeated_edges(ii, jj)


        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:
            
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(0) 

        # correlation volume for new edges
        if self.corr_impl == "volume":
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii,0].to(self.device).unsqueeze(0)
            fmap2 = self.video.fmaps[jj,c].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr) #如果 self.corr 为空，则将其设置为新计算出的 corr。否则，将新的 corr 与旧的 self.corr 连接。

            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.cuda.amp.autocast(enabled=False): # 禁用自动混合精度
            target, _ = self.video.reproject(ii, jj)
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0) #将两个张量沿着第 0 维（通常是行维度）拼接起来
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]


    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """


        with self.video.get_lock():
            self.video.images[ix] = self.video.images[ix+1]
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_sens[ix] = self.video.disps_sens[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]

            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]

        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:,~m]
            self.weight_inac = self.weight_inac[:,~m]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)


    @torch.cuda.amp.autocast(enabled=True)  #不支持混合精度计算
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False):
        """ run update operator on factor graph """

        # motion features
        with torch.cuda.amp.autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj) # coords1[1,22,48,64,2],mask为valid_mask,说明投影的3D点X0,X1的深度值是否大于MIN_DEPTH
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1) #self.coords0[48,64,2]，motn[1,22,48,64,4]
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0) #重新排列维度并且限制数值在(-64,64),变成[1,22,4,48,64]
        
        # correlation features
        corr = self.corr(coords1)

        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with torch.cuda.amp.autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float) #delta[1,22,48,64,2]
            self.weight = weight.to(dtype=torch.float) #[1,22,48,64,2]

            ht, wd = self.coords0.shape[0:2]
            self.damping[torch.unique(self.ii)] = damping   #self.damping[512,48,64];   damping[1,8,48,64]

            if use_inactive:    #在initialize阶段没有任何变
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)

            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight


            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            if 1:
                target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()   #contiguous 确保张量在内存中是连续的,为了提高性能或者其他操作
                weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
                # dense bundle adjustment
                self.video.cuda_ba(target, weight, damping, ii, jj, t0, t1, 
                    itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)
            else:
                # dense bundle adjustment
                self.video.py_ba(target, weight, damping, ii, jj, t0, t1, 
                    itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)

            if self.upsample:
                self.video.upsample(torch.unique(self.ii), upmask)

        self.age += 1


    @torch.cuda.amp.autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        t = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        for step in range(steps):
            print("Global BA Iteration #{}".format(step+1))
            with torch.cuda.amp.autocast(enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            s = 8
            for i in range(0, self.jj.max()+1, s):
                v = (self.ii >= i) & (self.ii < i + s)
                iis = self.ii[v]
                jjs = self.jj[v]

                ht, wd = self.coords0.shape[0:2]
                corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())

                with torch.cuda.amp.autocast(enabled=True):
                 
                    net, delta, weight, damping, upmask = \
                        self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis, jjs)

                    if self.upsample:
                        self.video.upsample(torch.unique(iis), upmask)

                self.net[:,v] = net
                self.target[:,v] = coords1[:,v] + delta.float()
                self.weight[:,v] = weight.float()
                self.damping[torch.unique(iis)] = damping

            damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP
            target = self.target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = self.weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            # dense bundle adjustment
            self.video.cuda_ba(target, weight, damping, self.ii, self.jj, 1, t, 
                itrs=itrs, lm=1e-5, ep=1e-2, motion_only=False)

            self.video.dirty[:t] = True

    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r 在一系列帧（从 t0 到 t1）之间添加边，这些边连接的帧之间的时间或空间距离应在 (c, r] 范围内 """

        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1)) #ii为14x14矩阵，每行分别为0-14;jj为14x14矩阵，每列分别为0-14
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)      #ii为255的long类型的张量，并将其移动到指定的设备
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)      #jj为255的long类型的张量，并将其移动到指定的设备

        c = 1 if self.video.stereo else 0

        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])        #神奇！得到的ii,jj对应的数组便是对应的帧图

    
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        """ add edges to the factor graph based on distance """

        t = self.video.counter.value
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx)
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=beta)
        d[ii - rad < jj] = np.inf   #确保在由(t0,t)组成的因子图中，一个帧与前面的rad帧之间不会有直接的连接
        d[d > 100] = np.inf
        #这种方法是在尝试对每对 (i, j) 创建一个邻域，并使这些邻域内的距离值变为无穷大。
        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf


        es = []
        for i in range(t0, t):
            if self.video.stereo:    #连接自己，并且设置距离为无穷大
                es.append((i, i))
                d[(i-t0)*(t-t1) + (i-t1)] = np.inf

            for j in range(max(i-rad-1,0), i): #连接i之前的rad内的边并创建双向链接
                es.append((i,j))
                es.append((j,i))
                d[(i-t0)*(t-t1) + (j-t1)] = np.inf

        ix = torch.argsort(d)
        for k in ix:
            if d[k].item() > thresh:
                continue

            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]
            
            # bidirectional
            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)