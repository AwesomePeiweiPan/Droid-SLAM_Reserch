import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


class S_DroidFrontend:
    def __init__(self, net, video, args):
        self.Good = args.Good
        
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(video, net.update, max_factors=48, upsample=args.upsample)

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        #预热阶段:初始化为8
        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        #keyframe_thresh 可以基于:时间间隔:有的SLAM系统会简单地每隔一定数量的帧选取一个关键帧，这也可以用keyframe_thresh来控制。这里初始化为4.0
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

        self.badT = []
        

    def __update(self):
        """ add edges, perform update """

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0, 
           self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])


        if self.Good == True:

            for itr in range(self.iters1):
                self.graph.update(None, None, use_inactive=True)

        # set initial pose for next frame
            poses = SE3(self.video.poses)
            d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

            if d.item() < self.keyframe_thresh:
                self.graph.rm_keyframe(self.t1 - 2)
            
                with self.video.get_lock():
                    self.video.counter.value -= 1
                    self.t1 -= 1

            else:
                for itr in range(self.iters2):
                    self.graph.update(None, None, use_inactive=True)

        # set pose for next itration
            self.video.poses[self.t1] = self.video.poses[self.t1-1]
            self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()
            self.video.intrinsics[self.t1] = self.video.intrinsics[self.t1-1]

        # update visualization
            self.video.dirty[self.graph.ii.min():self.t1] = True

        if self.Good == False:
            if self.is_initialized == False:

                for itr in range(self.iters1):
                    self.graph.update(None, None, use_inactive=True)

            # set initial pose for next frame
                poses = SE3(self.video.poses)
                d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

                if d.item() < self.keyframe_thresh:
                    self.graph.rm_keyframe(self.t1 - 2)
            
                    with self.video.get_lock():
                        self.video.counter.value -= 1
                        self.t1 -= 1

                else:
                    for itr in range(self.iters2):
                        self.graph.update(None, None, use_inactive=True)

            # set pose for next itration
                self.video.poses[self.t1] = self.video.poses[self.t1-1]
                self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

            # update visualization
                self.video.dirty[self.graph.ii.min():self.t1] = True
        
            else:
            
                with torch.cuda.amp.autocast(enabled=False):
                    coords1, mask = self.graph.video.reproject(self.graph.ii, self.graph.jj) # coords1[1,22,48,64,2],mask为valid_mask,说明投影的3D点X0,X1的深度值是否大于MIN_DEPTH
                    motn = torch.cat([coords1 - self.graph.coords0, self.graph.target - coords1], dim=-1) #self.coords0[48,64,2]，motn[1,22,48,64,4]
                    motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)
            
                with torch.cuda.amp.autocast(enabled=True):
                    corr = self.graph.corr(coords1)

                    self.graph.net, delta, weight, damping, upmask = \
                    self.graph.update_op(self.graph.net, self.graph.inp, corr, motn, self.graph.ii, self.graph.jj)

                    summed_weight = weight.sum(dim=(-3, -2, -1)) 
            
            # 找到 ii 和 jj 中等于 last_non_zero_value 的索引
            # 找到 ii 和 jj 中等于 last_non_zero_value 的索引
                indices_ii = (self.graph.ii == self.video.counter.value -1).nonzero(as_tuple=False).squeeze()
                indices_jj = (self.graph.jj == self.video.counter.value -1).nonzero(as_tuple=False).squeeze()

            # 过滤 ii 和 jj 的索引，确保 jj (对于 ii 的索引) 和 ii (对于 jj 的索引) 大于 last_non_zero_value - 6
                filtered_indices_ii = [idx for idx in indices_ii if self.graph.jj[idx] > self.video.counter.value -4 and self.graph.jj[idx] != self.graph.ii[idx]]
                filtered_indices_jj = [idx for idx in indices_jj if self.graph.ii[idx] > self.video.counter.value -4 and self.graph.ii[idx] != self.graph.jj[idx]]

        
                filtered_indices_ii_tensor = torch.tensor(filtered_indices_ii)
                filtered_indices_jj_tensor = torch.tensor(filtered_indices_jj)
                combined_unique_indices = torch.sort(torch.cat((filtered_indices_ii_tensor, filtered_indices_jj_tensor))).values


            #计算平均值
                selected_values = summed_weight[:, combined_unique_indices].squeeze()
                average_of_selected_values = selected_values.mean()

                if average_of_selected_values.item() > 200 and torch.all(selected_values > 10):
                
                    for itr in range(self.iters1):
                        self.graph.update(None, None, use_inactive=True)


                    for itr in range(self.iters2):
                        self.graph.update(None, None, use_inactive=True)

                # set pose for next itration
                    self.video.poses[self.t1] = self.video.poses[self.t1-1]
                    self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

                # update visualization
                    self.video.dirty[self.graph.ii.min():self.t1] = True
            
                else:
                    self.graph.rm_keyframe(self.t1 - 2)
            
                    with self.video.get_lock():
                    
                    #获取当前的tstamp
                        non_zero_indices = torch.nonzero(self.video.tstamp).squeeze()
                        last_non_zero_index = non_zero_indices[-1]
                        last_non_zero_value = self.video.tstamp[last_non_zero_index].item()            

                        self.video.counter.value -= 1
                        self.t1 -= 1

                        self.badT.append(last_non_zero_value)


    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value
        
        #创建深度学习网络，通过图片序号创建邻接矩阵
        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)  #graph的ii,jj,target[1,22,48,64,2]还有神经网络赋值

        #使用BA更新droid.video.poses和droid.video.disps
        for itr in range(8):
            self.graph.update(1, use_inactive=True)
        
        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)


        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.tstamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self):
        """ main update """

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()

        
