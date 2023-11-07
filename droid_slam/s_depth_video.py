import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict

from droid_net import cvx_upsample
import geom.projective_ops as pops

class S_DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, device="cuda:0"):  #buffer为数据缓存
                
        ###current keyframe count
        #使用multiprocessing库的示例，特别是使用Value类创建共享变量。在多进程编程中，共享变量是一种在多个进程间共享状态或数据的机制。
        #Value 创建一个在多个进程之间共享的变量。'i' 表示该变量是一个整数（integer）。0 是该共享整数的初始值。
        #self.counter 现在是一个可以在多个进程之间共享和修改的整数。
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]
        self.imageSeries = torch.zeros(buffer, device="cuda", dtype=torch.int16).share_memory_()

        ### state attributes ###
        #创建了全零浮点数[buffer]维度张量，并且存储在CUDA设备（通常是GPU）上。然后，这个张量被配置为可以在不同进程间共享内存
        #.share_memory_() 方法用于将一个张量（Tensor）的存储放入一个共享内存区域。这使得在多进程环境下，不同的进程可以访问和修改同一个张量
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        #创建全零，非负数，0-255的[buffer,3,ht,wd]维度的张量
        self.images = torch.zeros(buffer, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()

        #不为stereo, c=1;    为stereo, c=2
        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")

        
        
    def get_lock(self):
        return self.counter.get_lock()  

    def __item_setter(self, index, item):   #self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0])
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]    
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3] 

        if item[4] is not None:
            depth = item[4][3::8,3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    #一个线程安全的实现，用于向某种数据结构添加元素。它使用了一个锁来确保 append 操作是线程安全的，并使用 self.counter 的当前值作为索引
    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod #@staticmethod 是一个装饰器，用于指示一个方法是静态方法。静态方法是那些不需要访问实例变量或类变量的方法。因为静态方法不依赖于类的实例，它们可以直接通过类来调用，而不是类的实例。
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1) #成为一维张量
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.dirty[:self.counter.value] = True


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = S_DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None]) #[1,512,7]

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
        
        ii, jj = S_DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def cuda_ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)

            self.disps.clamp_(min=0.001)

    def py_ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            from geom.ba import BA
            poses = lietorch.SE3(self.poses[:t1][None]) #[1,8,7]
            disps = self.disps[:t1][None] #【1,8,48,64】
            for _ in range(itrs):
                poses, disps = BA(target, weight, eta, poses, disps, self.intrinsics[None], ii, jj, fixedp=t0)
            self.poses[:t1] = poses.data[0]
            self.disps[:t1] = disps[0]
            self.disps.clamp_(min=0.001)