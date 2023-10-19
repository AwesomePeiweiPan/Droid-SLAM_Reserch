import torch
import lietorch
import numpy as np

from lietorch import SE3
from v_factor_graph import V_FactorGraph


class V_DroidFrontend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update
        self.graph = V_FactorGraph(video, net.update, max_factors=48, upsample=args.upsample)

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

    def __update(self):
        """ add edges, perform update """

        self.count += 1
        self.t1 += 1

        # update visualization
        self.video.dirty[:self.t1] = True

    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value
        
        # initialization complete
        self.is_initialized = True

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

    def __call__(self):
        """ main update """

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()
