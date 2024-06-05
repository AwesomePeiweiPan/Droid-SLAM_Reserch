import torch
import lietorch
import numpy as np

import os

from droid_net import DroidNet
from s_depth_video import S_DepthVideo
from s_motion_filter import S_MotionFilter
from s_droid_frontend import S_DroidFrontend
from s_droid_backend import S_DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process

from s_visualization import Sdroid_visualization


class SDroid:
    def __init__(self, args):
        #Droid类继承Object父类
        super(SDroid, self).__init__()
        
        #形成一个net对象，拥有cnet对象, fnet对象, update对象
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = S_DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = S_MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = S_DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = S_DroidBackend(self.net, self.video, self.args)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

        
        

        if not self.disable_vis:
            self.visualizer = Process(target=Sdroid_visualization, args=(self.video,))
            self.visualizer.start()

        



    def load_weights(self, weights):
        """ load trained model weights """
        #形成net，包含BasicEncoder类的cnet, fnet;UpdateModule类的update
        self.net = DroidNet()
        #.items(): 对加载的权重（一个字典）调用 items() 函数，以得到一个由 (key, value) 对组成的迭代器。在这里，key 是权重张量的名称，value 是权重张量本身。
        #(k.replace("module.", ""), v) for (k, v) in ...: 这一部分是一个列表推导式，用于创建一个新的 OrderedDict。
        #   它遍历所有的 (key, value) 对，并通过 k.replace("module.", "") 移除每个键（权重名称）中的 "module." 前缀。
        #   这样做的原因通常是因为当模型是作为一个 nn.DataParallel 或 nn.DistributedDataParallel 实例保存时，权重名称会有一个额外的 "module." 前缀。
        #   当你尝试将这些权重加载到没有并行包装的模型实例时，这个前缀可能会导致问题。
        #OrderedDict([...]): 最终，这些经过处理的 (key, value) 对被放入一个有序字典（OrderedDict）
        #综合起来，这段代码实现了从文件加载模型权重并移除 "module." 前缀（如果存在）的功能，然后将处理后的权重保存到一个有序字典中。
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])
        #从 [3,128,3,3] 到 [2,128,3,3]
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        #从 [3,1] 到 [2,1]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        #load_state_dict() 是一个用于加载模型参数的方法。参数由 state_dict 字典提供，该字典通常包含每一层的权重和偏置
        self.net.load_state_dict(state_dict)
        #self.net.to("cuda:0") 将模型从CPU移动到编号为0的GPU上。如果你的机器上有多个GPU，"cuda:1"、"cuda:2" 等会用于不同编号的GPU。
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        #这样做的好处是能够减少内存消耗并提高性能，因为不需要存储中间状态（用于稍后的梯度计算）
        #需要注意的是，在这个上下文管理器内创建的任何张量都默认不需要梯度（.requires_grad 属性为 False）
        #简而言之，with torch.no_grad(): 是一种优化手段，用于在不需要梯度计算的情境下节省计算资源。
        with torch.no_grad():
            # check there is enough motion，创建droid.video的poses, disps等值
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

            # global bundle adjustment
            # self.backend()

    def terminate(self):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)
        
        #camera_trajectory = self.traj_filler(stream)
        #return camera_trajectory.data.cpu()
        #return camera_trajectory.inv().data.cpu().numpy()

