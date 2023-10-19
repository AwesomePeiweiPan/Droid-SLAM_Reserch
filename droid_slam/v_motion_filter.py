import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock


class V_MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        #torch.as_tensor([0.485, 0.456, 0.406], device=self.device): 创建一个含有三个元素的一维张量（Tensor）。
        #这三个元素是RGB通道的平均值。张量存储在self.device指定的设备上，可能是CPU或GPU。
        #[:, None, None]: 这部分将一维张量转换为三维张量，尺寸变为 [3, 1, 1]。这样做通常是为了方便后续与图像张量进行广播（broadcasting）操作
        #由于MEAN和STDV都是 [3, 1, 1] 形状的张量，它们可以直接与 [batch_size, 3, height, width] 形状的图像张量进行广播操作，从而同时对一个批次的多张图像进行标准化。
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2) #沿着dim=2分割为两个128
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    #这个装饰器用于包装那些执行张量计算的函数或方法。当装饰器启用时（enabled=True），在该函数或方法内部执行的张量计算会自动选择使用float16或float32来进行，以实现性能的优化。
    #fnet为BasicEncoder类的一个实例。BasicEncoder属于nn.Module的子类；nn.Module中定义了一个特殊的'__call__'方法，会调用forward方法
    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None):
        self.video.append(tstamp, image[0], None, None, None, None)

