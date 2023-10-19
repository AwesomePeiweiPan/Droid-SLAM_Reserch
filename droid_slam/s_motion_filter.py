import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock


class S_MotionFilter:
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
        """ main update operation - run on every frame in video """
        #创建一个SE(3)的单位元，并提取其作为一个简单的张量数据，同时去除所有大小为1的维度
        #Id=tensor([0,0,0,0,0,0,1])
        #Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        #[:, [2,1,0]]: 这个切片操作是用来调整图像的颜色通道顺序。
        #   通常，OpenCV读取图像时使用的是BGR（蓝绿红）顺序，而大多数图像处理库和深度学习模型则使用RGB（红绿蓝）顺序。
        #   这一操作将BGR转换为RGB。[2,3,384,512]中的第二维[3]从BGR变成RGB
        #None: 在切片操作中，这用于增加一个新的维度，image变成[1,2,3,384,512]
        #.to(self.device): 这一步是将张量移动到指定的计算设备上（CPU或GPU）
        #/ 255.0: 这一步是对图像进行归一化，将像素值从 [0, 255] 范围转换到 [0, 1] 范围。这通常是深度学习模型推理前的一个必要步骤。
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        #这两步操作合在一起，对输入张量进行了零均值（zero-mean）和单位标准差（unit standard deviation）的标准化。
        #这样通常可以帮助神经网络更快地收敛，并可能提高模型的性能。
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            #self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0])
            self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0])


        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            # check motion magnitue / add new frame to video
            #计算了 delta 张量在最后一个维度上的欧几里得范数，然后求这些范数值的平均，最终返回一个 Python 数字
            if delta.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0])

            else:
                self.count += 1