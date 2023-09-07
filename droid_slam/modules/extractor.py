import torch
#torch.nn 模块是 PyTorch 中的一个重要模块，用于定义神经网络相关的类和函数
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


DIM=32

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, multidim=False):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim

        ###批量归一化
        #这是最常用的归一化方法之一，通常用于卷积神经网络（CNNs）和全连接层（Dense layers）。
        #在每个小批量的数据中，这种归一化技术独立地对每个输入特征进行零均值和单位方差的归一化。
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)

        ###Batch normalization
        #它有助于解决深度网络训练中的一系列问题，包括梯度消失/梯度爆炸、模型训练不稳定等 
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        ###实例归一化
        #神经网络中的归一化层用于改善网络训练的稳定性和收敛速度。
        #通过归一化，网络中的每一层可以在一个更稳定的数值范围内接收数据，这通常可以加快训练速度，并有时还能提高模型性能
        #创建norm1层，使用nn.InstanceNorm2d类。这层是一个2D实例归一化层，用于对输入进行实例归一化操作
        #DIM表示归一化通道数，这里使用global参数32
        #实例归一化（Instance Normalization）是一种深度学习中的归一化方法，通常用于神经网络中的中间层。
        #它通过对每个样本的每个通道进行归一化，使得每个样本在每个通道上的分布接近标准正态分布。
        #这有助于加速收敛、提升模型的稳定性，并且在图像生成任务中也能够产生更好的效果。
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)


        #nn.Sequential() 创建了一个空的Sequential模型。
        #这意味着 self.norm1 实际上不会对通过它的数据执行任何操作。
        #可选的规范化层：在某些神经网络架构中，你可能希望可选地包含或排除规范化层（如Batch Normalization、Layer Normalization等）。
        #通过将 self.norm1 设置为一个空的 nn.Sequential() 对象，你可以在不更改其他代码的情况下轻松地排除规范化。
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        ###构建一个卷积神经网络的一部分 CNN
        #创建二维卷积层 参数含义: 输入通道数，输出通道数，卷积核7x7，卷积步长2，输入图像周围填充3个像素
        self.conv1 = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
        #创建ReLU激活函数
        self.relu1 = nn.ReLU(inplace=True)
        
        #用于后续的网络层中，表示输入的通道数
        self.in_planes = DIM
        #
        self.layer1 = self._make_layer(DIM,  stride=1)
        self.layer2 = self._make_layer(2*DIM, stride=2)
        self.layer3 = self._make_layer(4*DIM, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(4*DIM, output_dim, kernel_size=1)

        if self.multidim:
            self.layer4 = self._make_layer(256, stride=2)
            self.layer5 = self._make_layer(512, stride=2)

            self.in_planes = 256
            self.layer6 = self._make_layer(256, stride=1)

            self.in_planes = 128
            self.layer7 = self._make_layer(128, stride=1)

            self.up1 = nn.Conv2d(512, 256, 1)
            self.up2 = nn.Conv2d(256, 128, 1)
            self.conv3 = nn.Conv2d(128, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        #layer中含有conv1,conv2,norm1,norm2
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        b, n, c1, h1, w1 = x.shape
        #.view()方法用于调整张量的形状而不改变其数据。在这里，张量x的形状被改变为(b * n, c1, h1, w1)。
        x = x.view(b*n, c1, h1, w1) 

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)
