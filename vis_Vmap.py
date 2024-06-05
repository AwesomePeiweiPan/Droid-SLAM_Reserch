import sys
sys.path.append('droid_slam')
from tqdm import tqdm
import cv2
import numpy as np
import torch
import argparse
from torch.multiprocessing import Process
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import open3d as o3d
from lietorch import SE3
import loop_detect 
from s_droid import SDroid
import time
import os


if __name__ == "__main__":
    #Set Parameters for Droid-SLAM process
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to euroc sequence")
    parser.add_argument("--gt", help="path to gt file")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=2000)
    parser.add_argument("--image_size", default=[320,512])
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--beta", type=float, default=0.3)
    #确定新帧是否需要被添加为关键帧,如果与上一帧之间的光流距离大于这个值则加为新的关键帧 初始值为2.4
    parser.add_argument("--filter_thresh", type=float, default=0)
    #确定什么时候开始 droid.frontend.__initialize, droid.frontend.__update 初始值为15
    parser.add_argument("--warmup", type=int, default=8)
    #update 过程中, 最后判定新的帧与上一个关键帧之间的物理距离d是否大于这个阈值, 初始值为3.5
    parser.add_argument("--keyframe_thresh", type=float, default=0)
    #add_proximity_factors 前端过程中，如果两帧距离之间的物理距离小于这个值，则加双向边
    parser.add_argument("--frontend_thresh", type=float, default=17.5)
    #add_proximity_fatcor 过程中，对应的 初始 jj 的窗口大小
    parser.add_argument("--frontend_window", type=int, default=20)
    #add_proximity_fatcor 过程中，对应的 初始 ii 的窗口大小
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    #add_proximity_factors 后端过程中，如果两帧距离之间的物理距离小于这个值，则加双向边
    parser.add_argument("--backend_thresh", type=float, default=24.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=2)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--Good", action="store_true")
    args = parser.parse_args()
    #spawn启动更加稳定
    torch.multiprocessing.set_start_method('spawn')


    M1_path = "/data/peiweipan/VMapData/TransformedKeyPos/site2/KD01/"
    M2_path = "/data/peiweipan/VMapData/TransformedKeyPos/site2/KD02/"
    M3_path = "/data/peiweipan/VMapData/TransformedKeyPos/site2/KD03/"

        


    M_First = {}
    M_First['poses'] = np.load(os.path.join(M1_path, 'poses.npy'))
    M_First['disps'] = np.load(os.path.join(M1_path, 'disps.npy'))
    M_First['images'] = np.load(os.path.join(M1_path, 'images.npy'))
    M_First['intrinsics'] = np.load(os.path.join(M1_path, 'intrinsics.npy'))

    M_Second = {}
    M_Second['poses'] = np.load(os.path.join(M2_path, 'poses.npy'))
    M_Second['disps'] = np.load(os.path.join(M2_path, 'disps.npy'))
    M_Second['images'] = np.load(os.path.join(M2_path, 'images.npy'))
    M_Second['intrinsics'] = np.load(os.path.join(M2_path, 'intrinsics.npy'))


    # M_Third
    M_Third = {}
    M_Third['poses'] = np.load(os.path.join(M3_path, 'poses.npy'))
    M_Third['disps'] = np.load(os.path.join(M3_path, 'disps.npy'))
    M_Third['images'] = np.load(os.path.join(M3_path, 'images.npy'))
    M_Third['intrinsics'] = np.load(os.path.join(M3_path, 'intrinsics.npy'))

    droid_MH = SDroid(args)
    # 初始化一个列表，用于存储每组数据中 poses 的长度
    poses_lengths = []

    # 加载 M_First 数据
    droid_MH.video.poses[:M_First['poses'].shape[0]] = torch.from_numpy(M_First['poses'])
    droid_MH.video.disps[:M_First['disps'].shape[0]] = torch.from_numpy(M_First['disps'])
    droid_MH.video.images[:M_First['images'].shape[0]] = torch.from_numpy(M_First['images'])
    droid_MH.video.intrinsics[:M_First['intrinsics'].shape[0]] = torch.from_numpy(M_First['intrinsics'])

    # 将 M_First 的 poses 长度添加到列表
    poses_lengths.append(M_First['poses'].shape[0])

    # 定义一个变量，用于追踪当前填充的位置
    current_pos = M_First['poses'].shape[0]

    # 为 M_Second, M_Third, M_Fourth, M_Fifth 进行同样的操作
    datasets = [M_Second, M_Third]
    for dataset in datasets:
        poses_len = dataset['poses'].shape[0]
        disps_len = dataset['disps'].shape[0]
        images_len = dataset['images'].shape[0]
        intrinsics_len = dataset['intrinsics'].shape[0]

        droid_MH.video.poses[current_pos:current_pos + poses_len] = torch.from_numpy(dataset['poses'])
        droid_MH.video.disps[current_pos:current_pos + disps_len] = torch.from_numpy(dataset['disps'])
        droid_MH.video.images[current_pos:current_pos + images_len] = torch.from_numpy(dataset['images'])
        droid_MH.video.intrinsics[current_pos:current_pos + intrinsics_len] = torch.from_numpy(dataset['intrinsics'])

        # 更新 current_pos 和 poses_lengths
        current_pos += poses_len
        poses_lengths.append(poses_len)

    # 假设 MH_Mul['poses'] 已经存在并且我们知道它的形状
    end_value =M_First['poses'].shape[0] +M_Second['poses'].shape[0]+M_Third['poses'].shape[0]
    # 初始设置
    i=0
    droid_MH.video.counter.value = 0                
    droid_MH.video.imageSeries[M_First['poses'].shape[0]:M_First['poses'].shape[0] +M_Second['poses'].shape[0]] = 5
    i=M_First['poses'].shape[0] +M_Second['poses'].shape[0]
    droid_MH.video.imageSeries[i:i+M_Third['poses'].shape[0]] = 9
    i=i+M_Third['poses'].shape[0]



    time.sleep(5)


    # 循环
    for i in range(0, end_value-5, 5):
        print(i)
        droid_MH.video.dirty[:droid_MH.video.counter.value] = True
        droid_MH.video.counter.value += 5
        time.sleep(0.1)
        

    print("Finished")