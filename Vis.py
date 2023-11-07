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


if __name__ == "__main__":
    #Set Parameters for Droid-SLAM process
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to euroc sequence")
    parser.add_argument("--gt", help="path to gt file")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
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
    args = parser.parse_args()
    #spawn启动更加稳定
    torch.multiprocessing.set_start_method('spawn')

    reconstruction_path = "Vis"
    MH_Mul = {}
    MH_Mul['disps'] = np.load(f"reconstructions/{reconstruction_path}/disps.npy")
    MH_Mul['poses'] = np.load(f"reconstructions/{reconstruction_path}/poses.npy")
    MH_Mul['images'] = np.load(f"reconstructions/{reconstruction_path}/images.npy")
    MH_Mul['intrinsics'] = np.load(f"reconstructions/{reconstruction_path}/intrinsics.npy")

    droid_MH = SDroid(args)
    droid_MH.video.poses[:MH_Mul['poses'].shape[0]] = torch.from_numpy(MH_Mul['poses'])
    droid_MH.video.disps[:MH_Mul['disps'].shape[0]] = torch.from_numpy(MH_Mul['disps'])
    droid_MH.video.images[:MH_Mul['images'].shape[0]] = torch.from_numpy(MH_Mul['images'])
    droid_MH.video.intrinsics[:MH_Mul['intrinsics'].shape[0]] = torch.from_numpy(MH_Mul['intrinsics'])

    # 假设 MH_Mul['poses'] 已经存在并且我们知道它的形状
    end_value = MH_Mul['poses'].shape[0]
    # 初始设置
    droid_MH.video.counter.value = 0                
    droid_MH.video.imageSeries[280:600] = 1

    # 循环
    for i in range(end_value):
        print(i)
        droid_MH.video.dirty[:droid_MH.video.counter.value] = True
        droid_MH.video.counter.value += 1
        time.sleep(0.5)
        

    print("Finished")