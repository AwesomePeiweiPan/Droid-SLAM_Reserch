import sys
sys.path.append('droid_slam')
from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
from torch.multiprocessing import Process
import torch.nn.functional as F
from pathlib import Path

import loop_detect 
from s_droid import SDroid

if __name__ == "__main__":

    reconstruction_path_T = "Transformation"
    reconstruction_path_Poses = "MH02_SaveData"
    newPoses = loop_detect.getTransformedPoses(reconstruction_path_T, reconstruction_path_Poses)
    
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

    #读取数据
    reconstruction_path_MH01 = "MH01_SaveData"
    MH01 = {}
    # Load each numpy array from the saved path
    MH01['images'] = np.load(f"reconstructions/{reconstruction_path_MH01}/images.npy")
    MH01['disps'] = np.load(f"reconstructions/{reconstruction_path_MH01}/disps.npy")
    MH01['poses'] = np.load(f"reconstructions/{reconstruction_path_MH01}/poses.npy")
    MH01['fmaps'] = np.load(f"reconstructions/{reconstruction_path_MH01}/fmaps.npy")
    MH01['inps'] = np.load(f"reconstructions/{reconstruction_path_MH01}/inps.npy")
    MH01['nets'] = np.load(f"reconstructions/{reconstruction_path_MH01}/nets.npy")
    MH01['intrinsics'] = np.load(f"reconstructions/{reconstruction_path_MH01}/intrinsics.npy")

    reconstruction_path_MH02 = "MH02_SaveData"
    MH02 = {}
    MH02['images'] = np.load(f"reconstructions/{reconstruction_path_MH02}/images.npy")
    MH02['disps'] = np.load(f"reconstructions/{reconstruction_path_MH02}/disps.npy")
    MH02['fmaps'] = np.load(f"reconstructions/{reconstruction_path_MH02}/fmaps.npy")
    MH02['inps'] = np.load(f"reconstructions/{reconstruction_path_MH02}/inps.npy")
    MH02['nets'] = np.load(f"reconstructions/{reconstruction_path_MH02}/nets.npy")
    MH02['intrinsics'] = np.load(f"reconstructions/{reconstruction_path_MH02}/intrinsics.npy")


    droid_MH = SDroid(args)
    droid_MH.video.images[:MH01['images'].shape[0] + MH02['images'].shape[0]] = torch.cat([torch.from_numpy(MH01['images']), torch.from_numpy(MH02['images'])], dim=0)
    droid_MH.video.poses[:MH01['poses'].shape[0] + newPoses.shape[0]] = torch.cat([torch.from_numpy(MH01['poses']), newPoses], dim=0)
    droid_MH.video.disps[:MH01['disps'].shape[0] + MH02['disps'].shape[0]] = torch.cat([torch.from_numpy(MH01['disps']), torch.from_numpy(MH02['disps'])], dim=0)
    droid_MH.video.fmaps[:MH01['fmaps'].shape[0] + MH02['fmaps'].shape[0]] = torch.cat([torch.from_numpy(MH01['fmaps']), torch.from_numpy(MH02['fmaps'])], dim=0)
    droid_MH.video.inps[:MH01['inps'].shape[0] + MH02['inps'].shape[0]] = torch.cat([torch.from_numpy(MH01['inps']), torch.from_numpy(MH02['inps'])], dim=0)
    droid_MH.video.nets[:MH01['nets'].shape[0] + MH02['nets'].shape[0]] = torch.cat([torch.from_numpy(MH01['nets']), torch.from_numpy(MH02['nets'])], dim=0)
    droid_MH.video.intrinsics[:MH01['intrinsics'].shape[0] + MH02['intrinsics'].shape[0]] = torch.cat([torch.from_numpy(MH01['intrinsics']), torch.from_numpy(MH02['intrinsics'])], dim=0)

    droid_MH.video.counter.value=MH01['poses'].shape[0] + newPoses.shape[0]
    droid_MH.terminate()
    
    reconstruction_path_Vis = "Vis"
    loop_detect.save_reconstruction_for_Vis(droid_MH, reconstruction_path_Vis)

    



    print("finished")