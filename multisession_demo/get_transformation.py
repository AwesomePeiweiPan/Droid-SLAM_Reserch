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
    #Set Parameters for Droid-SLAM process
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to euroc sequence")
    parser.add_argument("--gt", help="path to gt file")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
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

    #加载Sequence.txt中数据
    sequence_path = '/home/peiweipan/fbow/Euroc_MH/Sequence.txt'
    sequence_data = loop_detect.get_SequenceData(sequence_path)

    #加载之前的地图数据
    MH01_Keyframes = loop_detect.load_reconstruction_MH("MH01_SaveData")
    MH02_Keyframes = loop_detect.load_reconstruction_MH("MH02_SaveData")

    #通过之前的地图数据给对象赋值
    droid_MH01 = SDroid(args)
    loop_detect.Give_Data(droid_MH01, MH01_Keyframes)
    droid_MH02 = SDroid(args)
    loop_detect.Give_Data(droid_MH02, MH02_Keyframes)

    #获取loop照片路径
    Loop_paths = '/home/peiweipan/fbow/Euroc_MH/KeyFrames/MH01_02_Loop'
    Loop_directories_list = loop_detect.get_cam0_subdirectories(Loop_paths)

    W2ToW1 = torch.empty((0, 7))
    for key, value_list in sequence_data.items():
        for idx, list_item in enumerate(value_list[1:], start=1):  # 从第二个列表项开始
            corresponding_loop_dir = Loop_directories_list[idx - 1]  # 因为我们从1开始计数，所以需要-1来获得正确的索引
            
            #为每个对象添加输入参数并创建对象
            args.datapath=corresponding_loop_dir
            args.warmup=len(value_list[0])
            droid_loop = SDroid(args)

            #使用value_list中的索引提取数据, 赋予循环对象初值
            droid_loop.video.poses[:len(value_list[0])] = droid_MH01.video.poses[torch.tensor(value_list[0])]
            droid_loop.video.disps[:len(value_list[0])] = droid_MH01.video.disps[torch.tensor(value_list[0])]
            for (t, image, intrinsics) in tqdm(loop_detect.image_stream(args.datapath, stereo=args.stereo, stride=1)):
                droid_loop.track(t, image, intrinsics=intrinsics)

            #计算转换矩阵
            MH_old_poses = torch.tensor(list_item).tolist()
            MH_new_poses = list(range(len(value_list[0]), (len(value_list[0]) + len(list_item))))
            
            W2ToW1_element=torch.tensor((lietorch.SE3(droid_MH02.video.poses[MH_old_poses])*(lietorch.SE3(droid_loop.video.poses[MH_new_poses]).inv())).data.cpu().numpy())
            W2ToW1 = torch.cat([W2ToW1, W2ToW1_element], 0)
            
            del droid_loop
    
    #使用四分数法得出平均值并打印结果
    T = loop_detect.compute_filtered_mean(W2ToW1)
    loop_detect.printTransformMatrix(W2ToW1)
    
    #存储转换矩阵
    Transformation = T.cpu().numpy()
    reconstruction_path = "Transformation"
    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/T_MH01_MH02.npy".format(reconstruction_path), Transformation)

    print("Finish")


    








'''
Sdroid = SDroid(args)
    droidNew = SDroid(args)
    droid2 = SDroid(args)
    droidT=SDroid(args)
    #给16张照片初值
    for i in range(0,15):
        Sdroid.video.poses[i]= MH01poses[i+3636]
        print("给的照片初始值")
        print(i+3636)
        print(MH01poses[i+3636])
    for (t, image, intrinsics) in tqdm(ld.image_stream(args.datapath, stereo=args.stereo, stride=1)):
        Sdroid.track(t, image, intrinsics=intrinsics)
    traj_est = Sdroid.terminate(ld.image_stream(args.datapath, stride=1))
    for i in range(0,26):
        droidNew.video.poses[i]=traj_est[i+15]
    for i in range(15,41):
        droid2.video.poses[i-18]=MH02poses[i]
    ii=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    Gs2 = lietorch.SE3(droid2.video.poses)
    Gsnew = lietorch.SE3(droidNew.video.poses)
    Gij = Gs2[ii] * (Gsnew[ii].inv())  
    for i in range(0,26):
        print(Gij[i].data)
'''