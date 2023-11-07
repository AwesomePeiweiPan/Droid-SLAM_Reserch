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
from droid import Droid
import torch.nn.functional as F
import os
import loop_detect 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to euroc sequence")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[320,512])
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--stereo", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=17.5)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=24.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=2)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    ###设定默认参数
    args.stereo = True
    args.disable_vis = True

    ###设定图片数据源
    datapaths = [
    "/home/peiweipan/Projects/DroidSlam/EurocData/OriginalData/MH01/mav0/cam0/data/",
    "/home/peiweipan/Projects/DroidSlam/EurocData/OriginalData/MH02/mav0/cam0/data/",
    "/home/peiweipan/Projects/DroidSlam/EurocData/OriginalData/MH03/mav0/cam0/data/",
    "/home/peiweipan/Projects/DroidSlam/EurocData/OriginalData/MH04/mav0/cam0/data/",
    "/home/peiweipan/Projects/DroidSlam/EurocData/OriginalData/MH05/mav0/cam0/data/",
    ]

    ###设定关键帧存储的目标文件
    dst_paths = [
    "/home/peiweipan/Projects/DroidSlam/EurocData/KeyFrames/MH01",
    "/home/peiweipan/Projects/DroidSlam/EurocData/KeyFrames/MH02",
    "/home/peiweipan/Projects/DroidSlam/EurocData/KeyFrames/MH03",
    "/home/peiweipan/Projects/DroidSlam/EurocData/KeyFrames/MH04",
    "/home/peiweipan/Projects/DroidSlam/EurocData/KeyFrames/MH05",
    ]

    ###设定关键帧数字数据的目标文件
    reconstruction_paths = [
    "/home/peiweipan/Projects/DroidSlam/EurocData/KeyFrames/KD01",
    "/home/peiweipan/Projects/DroidSlam/EurocData/KeyFrames/KD02",
    "/home/peiweipan/Projects/DroidSlam/EurocData/KeyFrames/KD03",
    "/home/peiweipan/Projects/DroidSlam/EurocData/KeyFrames/KD04",
    "/home/peiweipan/Projects/DroidSlam/EurocData/KeyFrames/KD05",
    ]

    # 确保所有列表的长度相同
    assert len(datapaths) == len(dst_paths) == len(reconstruction_paths)
    for datapath, dst_path, reconstruction_path in zip(datapaths, dst_paths, reconstruction_paths):
        #清空文件夹
        loop_detect.clear_directory(dst_path)
        loop_detect.clear_directory(reconstruction_path)

        args.datapath = datapath
        src_path = datapath
        args.reconstruction_path = reconstruction_path

        droid = Droid(args)
        for (t, image, intrinsics) in tqdm(loop_detect.Euroc_image_stream(args.datapath, stereo=args.stereo, stride=1)):
            droid.track(t, image, intrinsics=intrinsics)

        loop_detect.extract_images_by_timestamp(src_path, dst_path, droid.video.tstamp)
        droid.terminate(args.reconstruction_path, loop_detect.Euroc_image_stream(args.datapath, stride=1))

        del droid
        torch.cuda.empty_cache()

        print("Finished processing for:", datapath)
