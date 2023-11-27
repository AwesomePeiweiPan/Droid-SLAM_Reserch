import os
import sys
sys.path.append('droid_slam')
from tqdm import tqdm
import numpy as np
import torch
import lietorch
import shutil
from pathlib import Path
import loop_detect 
from s_droid import SDroid
import argparse
import torchgeometry as tgm
import re

transformed_pose_path = "/home/peiweipan/Projects/DroidSlam/EurocData/TransformedKeyPos/"
N=2
AllMaps_poses_path = "/home/peiweipan/Projects/DroidSlam/EurocData/AllMapsPoses/"



def extract_number(s):
    """ 从字符串中提取出数字部分并返回 """
    numbers = re.findall(r'\d+', s)
    return int(numbers[0]) if numbers else 0

def get_sorted_folder_paths(path):
    """ 返回给定路径下按数字顺序排序的所有文件夹路径 """

    # 确保路径存在
    if not os.path.exists(path):
        print(f"路径不存在: {path}")
        return []

    # 获取所有文件夹并按数字排序
    folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    folders.sort(key=extract_number)

    # 构建并返回文件夹路径的列表
    sorted_folder_paths = [os.path.join(path, folder) for folder in folders]
    return sorted_folder_paths

def load_and_sample_data(folder_path, N):
    """
    从指定文件夹加载NumPy文件，并按照第0维度每隔N个元素抽样保存。

    :param folder_path: 包含NumPy文件的文件夹路径。
    :param N: 每隔N个元素进行一次抽样。
    :return: 一个字典，键为文件名（去掉.npy扩展名），值为抽样后的数据。
    """
    data = {}
    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            file_path = os.path.join(folder_path, file)
            loaded_data = np.load(file_path)
            # 每隔N个元素进行抽样
            sampled_data = loaded_data[::N]
            # 将抽样后的数据保存到字典中，键为文件名去掉.npy扩展名
            data[file[:-4]] = sampled_data

    return data

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
    args.buffer = 700
    args.stereo = True
    args.disable_vis = True

    Droid_All = SDroid(args)

    current_pos = 0

    initial = True

    sorted_folders = get_sorted_folder_paths(transformed_pose_path)
    for folder_path in sorted_folders:
        sampled_data = load_and_sample_data(folder_path, N)
        if initial:
            Droid_All.video.tstamp[:sampled_data['tstamps'].shape[0]] = torch.from_numpy(sampled_data['tstamps'])
            Droid_All.video.poses[:sampled_data['poses'].shape[0]] = torch.from_numpy(sampled_data['poses'])
            Droid_All.video.disps[:sampled_data['disps'].shape[0]] = torch.from_numpy(sampled_data['disps'])
            Droid_All.video.images[:sampled_data['images'].shape[0]] = torch.from_numpy(sampled_data['images'])
            Droid_All.video.intrinsics[:sampled_data['intrinsics'].shape[0]] = torch.from_numpy(sampled_data['intrinsics'])
            Droid_All.video.fmaps[:sampled_data['fmaps'].shape[0]] = torch.from_numpy(sampled_data['fmaps'])
            Droid_All.video.inps[:sampled_data['inps'].shape[0]] = torch.from_numpy(sampled_data['inps'])
            Droid_All.video.nets[:sampled_data['nets'].shape[0]] = torch.from_numpy(sampled_data['nets'])
            current_pos = sampled_data['poses'].shape[0]

            print(sampled_data['poses'].shape[0])

            initial = False
        else:
            ts_len = sampled_data['tstamps'].shape[0]
            poses_len = sampled_data['poses'].shape[0]
            disps_len = sampled_data['disps'].shape[0]
            images_len = sampled_data['images'].shape[0]
            intrinsics_len = sampled_data['intrinsics'].shape[0]
            fmaps_len = sampled_data['fmaps'].shape[0]
            inps_len = sampled_data['inps'].shape[0]
            nets_len = sampled_data['nets'].shape[0]

            Droid_All.video.tstamp[current_pos:current_pos + ts_len] = torch.from_numpy(sampled_data['tstamps'])
            Droid_All.video.poses[current_pos:current_pos + poses_len] = torch.from_numpy(sampled_data['poses'])
            Droid_All.video.disps[current_pos:current_pos + disps_len] = torch.from_numpy(sampled_data['disps'])
            Droid_All.video.images[current_pos:current_pos + images_len] = torch.from_numpy(sampled_data['images'])
            Droid_All.video.intrinsics[current_pos:current_pos + intrinsics_len] = torch.from_numpy(sampled_data['intrinsics'])
            Droid_All.video.fmaps[current_pos:current_pos + fmaps_len] = torch.from_numpy(sampled_data['fmaps'])
            Droid_All.video.inps[current_pos:current_pos + inps_len] = torch.from_numpy(sampled_data['inps'])
            Droid_All.video.nets[current_pos:current_pos + nets_len] = torch.from_numpy(sampled_data['nets'])

            # 更新 current_pos 和 poses_lengths
            current_pos += poses_len
            print(sampled_data['poses'].shape[0])

    Droid_All.video.counter.value = current_pos
    Droid_All.terminate()

    ts_file_path = os.path.join(AllMaps_poses_path, "tstamp.npy")
    pose_file_path = os.path.join(AllMaps_poses_path, "poses.npy")
    disps_file_path = os.path.join(AllMaps_poses_path, "disps.npy")
    images_file_path = os.path.join(AllMaps_poses_path, "images.npy")
    intrinsics_file_path = os.path.join(AllMaps_poses_path, "intrinsics.npy")
    fmaps_file_path = os.path.join(AllMaps_poses_path, "fmaps.npy")
    inps_file_path = os.path.join(AllMaps_poses_path, "inps.npy")
    nets_file_path = os.path.join(AllMaps_poses_path, "nets.npy")

    np.save(ts_file_path, Droid_All.video.tstamp.cpu().numpy()) 
    np.save(pose_file_path, Droid_All.video.poses.cpu().numpy()) 
    np.save(disps_file_path, Droid_All.video.disps.cpu().numpy()) 
    np.save(images_file_path, Droid_All.video.images.cpu().numpy()) 
    np.save(intrinsics_file_path, Droid_All.video.intrinsics.cpu().numpy()) 
    np.save(fmaps_file_path, Droid_All.video.fmaps.cpu().numpy()) 
    np.save(inps_file_path, Droid_All.video.inps.cpu().numpy()) 
    np.save(nets_file_path, Droid_All.video.nets.cpu().numpy()) 