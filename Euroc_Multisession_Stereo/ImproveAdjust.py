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

import torch.nn.functional as F
import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from scipy.spatial.transform import Rotation as R


group_sequence_path = '/home/peiweipan/fbow/Euroc_Maps/GroupSequence/GroupSequence.txt' 
loop_folder_path = '/home/peiweipan/Projects/DroidSlam/EurocData/Loop'  
keyframe_data_file_path = '/home/peiweipan/Projects/DroidSlam/EurocData/KeyFrames' 
transfomed_pose_path = "/home/peiweipan/Projects/DroidSlam/EurocData/ImproveTrans/"  # 替换为新文件夹的路径



def find_loop_folder(loop_folder_path, line):
    # 构建文件夹名称并找到文件夹路径
    folder_name = '_'.join(line.split())
    folder_path = os.path.join(loop_folder_path, folder_name)

    # 检查文件夹下的.txt文件
    txt_file = None
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            txt_file = os.path.join(folder_path, file)
            break

    return folder_path, txt_file

def find_keyframes_data(keyframe_data_file_path, line):
    # 分离数字并构建KD文件夹名称
    num1, num2 = line.split()
    kd_folder1 = os.path.join(keyframe_data_file_path, f"KD{int(num1):02d}")
    kd_folder2 = os.path.join(keyframe_data_file_path, f"KD{int(num2):02d}")

    return kd_folder1, kd_folder2

def copy_folder(src, dst):
    dst_folder_path = os.path.join(dst, os.path.basename(src))
    # 如果目标文件夹已存在，则先清空它
    if os.path.exists(dst_folder_path):
        shutil.rmtree(dst_folder_path)
    shutil.copytree(src, dst_folder_path)
    return dst_folder_path

def check_and_update_path(folder, reference_folder):
    folder_name = os.path.basename(folder)
    reference_folder_path = os.path.join(reference_folder, folder_name)

    if os.path.exists(reference_folder_path):
        return reference_folder_path  # 如果新文件夹中已存在对应的文件夹，则返回新路径
    return folder



if __name__ == '__main__':
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

    ###设定默认参数
    args.stereo = True
    args.disable_vis = True

    loop_detect.clear_all_subdirectories(transfomed_pose_path)
    is_first_iteration = True  
    with open(group_sequence_path, 'r') as file:
        for line in file:
            single_loop_folder_path, single_group_sequence_path = find_loop_folder(loop_folder_path, line.strip())
            kd_folder1, kd_folder2 = find_keyframes_data(keyframe_data_file_path, line.strip())

            if is_first_iteration:
                new_kd_folder1_path = copy_folder(kd_folder1, transfomed_pose_path)
                kd_folder1 = new_kd_folder1_path  # 更新 kd_folder1 的路径为新文件夹的路径
                is_first_iteration = False
            else:
                kd_folder1 = check_and_update_path(kd_folder1, transfomed_pose_path)
                kd_folder2 = check_and_update_path(kd_folder2, transfomed_pose_path)

            #加载Sequence.txt中数据
            sequence_data = loop_detect.get_SequenceData(single_group_sequence_path)

            #加载之前的地图数据
            MH_Keyframes_First = loop_detect.load_reconstruction(kd_folder1)
            len_First = len(MH_Keyframes_First['poses'])
            MH_Keyframes_Second = loop_detect.load_reconstruction(kd_folder2)
            len_Second = len(MH_Keyframes_Second['poses'])

            #通过之前的地图数据给对象赋值
            droid_MH_First = SDroid(args)
            loop_detect.Give_Data(droid_MH_First, MH_Keyframes_First)
            droid_MH_Second = SDroid(args)
            loop_detect.Give_Data(droid_MH_Second, MH_Keyframes_Second)

            #获取loop照片路径
            Loop_directories_list = loop_detect.get_cam0_subdirectories(single_loop_folder_path)

            
            OriginalCoords = np.empty((0,7))
            NewCoords = np.empty((0,7))
            W2ToW1 = torch.empty((0, 7))
            
            idx_accumulator = 0
            for key, value_list in sequence_data.items():
                if key > 0:
                        idx_accumulator += len(sequence_data[key - 1]) - 1
                for idx, list_item in enumerate(value_list[1:], start=1):  # 从第二个列表项开始
                    corresponding_idx = idx_accumulator + (idx - 1)
                    corresponding_loop_dir = Loop_directories_list[corresponding_idx]  # 因为我们从1开始计数，所以需要-1来获得正确的索引
            
                    #为每个对象添加输入参数并创建对象
                    args.datapath=corresponding_loop_dir
                    args.warmup=len(value_list[0])
                    droid_loop = SDroid(args)

                    #使用value_list中的索引提取数据, 赋予循环对象初值
                    droid_loop.video.poses[:len(value_list[0])] = droid_MH_First.video.poses[torch.tensor(value_list[0])]
                    droid_loop.video.disps[:len(value_list[0])] = droid_MH_First.video.disps[torch.tensor(value_list[0])]
                    for (t, image, intrinsics) in tqdm(loop_detect.image_stream(args.datapath, stereo=args.stereo, stride=1)):
                        droid_loop.track(t, image, intrinsics=intrinsics)
                    #droid_loop.terminate()

                    #计算转换矩阵
                    MH_old_poses = torch.tensor(list_item).tolist()
                    MH_new_poses = list(range(len(value_list[0]), (len(value_list[0]) + len(list_item))))

                    old_numpy = droid_MH_Second.video.poses[MH_old_poses].cpu().numpy()
                    new_numpy = droid_loop.video.poses[MH_new_poses].cpu().numpy()
                    
                    OriginalCoords = np.vstack([OriginalCoords, old_numpy])
                    NewCoords = np.vstack([NewCoords, new_numpy])

                    W2ToW1_element=torch.tensor((lietorch.SE3(droid_MH_Second.video.poses[MH_old_poses])*(lietorch.SE3(droid_loop.video.poses[MH_new_poses]).inv())).data.cpu().numpy())                        
                    W2ToW1 = torch.cat([W2ToW1, W2ToW1_element], 0)

                    del droid_loop
                    torch.cuda.empty_cache()   
            
            while True:
                NewCoords_length = NewCoords.shape[0]
                timestamps = np.arange(NewCoords_length)

                traj_ref = PoseTrajectory3D(
                    positions_xyz=NewCoords[:,:3],
                    orientations_quat_wxyz=NewCoords[:,3:],
                    timestamps=np.array(timestamps))
            
                traj_est = PoseTrajectory3D(
                    positions_xyz=OriginalCoords[:,:3],
                    orientations_quat_wxyz=OriginalCoords[:,3:],
                    timestamps=np.array(timestamps))

            
                result = main_ape.ape(traj_est, traj_ref, est_name='traj', 
                    pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
                
                print(result.stats["std"])
                
                if result.stats["std"]>0.05:
                    W2ToW1, removed_row = loop_detect.remove_outlier_row(W2ToW1)
                    OriginalCoords = loop_detect.remove_row_from_array(OriginalCoords, removed_row)
                    NewCoords = loop_detect.remove_row_from_array(NewCoords, removed_row)
                else:
                    break
            



            
            alignment_transformation_sim3 = result.np_arrays["alignment_transformation_sim3"]
            rotation_matrix = alignment_transformation_sim3[:3, :3]
            translation_vector = alignment_transformation_sim3[:3, 3]
            rotation_quaternion = R.from_matrix(rotation_matrix).as_quat() 
            combined_array = np.hstack([translation_vector, rotation_quaternion])

            # 将 NumPy 数组转换为 PyTorch 张量
            T = torch.from_numpy(combined_array)

            # 改变张量的数据类型为 torch.float32
            T = T.to(dtype=torch.float32)
            print(T)

            #转换坐标并储存，节省了空间
            New_MH_Keyframes_Second_pose = loop_detect.getTransformedPoses(T, kd_folder2)
            new_pose_file_path = os.path.join(kd_folder2, "newPosBeforeBackend.npy")
            np.save(new_pose_file_path, New_MH_Keyframes_Second_pose)  # 使用完整路径保存

            #清理CUDA缓存
            del droid_MH_First
            del droid_MH_Second
            torch.cuda.empty_cache()

            #读取数据并且存储到非显卡内存中
            MH_First={}
            MH_Second={}
            loop_detect.Give_Data_To_Var(MH_First,MH_Second,kd_folder1,kd_folder2)

            droid_Pairs = SDroid(args)
            droid_Pairs.video.images[:MH_First['images'].shape[0] + MH_Second['images'].shape[0]] = torch.cat([torch.from_numpy(MH_First['images']), torch.from_numpy(MH_Second['images'])], dim=0)
            droid_Pairs.video.poses[:MH_First['poses'].shape[0] + MH_Second['poses'].shape[0]] = torch.cat([torch.from_numpy(MH_First['poses']), torch.from_numpy(MH_Second['poses'])], dim=0)
            droid_Pairs.video.disps[:MH_First['disps'].shape[0] + MH_Second['disps'].shape[0]] = torch.cat([torch.from_numpy(MH_First['disps']), torch.from_numpy(MH_Second['disps'])], dim=0)
            droid_Pairs.video.fmaps[:MH_First['fmaps'].shape[0] + MH_Second['fmaps'].shape[0]] = torch.cat([torch.from_numpy(MH_First['fmaps']), torch.from_numpy(MH_Second['fmaps'])], dim=0)
            droid_Pairs.video.inps[:MH_First['inps'].shape[0] + MH_Second['inps'].shape[0]] = torch.cat([torch.from_numpy(MH_First['inps']), torch.from_numpy(MH_Second['inps'])], dim=0)
            droid_Pairs.video.nets[:MH_First['nets'].shape[0] + MH_Second['nets'].shape[0]] = torch.cat([torch.from_numpy(MH_First['nets']), torch.from_numpy(MH_Second['nets'])], dim=0)
            droid_Pairs.video.intrinsics[:MH_First['intrinsics'].shape[0] + MH_Second['intrinsics'].shape[0]] = torch.cat([torch.from_numpy(MH_First['intrinsics']), torch.from_numpy(MH_Second['intrinsics'])], dim=0)

            droid_Pairs.video.counter.value=MH_First['poses'].shape[0] + MH_Second['poses'].shape[0]
            droid_Pairs.terminate()

            selected_files = ['disps.npy', 'fmaps.npy', 'images.npy', 'inps.npy', 'intrinsics.npy', 'nets.npy', 'tstamps.npy']
            loop_detect.copy_to_transformed_file(transfomed_pose_path, kd_folder2, selected_files)
            after_Back_pose = droid_Pairs.video.poses[len_First:len_First+len_Second].cpu().numpy()
            kd_folder_name = os.path.basename(kd_folder2)
            after_back_pose_file_path = os.path.join(transfomed_pose_path, kd_folder_name, "poses.npy")
            np.save(after_back_pose_file_path, after_Back_pose) 

            del droid_Pairs
            torch.cuda.empty_cache()


            


