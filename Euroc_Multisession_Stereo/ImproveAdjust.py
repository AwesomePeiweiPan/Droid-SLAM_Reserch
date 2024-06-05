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
from evo.core.trajectory import PosePath3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from scipy.spatial.transform import Rotation as R


group_sequence_path = '/home/peiweipan/fbow/Euroc_Data_more/GroupSequence/GroupSequence.txt' 
loop_folder_path = '/home/peiweipan/Projects/DroidSlam/Euroc_Data/Loop_more'  
keyframe_data_file_path = '/home/peiweipan/Projects/DroidSlam/Euroc_Data/KeyFrames_more/' 
transfomed_pose_path = "/home/peiweipan/Projects/DroidSlam/Euroc_Data/TransformedKeyPos_more/"  # 替换为新文件夹的路径



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
    parser.add_argument("--Good", action="store_true")
    args = parser.parse_args()
    #spawn启动更加稳定
    torch.multiprocessing.set_start_method('spawn')

    ###设定默认参数
    args.stereo = True
    args.disable_vis = True
    args.Good = False

    outer_loop_counter = 0  # 追踪外循环的次数

    loop_detect.clear_all_subdirectories(transfomed_pose_path)
    is_first_iteration = True  
    with open(group_sequence_path, 'r') as file:
        for line in file:
            
            outer_loop_counter += 1  # 每次外循环迭代时增加计数
            good_point = 0  # 确保每次外循环开始时good_point重置为0

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
            #droid_MH_Second = SDroid(args)
            #loop_detect.Give_Data(droid_MH_Second, MH_Keyframes_Second)

            #获取loop照片路径
            Loop_directories_list = loop_detect.get_cam0_subdirectories(single_loop_folder_path)

            
            OriginalCoords = np.empty((0,7))
            NewCoords = np.empty((0,7))
            W2ToW1 = torch.empty((0, 7))
            
            loop_count = 0
            good_point = 0
            first_order = False
            exit_loop = False 
            args.Good = False
            skip_next_iteration = False


            idx_accumulator = 0
            for key, value_list in sequence_data.items():
                if exit_loop:  # 检查是否需要退出外层循环
                        break       
                if key > 0:
                        idx_accumulator += len(sequence_data[key - 1]) - 1

                if len(value_list[0:][0]) < 30:
                    print("参照组中的图片数量不够，放弃")
                    continue


                for idx, list_item in enumerate(value_list[1:], start=1):

                    if skip_next_iteration:
                        skip_next_iteration = False  # 重置标志
                        continue  # 跳过本次循环
                    
                    loop_count += 1
                    
                    # 从第二个列表项开始
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
                        if t < 80:
                            droid_loop.track(t, image, intrinsics=intrinsics)
                        else:
                             break

                    badT = droid_loop.frontend.badT
                    print(f"第 {outer_loop_counter} 次地图之间的匹配，匹配失败的图片张数为：{len(badT)}")
                    if (len(badT)>4):
                         print("这一组图片匹配失败，换到下一组")
                         del droid_loop
                         torch.cuda.empty_cache()


                    if loop_count == 1 and len(badT) < 5:
                        del droid_loop
                        torch.cuda.empty_cache()

                        droid_loop = SDroid(args)
                        args.Good = True


                        droid_loop = SDroid(args)
                        droid_loop.video.poses[:len(value_list[0])] = droid_MH_First.video.poses[torch.tensor(value_list[0])]
                        droid_loop.video.disps[:len(value_list[0])] = droid_MH_First.video.disps[torch.tensor(value_list[0])]
                        for (t, image, intrinsics) in tqdm(loop_detect.image_stream(args.datapath, stereo=args.stereo, stride=1)):
                            droid_loop.track(t, image, intrinsics=intrinsics) 

                        droid_loop.terminate()

                        


                        
                        good_point += 1
                        is_increasing = all(x < y for x, y in zip(list_item, list_item[1:]))
                        if is_increasing:
                            first_order = True
                            images1 = droid_loop.video.images[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()
                            poses1 = droid_loop.video.poses[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()
                            disps1 = droid_loop.video.disps[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()
                            intrinsics1 = droid_loop.video.intrinsics[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()
                            del droid_loop
                            torch.cuda.empty_cache()
                        else:
                            images1 = droid_loop.video.images[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()[::-1]
                            poses1 = droid_loop.video.poses[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()[::-1]
                            disps1 = droid_loop.video.disps[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()[::-1]
                            intrinsics1 = droid_loop.video.intrinsics[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()[::-1]
                            del droid_loop
                            torch.cuda.empty_cache()
                    
                    if loop_count == 2 and len(badT) < 5:
                            good_point += 1
                            del droid_loop
                            torch.cuda.empty_cache()

                            droid_loop = SDroid(args)
                            args.Good = True


                            droid_loop = SDroid(args)
                            droid_loop.video.poses[:len(value_list[0])] = droid_MH_First.video.poses[torch.tensor(value_list[0])]
                            droid_loop.video.disps[:len(value_list[0])] = droid_MH_First.video.disps[torch.tensor(value_list[0])]
                            for (t, image, intrinsics) in tqdm(loop_detect.image_stream(args.datapath, stereo=args.stereo, stride=1)):
                                droid_loop.track(t, image, intrinsics=intrinsics) 

                            droid_loop.terminate()
                            
                            
                            

                            is_increasing = all(x < y for x, y in zip(list_item, list_item[1:]))
                            if is_increasing:
                                images2 = droid_loop.video.images[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()
                                poses2 = droid_loop.video.poses[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()
                                disps2 = droid_loop.video.disps[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()
                                intrinsics2 = droid_loop.video.intrinsics[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()
                                del droid_loop
                                torch.cuda.empty_cache()
                            else:
                                images2 = droid_loop.video.images[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()[::-1]
                                poses2 = droid_loop.video.poses[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()[::-1]
                                disps2 = droid_loop.video.disps[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()[::-1]
                                intrinsics2 = droid_loop.video.intrinsics[len(value_list[0]):(len(value_list[0]) + len(list_item))].cpu().numpy()[::-1]
                                del droid_loop
                                torch.cuda.empty_cache()

                    if good_point==2 and first_order == True:
                        images = np.concatenate((images2, images1), axis=0)
                        poses = np.concatenate((poses2, poses1), axis=0)
                        disps = np.concatenate((disps2, disps1), axis=0)
                        intrinsics = np.concatenate((intrinsics2, intrinsics1), axis=0)

                        kd_folder_name = os.path.basename(kd_folder2)
                        folder_path = os.path.join(transfomed_pose_path, kd_folder_name)
                        # 创建文件夹（如果尚不存在）
                        os.makedirs(folder_path, exist_ok=True)         

                        path2 = os.path.join(transfomed_pose_path, kd_folder_name, "poses.npy")
                        np.save(path2, poses) 

                        path2 = os.path.join(transfomed_pose_path, kd_folder_name, "images.npy")
                        np.save(path2, images) 

                        path2 = os.path.join(transfomed_pose_path, kd_folder_name, "intrinsics.npy")
                        np.save(path2, intrinsics) 

                        path2 = os.path.join(transfomed_pose_path, kd_folder_name, "disps.npy")
                        np.save(path2, disps) 

                        exit_loop = True
                        break

                    if good_point==2 and first_order == False:
                        images = np.concatenate((images1, images2), axis=0)
                        poses = np.concatenate((poses1, poses2), axis=0)
                        disps = np.concatenate((disps1, disps2), axis=0)
                        intrinsics = np.concatenate((intrinsics1, intrinsics2), axis=0)

                        kd_folder_name = os.path.basename(kd_folder2)
                        folder_path = os.path.join(transfomed_pose_path, kd_folder_name)
                        # 创建文件夹（如果尚不存在）
                        os.makedirs(folder_path, exist_ok=True)  

                        path2 = os.path.join(transfomed_pose_path, kd_folder_name, "poses.npy")
                        np.save(path2, poses) 

                        path2 = os.path.join(transfomed_pose_path, kd_folder_name, "images.npy")
                        np.save(path2, images) 

                        path2 = os.path.join(transfomed_pose_path, kd_folder_name, "intrinsics.npy")
                        np.save(path2, intrinsics) 

                        path2 = os.path.join(transfomed_pose_path, kd_folder_name, "disps.npy")
                        np.save(path2, disps) 


                        exit_loop = True
                        break
                    
                    
                    if loop_count == 1 and good_point ==0:
                            skip_next_iteration = True  # 设置跳过下一次循环的标志
                            loop_count = 0
                            good_point = 0
                            first_order = False
                            args.Good = False
                            #break
                    elif loop_count == 2:
                            loop_count = 0
                            good_point = 0
                            first_order = False
                            args.Good = False
                            #break


            if good_point != 2:
                print(f"外循环失败于第 {outer_loop_counter} 次迭代，所有的loop都没有衔接上")
                break  # 如果不满足条件则跳出最外层循环                 

            #清理CUDA缓存
            del droid_MH_First
            #del droid_MH_Second
            torch.cuda.empty_cache()

            selected_files = ['fmaps.npy', 'inps.npy', 'nets.npy', 'tstamps.npy']
            loop_detect.copy_to_transformed_file(transfomed_pose_path, kd_folder2, selected_files) 
            torch.cuda.empty_cache()



            


