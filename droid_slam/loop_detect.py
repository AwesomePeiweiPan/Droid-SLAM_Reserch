import os
import shutil

import sys
sys.path.append('droid_slam')
import numpy as np
import torch
import cv2
import os
import glob 
import torch.nn.functional as F
import re
import lietorch
from pathlib import Path

#清空文件夹
def clear_directory(path):
    if os.path.exists(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

#Euroc 数据文件读取
def Euroc_image_stream(datapath, image_size=[320, 512], stereo=False, stride=1):
    """ image generator """

    K_l = np.array([458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
    R_l = np.array([
         0.999966347530033, -0.001422739138722922, 0.008079580483432283, 
         0.001365741834644127, 0.9999741760894847, 0.007055629199258132, 
        -0.008089410156878961, -0.007044357138835809, 0.9999424675829176
    ]).reshape(3,3)
    
    P_l = np.array([435.2046959714599, 0, 367.4517211914062, 0,  0, 435.2046959714599, 252.2008514404297, 0,  0, 0, 1, 0]).reshape(3,4)
    map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3,:3], (752, 480), cv2.CV_32F)
    
    K_r = np.array([457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1]).reshape(3,3)
    d_r = np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0]).reshape(5)
    R_r = np.array([
         0.9999633526194376, -0.003625811871560086, 0.007755443660172947, 
         0.003680398547259526, 0.9999684752771629, -0.007035845251224894, 
        -0.007729688520722713, 0.007064130529506649, 0.999945173484644
    ]).reshape(3,3)
    
    P_r = np.array([435.2046959714599, 0, 367.4517211914062, -47.90639384423901, 0, 435.2046959714599, 252.2008514404297, 0, 0, 0, 1, 0]).reshape(3,4)
    map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3,:3], (752, 480), cv2.CV_32F)

    intrinsics_vec = [435.2046959714599, 435.2046959714599, 367.4517211914062, 252.2008514404297]
    ht0, wd0 = [480, 752]

    # read all png images in folder
    images_left = sorted(glob.glob(os.path.join(datapath, '*.png')))[::stride]
    images_right = [x.replace('cam0', 'cam1') for x in images_left]

    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        if stereo and not os.path.isfile(imgR):
            continue
        #取最后一个/后面的数字，并且取.png前面的数字
        tstamp = float(imgL.split('/')[-1][:-4])  
        #重新映射图像的像素。这个函数常用于校正畸变的图像，例如鱼眼镜头拍摄的图像，并使用双线性插值处理图像      
        images = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
        if stereo:
            images += [cv2.remap(cv2.imread(imgR), map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]
        
        images = torch.from_numpy(np.stack(images, 0))   #把images按照第0维度叠加，变成[2,480,752,3]
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False) #双线性插值改变图片的大小到image_size
        
        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        yield stride*t, images, intrinsics

#用于提取关键帧图像到指定的文件夹中
def extract_images_by_timestamp(src_folder: str, dst_folder: str, tstamp: torch.Tensor):
    # 确保目标文件夹存在
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    # 如果目标文件夹中已经有数据，则清除所有的图片
    existing_files = [f for f in os.listdir(dst_folder) if f.endswith('.png')]
    for file in existing_files:
        os.remove(os.path.join(dst_folder, file))

    # 1. 列出文件夹中所有的.png图像
    all_files = [f for f in os.listdir(src_folder) if f.endswith('.png')]

    # 2. 使用图像的文件名中的数字进行排序
    sorted_files = sorted(all_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

    # 将tstamp tensor转换为list
    tstamp_list = tstamp.squeeze().int().tolist()

    # 3. 使用tstamp中的值提取相应的图像
    for idx in tstamp_list:
        src_path = os.path.join(src_folder, sorted_files[idx])
        dst_path = os.path.join(dst_folder, sorted_files[idx])
        shutil.copy2(src_path, dst_path)

#传统的文件读取函数，但是文件名有改动
def image_stream(datapath, image_size=[320, 512], stereo=False, stride=1):
    
    """ image generator """

    K_l = np.array([458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
    R_l = np.array([
         0.999966347530033, -0.001422739138722922, 0.008079580483432283, 
         0.001365741834644127, 0.9999741760894847, 0.007055629199258132, 
        -0.008089410156878961, -0.007044357138835809, 0.9999424675829176
    ]).reshape(3,3)
    
    P_l = np.array([435.2046959714599, 0, 367.4517211914062, 0,  0, 435.2046959714599, 252.2008514404297, 0,  0, 0, 1, 0]).reshape(3,4)
    map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3,:3], (752, 480), cv2.CV_32F)
    
    K_r = np.array([457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1]).reshape(3,3)
    d_r = np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0]).reshape(5)
    R_r = np.array([
         0.9999633526194376, -0.003625811871560086, 0.007755443660172947, 
         0.003680398547259526, 0.9999684752771629, -0.007035845251224894, 
        -0.007729688520722713, 0.007064130529506649, 0.999945173484644
    ]).reshape(3,3)
    
    P_r = np.array([435.2046959714599, 0, 367.4517211914062, -47.90639384423901, 0, 435.2046959714599, 252.2008514404297, 0, 0, 0, 1, 0]).reshape(3,4)
    map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3,:3], (752, 480), cv2.CV_32F)

    intrinsics_vec = [435.2046959714599, 435.2046959714599, 367.4517211914062, 252.2008514404297]
    ht0, wd0 = [480, 752]

    # read all png images in folder
    images_left = sorted(glob.glob(os.path.join(datapath, '*.png')), key=lambda x: int(os.path.basename(x).split('_')[0]))[::stride]
    images_right = [x.replace('cam0', 'cam1') for x in images_left]

    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        if stereo and not os.path.isfile(imgR):
            continue 
        #重新映射图像的像素。这个函数常用于校正畸变的图像，例如鱼眼镜头拍摄的图像，并使用双线性插值处理图像      
        images = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
        if stereo:
            images += [cv2.remap(cv2.imread(imgR), map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]
        
        images = torch.from_numpy(np.stack(images, 0))   #把images按照第0维度叠加，变成[2,480,752,3]
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False) #双线性插值改变图片的大小到image_size
        
        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        yield stride*t, images, intrinsics

#读取Sequence.txt
def get_SequenceData(file_path):
    data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_block = None
    for line in lines:
        # Check if the line is a block header (e.g., "0")
        if re.match(r'^\d+$', line.strip()):
            current_block = int(line.strip())
            data[current_block] = []
            continue

        # For lines with [Order] or [ReverseOrder] labels
        match_with_label = re.search(r': ([\d\s]+)\s*\[(Order|ReverseOrder)\]\s*', line)
        if match_with_label:
            numbers = list(map(int, match_with_label.group(1).split()))
            order = match_with_label.group(2)
            if order == "ReverseOrder":
                numbers = numbers[::-1]
            data[current_block].append(numbers)
            continue

        # For lines without labels (e.g., "First Images Series:")
        match_without_label = re.search(r': ([\d\s]+)\s*$', line)
        if match_without_label:
            numbers = list(map(int, match_without_label.group(1).split()))
            data[current_block].append(numbers)

    return data

#读取MH保存的所有数据
def load_reconstruction_MH(reconstruction_path):
    data = {}

    # Load each numpy array from the saved path
    data['tstamps'] = np.load(f"reconstructions/{reconstruction_path}/tstamps.npy")
    data['images'] = np.load(f"reconstructions/{reconstruction_path}/images.npy")
    data['disps'] = np.load(f"reconstructions/{reconstruction_path}/disps.npy")
    data['poses'] = np.load(f"reconstructions/{reconstruction_path}/poses.npy")
    data['intrinsics'] = np.load(f"reconstructions/{reconstruction_path}/intrinsics.npy")
    data['fmaps'] = np.load(f"reconstructions/{reconstruction_path}/fmaps.npy")
    data['inps'] = np.load(f"reconstructions/{reconstruction_path}/inps.npy")
    data['nets'] = np.load(f"reconstructions/{reconstruction_path}/nets.npy")

    return data

def load_reconstruction(reconstruction_path):
    data = {}

    # Load each numpy array from the saved path
    data['tstamps'] = np.load(os.path.join(reconstruction_path, 'tstamps.npy'))
    data['images'] = np.load(os.path.join(reconstruction_path, 'images.npy'))
    data['disps'] = np.load(os.path.join(reconstruction_path, 'disps.npy'))
    data['poses'] = np.load(os.path.join(reconstruction_path, 'poses.npy'))
    data['intrinsics'] = np.load(os.path.join(reconstruction_path, 'intrinsics.npy'))
    data['fmaps'] = np.load(os.path.join(reconstruction_path, 'fmaps.npy'))
    data['inps'] = np.load(os.path.join(reconstruction_path, 'inps.npy'))
    data['nets'] = np.load(os.path.join(reconstruction_path, 'nets.npy'))

    return data


#通过MH保存的数据给对象赋值
def Give_Data(droid_MH, MH_Keyframes):
    droid_MH.video.tstamp[:MH_Keyframes['tstamps'].shape[0]] = torch.from_numpy(MH_Keyframes['tstamps'])
    droid_MH.video.poses[:MH_Keyframes['poses'].shape[0]] = torch.from_numpy(MH_Keyframes['poses'])
    droid_MH.video.images[:MH_Keyframes['images'].shape[0]] = torch.from_numpy(MH_Keyframes['images'])
    droid_MH.video.disps[:MH_Keyframes['disps'].shape[0]] = torch.from_numpy(MH_Keyframes['disps'])
    droid_MH.video.intrinsics[:MH_Keyframes['intrinsics'].shape[0]] = torch.from_numpy(MH_Keyframes['intrinsics'])
    droid_MH.video.fmaps[:MH_Keyframes['fmaps'].shape[0]] = torch.from_numpy(MH_Keyframes['fmaps'])
    droid_MH.video.inps[:MH_Keyframes['inps'].shape[0]] = torch.from_numpy(MH_Keyframes['inps'])
    droid_MH.video.nets[:MH_Keyframes['nets'].shape[0]] = torch.from_numpy(MH_Keyframes['nets'])

#给出图像路径
def get_cam0_subdirectories(base_path):
    cam0_dirs = []  # List to store directories with "cam0" in them

    # Walk through each directory
    for root, dirs, files in os.walk(base_path):
        # If there are no subdirectories in the current root
        if not dirs and 'cam0' in root:
            cam0_dirs.append(root)

    # Custom sorting function
    def sort_key(path):
        # Extract all digits from the path and convert to integer
        numbers = [int(n) for n in filter(str.isdigit, os.path.normpath(path).replace(os.path.sep, ''))]
        return numbers

    cam0_dirs.sort(key=sort_key)
    return cam0_dirs

#使用四分数法取得转换平均值
def compute_filtered_mean(tensor):
    """
    Compute the average of each column of the tensor after removing outliers using the IQR method.

    :param tensor: Input tensor (2D).
    :return: A tensor containing the filtered mean of each column.
    """
    # 初始化一个空的张量来存储每列的平均值
    column_means_filtered = torch.empty((1, tensor.size(1)))

    for i in range(tensor.size(1)):
        column = tensor[:, i]
        Q1 = torch.quantile(column, 0.25)
        Q3 = torch.quantile(column, 0.75)
        IQR = Q3 - Q1
        
        # 定义极限值范围
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 过滤数据
        filtered_values = column[(column >= lower_bound) & (column <= upper_bound)]
        
        # 计算平均值并保存
        column_means_filtered[0, i] = filtered_values.mean()

    return column_means_filtered

#打印转换矩阵结果
def printTransformMatrix(W2ToW1):
    print("All Transformation Matrix")
    for row in W2ToW1:
        print(row)    
    column_means = compute_filtered_mean(W2ToW1)
    column_means = column_means.unsqueeze(0)
    print()
    print("Mean Transformation Matrix")
    print(column_means)

#通过存储的中间文件给整个第二个地图新的坐标
def getTransformedPoses_2(reconstruction_path_T, reconstruction_path_Poses):
    Data = {}
    Data['Transformation'] = np.load(f"reconstructions/{reconstruction_path_T}/T_MH01_MH02.npy")
    Data['poses'] = np.load(f"reconstructions/{reconstruction_path_Poses}/poses.npy")
    T = torch.from_numpy(Data['Transformation'])
    poses = torch.from_numpy(Data['poses'])

    length = len(Data['poses'])
    sequence = list(range(length))
    T_tensor = T.repeat(length, 1)
    T_l=lietorch.SE3(T_tensor)
    MH_poses = lietorch.SE3(poses)
    return torch.tensor((T_l[sequence].inv()*(MH_poses[sequence])).data.cpu().numpy())

#直接转换第二个地图新的坐标
def getTransformedPoses(T, second_coor):
    Data = {}
    Data['poses'] = np.load(os.path.join(second_coor, 'poses.npy'))
    poses = torch.from_numpy(Data['poses'])

    length = len(Data['poses'])
    sequence = list(range(length))
    T_tensor = T.repeat(length, 1)
    T_l=lietorch.SE3(T_tensor)
    MH_poses = lietorch.SE3(poses)
    new_pose = torch.tensor((T_l[sequence].inv()*(MH_poses[sequence])).data.cpu().numpy())
    return new_pose

def getTransformedPoses2(T, second_coor):
    Data = {}
    Data['poses'] = np.load(os.path.join(second_coor, 'poses.npy'))
    poses = torch.from_numpy(Data['poses'])

    length = len(Data['poses'])
    sequence = list(range(length))
    T_tensor = T.repeat(length, 1)
    T_l=lietorch.SE3(T_tensor)
    MH_poses = lietorch.SE3(poses)
    new_pose = torch.tensor((T_l[sequence]*(MH_poses[sequence])).data.cpu().numpy())
    return new_pose



def save_reconstruction_for_Vis(droid, reconstruction_path):

    from pathlib import Path

    t = droid.video.counter.value
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)

def Give_Data_To_Var(MH01, MH02, MH01_Path, MH02_Path):
    MH01['tstamps'] = np.load(os.path.join(MH01_Path, 'tstamps.npy'))
    MH01['images'] = np.load(os.path.join(MH01_Path, 'images.npy'))
    MH01['disps'] = np.load(os.path.join(MH01_Path, 'disps.npy'))
    MH01['poses'] = np.load(os.path.join(MH01_Path, 'poses.npy'))
    MH01['fmaps'] = np.load(os.path.join(MH01_Path, 'fmaps.npy'))
    MH01['inps'] = np.load(os.path.join(MH01_Path, 'inps.npy'))
    MH01['nets'] = np.load(os.path.join(MH01_Path, 'nets.npy'))
    MH01['intrinsics'] = np.load(os.path.join(MH01_Path, 'intrinsics.npy'))

    MH02['tstamps'] = np.load(os.path.join(MH02_Path, 'tstamps.npy'))
    MH02['images'] = np.load(os.path.join(MH02_Path, 'images.npy'))
    MH02['disps'] = np.load(os.path.join(MH02_Path, 'disps.npy'))
    MH02['poses'] = np.load(os.path.join(MH02_Path, 'newPosBeforeBackend.npy'))
    MH02['fmaps'] = np.load(os.path.join(MH02_Path, 'fmaps.npy'))
    MH02['inps'] = np.load(os.path.join(MH02_Path, 'inps.npy'))
    MH02['nets'] = np.load(os.path.join(MH02_Path, 'nets.npy'))
    MH02['intrinsics'] = np.load(os.path.join(MH02_Path, 'intrinsics.npy'))


def copy_to_transformed_file(new_folder_path, source_folder, selected_files):
    # 转换为 Path 对象
    new_folder_path = Path(new_folder_path)
    source_folder = Path(source_folder)

    # 创建与 kd_folder2 相同名称的子文件夹
    target_folder = new_folder_path / source_folder.name
    target_folder.mkdir(parents=True, exist_ok=True)

    # 遍历并复制选定的文件
    for file_name in selected_files:
        source_file = source_folder / file_name
        target_file = target_folder / file_name

        if source_file.exists():
            shutil.copy(source_file, target_file)
        else:
            print(f"File not found: {source_file}")


def clear_all_subdirectories(path):
    # 检查路径是否存在
    if os.path.exists(path):
        # 遍历文件夹中的所有子文件夹
        for folder_name in os.listdir(path):
            folder_path = os.path.join(path, folder_name)
            try:
                # 删除文件夹及其所有内容
                if os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)
            except Exception as e:
                print(f'Failed to delete {folder_path}. Reason: {e}')
    else:
        print(f"The path does not exist: {path}")


def remove_outlier_row(tensor, tensor2):
    

    # 计算每一行与平均值的差异
    diffs = (tensor[:, :3] - tensor2[:, :3]).norm(dim=1)

    # 找到差异最大的行
    max_diff_index = torch.argmax(diffs)

    # 移除该行
    tensor = torch.cat([tensor[:max_diff_index], tensor[max_diff_index + 1:]])

    return tensor, max_diff_index.item()

def remove_row_from_array(array, row_index):
    # 移除指定索引的行
    new_array = np.delete(array, row_index, axis=0)
    return new_array









