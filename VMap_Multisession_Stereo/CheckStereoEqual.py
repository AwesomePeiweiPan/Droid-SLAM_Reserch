import os
import shutil

def copy_matching_files(source_folder_1, source_folder_2, destination_folder):
    # 创建目标文件夹，如果它不存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 获取第一个文件夹中所有文件的文件名
    files_in_first_folder = set(os.listdir(source_folder_1))

    # 遍历第二个文件夹
    for file_name in os.listdir(source_folder_2):
        # 检查是否在第一个文件夹中存在相同的文件名
        if file_name in files_in_first_folder:
            # 构建源和目标文件的完整路径
            source_file = os.path.join(source_folder_2, file_name)
            destination_file = os.path.join(destination_folder, file_name)

            # 复制文件
            shutil.copy(source_file, destination_file)

# 指定文件夹路径
source_folder_1 = '/home/peiweipan/Projects/DroidSlam/VMapData/OriginalData/site2/V03/cam0/data/'  # 更新为第一个文件夹的路径
source_folder_2 = '/home/peiweipan/Projects/DroidSlam/VMapData/OriginalData/site2/V03/cam1/data/' # 更新为第二个文件夹的路径
destination_folder = '/home/peiweipan/Projects/DroidSlam/VMapData/OriginalData/site2/V03/cam3/data/' # 更新为目标文件夹的路径

# 执行函数
copy_matching_files(source_folder_1, source_folder_2, destination_folder)
