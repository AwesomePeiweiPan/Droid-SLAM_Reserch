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
    images_left = sorted(glob.glob(os.path.join(datapath, '*.png')))[::stride]
    images_right = [x.replace('cam0', 'cam1') for x in images_left]

    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        if stereo and not os.path.isfile(imgR):
            continue
        tstamp = float(imgL.split('/')[-1][:-4])        
        images = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
        if stereo:
            images += [cv2.remap(cv2.imread(imgR), map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]
        
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False)
        
        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        yield stride*t, images, intrinsics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to euroc sequence")
    parser.add_argument("--gt", help="path to gt file")
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
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    args.datapath = "/home/peiweipan/Projects/DroidSlam/EurocData/OriginalData/MH01/mav0/cam0/data/"
    M1_path = "/home/peiweipan/Projects/DroidSlam/EurocData/KeyFrames_Improve/KD01/"
    args.gt = "/home/peiweipan/Projects/DroidSlam/data/euroc_groundtruth/MH_01_easy.txt"
    args.stereo = True
    args.disable_vis = True

    M_First = {}
    M_First['tstamps'] = np.load(os.path.join(M1_path, 'tstamps.npy'))
    M_First['poses'] = np.load(os.path.join(M1_path, 'poses.npy'))
    M_First['disps'] = np.load(os.path.join(M1_path, 'disps.npy'))
    M_First['images'] = np.load(os.path.join(M1_path, 'images.npy'))
    M_First['intrinsics'] = np.load(os.path.join(M1_path, 'intrinsics.npy'))
    M_First['fmaps'] = np.load(os.path.join(M1_path, 'fmaps.npy'))
    M_First['inps'] = np.load(os.path.join(M1_path, 'inps.npy'))
    M_First['nets'] = np.load(os.path.join(M1_path, 'nets.npy'))

    droid_MH = Droid(args)
    droid_MH.video.tstamp[:M_First['tstamps'].shape[0]] = torch.from_numpy(M_First['tstamps'])
    droid_MH.video.poses[:M_First['poses'].shape[0]] = torch.from_numpy(M_First['poses'])
    droid_MH.video.disps[:M_First['disps'].shape[0]] = torch.from_numpy(M_First['disps'])
    droid_MH.video.images[:M_First['images'].shape[0]] = torch.from_numpy(M_First['images'])
    droid_MH.video.intrinsics[:M_First['intrinsics'].shape[0]] = torch.from_numpy(M_First['intrinsics'])
    droid_MH.video.fmaps[:M_First['fmaps'].shape[0]] = torch.from_numpy(M_First['fmaps'])
    droid_MH.video.inps[:M_First['inps'].shape[0]] = torch.from_numpy(M_First['inps'])
    droid_MH.video.nets[:M_First['nets'].shape[0]] = torch.from_numpy(M_First['nets'])

    droid_MH.video.counter.value=M_First['tstamps'].shape[0]

    traj_est = droid_MH.terminate_eva(image_stream(args.datapath, stereo=args.stereo, stride=1))

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    images_list = sorted(glob.glob(os.path.join(args.datapath, '*.png')))
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    traj_est = PoseTrajectory3D(
        positions_xyz=1.10 * traj_est[2000:3000,:3],
        orientations_quat_wxyz=traj_est[2000:3000,3:],
        timestamps=np.array(tstamps[2000:3000]))
    



    
    traj_ref = file_interface.read_tum_trajectory_file(args.gt)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    print(result)