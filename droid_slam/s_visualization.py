import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d

from lietorch import SE3
import geom.projective_ops as pops

#CAM_POINTS 和 CAM_LINES 是描述相机模型的3D结构和连接的全局常数数组
#用于在3D环境中可视化相机的姿态。
CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

#此函数实现了图像的白平衡调整。白平衡是摄影术语，意在调整图片中的颜色使其更自然
def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

#该函数用于创建一个3D相机模型，可以在3D环境中表示相机的位置和方向
def create_camera_actor(g, scale=0.2):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

#此函数用于从numpy数组中创建一个点云，表示3D空间中的点。
def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def S_droid_visualization(video, device="cuda:0"):
    torch.cuda.set_device(device)
    S_droid_visualization.video = video
    S_droid_visualization.cameras = {}
    S_droid_visualization.points = {}
    S_droid_visualization.warmup = 8
    S_droid_visualization.scale = 1.0
    S_droid_visualization.ix = 0

    S_droid_visualization.filter_thresh = 0.005

    def increase_filter(vis):
        S_droid_visualization.filter_thresh *= 2
        with S_droid_visualization.video.get_lock():
            S_droid_visualization.video.dirty[:S_droid_visualization.video.counter.value] = True

    def decrease_filter(vis):
        S_droid_visualization.filter_thresh *= 0.5
        with S_droid_visualization.video.get_lock():
            S_droid_visualization.video.dirty[:S_droid_visualization.video.counter.value] = True
    
    #一个回调函数，它在每一帧中都被调用，用于更新3D视图中的内容。
    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

        with torch.no_grad():

            with video.get_lock():
                t = video.counter.value 
                dirty_index, = torch.where(video.dirty.clone())
                dirty_index = dirty_index

            if len(dirty_index) == 0:
                return

            video.dirty[dirty_index] = False

            # convert poses to 4x4 matrix
            poses = torch.index_select(video.poses, 0, dirty_index)
            disps = torch.index_select(video.disps, 0, dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()  

            images = torch.index_select(video.images, 0, dirty_index)
            images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
            points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()      

            thresh = S_droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))

            count = droid_backends.depth_filter(
                video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)

            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))

            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()

                if ix in S_droid_visualization.cameras:
                    vis.remove_geometry(S_droid_visualization.cameras[ix])
                    del S_droid_visualization.cameras[ix]

                if ix in S_droid_visualization.points:
                    vis.remove_geometry(S_droid_visualization.points[ix])
                    del S_droid_visualization.points[ix]
                
                    ### add camera actor ###
                    cam_actor = create_camera_actor(True)
                    cam_actor.transform(pose)
                    vis.add_geometry(cam_actor)
                    S_droid_visualization.cameras[ix] = cam_actor

                    mask = masks[i].reshape(-1)
                    pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                    clr = images[i].reshape(-1, 3)[mask].cpu().numpy()

                    print("callback")
                    ## add point actor ###
                    point_actor = create_point_actor(pts, clr)
                    vis.add_geometry(point_actor)
                    S_droid_visualization.points[ix] = point_actor
            
                if len(S_droid_visualization.cameras) >= S_droid_visualization.warmup:
                    cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

                S_droid_visualization.ix += 1
                vis.poll_events()
                vis.update_renderer()

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_filter)
    vis.register_key_callback(ord("A"), decrease_filter)

    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.run()
    vis.destroy_window()



























