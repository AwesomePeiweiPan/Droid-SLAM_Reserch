motion_filter.py
    if delta.norm(dim=-1).mean().item() > self.thresh:
        改变thresh的值，最好使得每一张图片都可以被当作关键帧
    Id = lietorch.SE3.Identity(1,).data.squeeze()
        位置初始值改变
depth_video.py
    Gs = lietorch.SE3(self.poses[None])
        初始化的时候改变