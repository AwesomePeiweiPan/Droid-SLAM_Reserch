import torch
import torch.nn.functional as F

from lietorch import SE3, Sim3

MIN_DEPTH = 0.2

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1)

def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())

    return torch.stack([x, y], dim=-1)

def iproj(disps, intrinsics, jacobian=False):  #  X0, Jz = iproj(depths[:,ii], intrinsics[:,ii], jacobian=jacobian)
    """ pinhole camera inverse projection """
    ht, wd = disps.shape[2:]   
    fx, fy, cx, cy = extract_intrinsics(intrinsics) #从[1,22,4]到[1,22,1,1]
    
    y, x = torch.meshgrid(  #y,x都是[48,64], y为每一行从0到47；x是每一列从0到63
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    i = torch.ones_like(disps) #全是1，[1,22,48,64]
    X = (x - cx) / fx       #归一化坐标计算X，这里没有提供Z值，否则就是相机坐标系下的3D坐标，酷！
    Y = (y - cy) / fy       #归一化坐标计算Y，这里没有提供Z值，否则就是相机坐标系下的3D坐标，酷！
    pts = torch.stack([X, Y, i, disps], dim=-1)     #pts[1,22,48,64,4] 沿着新的最后一个维度，放置X,Y,i,disps

    if jacobian:
        J = torch.zeros_like(pts)
        J[...,-1] = 1.0 #J的最后一个维度设置为1
        return pts, J

    return pts, None

def proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """ pinhole camera projection """
    fx, fy, cx, cy = extract_intrinsics(intrinsics) #[1,22,1,1]
    X, Y, Z, D = Xs.unbind(dim=-1)  #Xs为[1，22，48，64，4] 放置X,Y,i,disps, 为[1,22,48,64]

    Z = torch.where(Z < 0.5*MIN_DEPTH, torch.ones_like(Z), Z)
    d = 1.0 / Z

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        coords = torch.stack([x, y, D*d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)       #返回 [1,22,48,64,2]

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
                # o,     o,    -D*d*d,  d,
        ], dim=-1).view(B, N, H, W, 2, 4)

        return coords, proj_jac

    return coords, None

def actp(Gij, X0, jacobian=False):
    """ action on point cloud """
    X1 = Gij[:,:,None,None] * X0
    
    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)  #把X1的最后一个维度拆开
        o = torch.zeros_like(d)
        B, N, H, W = d.shape            #[1,22,48,64]

        if isinstance(Gij, SE3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,
                o,  d,  o, -Z,  o,  X, 
                o,  o,  d,  Y, -X,  o,
                o,  o,  o,  o,  o,  o,
            ], dim=-1).view(B, N, H, W, 4, 6)    #【1，22，48，64，4，6】

        elif isinstance(Gij, Sim3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,  X,
                o,  d,  o, -Z,  o,  X,  Y,
                o,  o,  d,  Y, -X,  o,  Z,
                o,  o,  o,  o,  o,  o,  o
            ], dim=-1).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None

def projective_transform(poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False):    # projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)
    """ map points from ii->jj """

    # inverse project (pinhole) X0为[1,22,48,64,4] 沿着新的最后一个维度，放置X,Y,i,disps i投影到j
    X0, Jz = iproj(depths[:,ii], intrinsics[:,ii], jacobian=jacobian)
    
    # transform [1,22,7]，每7个元素的向量描述了从ii位姿到jj位姿的相对变换
    Gij = poses[:,jj] * poses[:,ii].inv()
    #Gij.data[:, ii == jj] 选择了 Gij 张量中所有批次和那些 ii 等于 jj 的相机位姿。这通常用于表示相对于自身（即从相机 i 到相机 i）的相对位姿。
    Gij.data[:,ii==jj] = torch.as_tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda")
    X1, Ja = actp(Gij, X0, jacobian=jacobian)   #传递给X1 

    # project (pinhole)                        
    x1, Jp = proj(X1, intrinsics[:,jj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too close to camera 代表深度的第三维；新的张量 valid，值为 1 的位置表示 X1 和 X0 在那个位置的深度都大于 MIN_DEPTH，值为 0 的位置表示至少有一个不满足这个条件
    valid = ((X1[...,2] > MIN_DEPTH) & (X0[...,2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1) #最后加一维度，从[1,22,48,64]到[1,22,48,64,1]

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:,:,None,None,None].adjT(Jj)

        Jz = Gij[:,:,None,None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid

def induced_flow(poses, disps, intrinsics, ii, jj):
    """ optical flow induced by camera motion """

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, ii, jj, False)

    return coords1[...,:2] - coords0, valid

