**输入image格式**
    image[2,3,320,512]
**建立droid实例**
``  args``
        backend_nms=2
        backend_radius=2
        backend_thresh=24.0
        beta=0.3
        buffer=512
        calib=None
        disable_vis=True
        filter_thresh=2.4
        frontend_nms=1
        frontend_radius=2
        frontend_thresh=17.5
        frontend_window=20
        image_size=[320, 512]
        datapath=''/home/peiweipan/Projects/DroidSlam/datasets/MH01/''
        keyframe_thresh=3.5
        reconstruction_path=None
        stereo=True
        stride=3
        t0=0
        upsample=False
        warmup=15
        weights='droid.pth'
``  droid.net(DroidNet)``
        cnet(BasicEncoder)
            conv1(Conv2d(in_channels=3,output_channels=32,kernel_size=(7,7),stride=(2,2),padding=(3,3)))
            conv2(Conv2d(in_channels=128,output_channels=256,kernel_size=(1,1),stride=(1,1),padding=(0,0)))
            layer1(Sequential(...))
            layer2(Sequential(...))
            layer3(Sequential(...))
            norm1(Sequential(...))
            relu1(ReLU)
        fnet(BasicEncoder)
            conv1(Conv2d(in_channels=3,output_channels=32,kernel_size=(7,7),stride=(2,2),padding=(3,3)))
            conv2(Conv2d(in_channels=128,output_channels=128,kernel_size=(1,1),stride=(1,1),padding=(0,0)))
            layer1(Sequential(...))
            layer2(Sequential(...))
            layer3(Sequential(...))
            norm1(InstanceNorm2d(num_features=32,eps=1e-05,momentum=0.1,affine=False,track_running_stats=False))
        update(UpdateModule)
            agg(GraphAgg)
                conv1(Conv2d(in_channels=128,output_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)))
                conv2(Conv2d(in_channels=128,output_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)))
                eta(Sequential(...))
                relu(ReLU)
                upmask(Sequential())
            corr_encoder(Sequential(...))
            delta(Sequential(...))
            flow_encoder(Sequential(...))
            gru(ConvGRU)
                convq(Conv2d(...))
                convq_glo(Conv2d(...))
                convr(Conv2d(...))
                convr_glo(Conv2d(...))
                convz(Conv2d(...))
                convz_glo(Conv2d(...))
                w(Conv2d(...))
            weight(Sequential)
``  droid.video(DepthVideo)``
        ht=320
        wd=512
        counter(Value('i',0))
        ready(Value('i',0))
        dirty(tensor[512]，全为False)
        disps(tensor[512,40,64],全为1)
        disps_sens（tensor[512,40,64],全为0）
        disps_up（tensor[512,320,512],全为0）
        fmaps (tensor[512,2,128,40,64],全为0)
        images(tensor[512,3,320,512],全为0)
        inps(tensor[512,128,40,64],全为0)
        intrinsics(tensor[512,4],全为0)
        nets(tensor[512,128,40,64],全为0)
        poses(tensor[512,7],每一行为(0,0,0,0,0,0,1))
        red(tensor[512],全为False)
        tstamp(tensor[512],全为0)     
``  droid.filterx(MotionFilter)``
        cnet=droid.net.cnet
        fnet=droid.net.fnet
        update=droid.net.update
        video=droid.video
        thresh=2.4
        device="cuda:0"
        count=0
        MEAN(tensor[3,1,1],第0维分别为(0.485, 0.456, 0.406))
        STDV(tensor[3,1,1],第0维分别为(0.509, 0.504, 0.505))
``  droid.frontend(DroidFrontend)``
        video=droid.video
        update_op=droid.net.update
        t0=0
        t1=0
        is_initialized=False
        count=0
        max_age=25
        iters1=4
        iters2=2
        warmup=args.warmup=8
        beta=args.beta=0.3
        frontend_nms=args.frontend_nms=1
        keyframe_thresh=args.keyframe_thresh=3.5
        frontend_window=args.frontend_window=20
        frontend_thresh=self.frontend.thresh=17.5
        frontend_radius=args.frontend_radius=2
        graph(FactorGraph)
            ht=40
            wd=64
            video=droid.video
            update_op=droid.net.update
            device="cuda:0"
            max_factors=48
            corr_impl='volume'
            upsample=False
            coords0(tensor[40,64,2],第:,:,0维度每一行从左往右0-47，第:,:,1维度每一列从上到下0-63)
            ii(tensor[])
            ii_bad(tensor[])
            ii_inac(tensor[])
            jj(tensor[])
            jj_bad(tensor[])
            jj_inac(tensor[])
            age(tensor[])
            corr(None)
            net(None)
            inp(None)
            damping(tensor[512,40,64],全为1e-6)
            target(tensor[1,0,40,64,2],全为0)
            weight(tensor[1,0,40,64,2],全为0)
            target_inac(tensor[1,0,40,64,2],全为0)
            weight_inac(tensor[1,0,40,64,2],全为0)
``  droid.backend(DroidBackend)``
        t0=0
        t1=0
        upsample = args.upsample=False
        beta = args.beta=0.3
        backend_thresh = args.backend_thresh=24.0
        backend_radius = args.backend_radius=2
        backend_nms = args.backend_nms=2
        video=droid.video
        update_op=droid.net.update
``  traj_filler(PoseTrajectoryFiller)``
        cnet=droid.net.cnet
        fnet=droid.net.fnet
        update=droid.net.update
        count=0
        video=droid.video
        device="cuda:0"
        MEAN(tensor[3,1,1],第0维分别为(0.485, 0.456, 0.406))
        STDV(tensor[3,1,1],第0维分别为(0.509, 0.504, 0.505))
**积累关键帧到 counter=15=args.warmup, t=29**
    *droid.filterx.track(tstamp:循环t,image[2,3,320,512]:循环的image,depth:None,intrinsics: tensor[4],值为(256., 256., 256., 192.))*
        when counter=0
``          droid.filterx(MotionFilter)``   增加的元素
                droid.filterx.fmap(tensor[2,128,40,64])
                    通过将输入的 image 第一维改变成R,G,B通道，增加一维度, 归一化/255
                    经过droid.filterx.MEAN和STDV成为inputs[1,2,3,320,512]
                    并且经过droid.filterx._feature_encoder()函数处理得到gmap[2,128,40,64],
                    最后赋值给fmap: 值是满的并且混乱
                droid.filterx.inp(tensor[1,128,40,64])
                    通过将输入的 image 经过上个fmap的处理得到inputs后，取第0维
                    经过droid.filterx._context_encoder(函数处理得到)
                droid.filterx.net(tensor[1,128,40,64])
                    通过将输入的 image 经过上个fmap的处理得到inputs后，取第0维
                    经过droid.filterx._context_encoder(函数处理得到)
                *droid.video.append() 将droid.video.counter使用droid.video.get_lock()锁定,以下是参数列表解释*
                    tstamp: 循环的t
                    image[0]取image的第0维的第0个，不要第一个
                    Id: tensor(7) 只有最后一个值是1，其余为0
                    depth: None
                    intrinsics / 8.0: tensor[4](32.,32.,32.,24)
                    gmap[2,128,40,64]: 值是满的并且混乱,来自于droid.filterx.__feature_encoder(inputs)，包含了image[2,3,320,512]
                    net[0,0]:tensor[40,64]: 值是满的并且混乱,与droid.filterx.net一样
                    inp[0,0]:tensor[40,64]: 值是满的并且混乱,与droid.filterx.inp一样
``                      droid.video(DepthVideo)``   更新的元素
                            droid.video.counter=0 -> droid.video.counter=1
                            droid.video.tstamp(tensor[512],全为0) -> droid.tstamp[0]=0
                            droid.video.images(tensor[512,3,320,512],全为0) -> 第[0,:,:,:]维赋值image[0]
                            droid.video.poses(tensor[512,7],每一行为(0,0,0,0,0,0,1)) -> images.video.poses第[0,:]维度赋Id
                            droid.video.disps(tensor[512,40,64],全为1) -> 第[0,:,:]维赋值1
                            droid.video.disps_sens（tensor[512,40,64],全为0）没变化
                            droid.video.intrinsics(tensor[512,4],全为0) -> 第[0,:]维赋值(32,32,32,24)
                            droid.video.fmaps (tensor[512,2,128,40,64],全为0) -> 第[0,:,:,:,:]赋值gmap[2,128,40,64]: 值是满的并且混乱
                            droid.video.nets(tensor[512,128,40,64],全为0) -> 第[0,:,:,:]赋值net[0,0]:tensor[40,64]: 值是满的并且混乱
                            droid.video.inps(tensor[512,128,40,64],全为0) -> 第[0,:,:,:]赋值inp[0,0]:tensor[40,64]: 值是满的并且混乱
        when counter!=0 
            判定新加入的image的flow的norm(deltax,deltay).mean().item()是否大于droid.filterx.thresh=2.4
            如果大于,则进行以下代码，如果没有,则droid.filterx.count+1;但其实如果大于,droid.filterx.count会再次变成0
            *_, delta, weight = droid.filterx.update(droid.filterx.net[None], droid.filterx.inp[None], corr)    以下是参数及其返回值解释*
                droid.filterx.net(tensor[1,128,40,64]):使用的是上一次旧的数据
                droid.filterx.inp(tensor[1,128,40,64]):使用的是上一次旧的数据
                corr(tensor[1,1,196,40,64]): corr = CorrBlock(droid.filterx.fmap[None,[0]], gmap[None,[0]])(coords0)  以下是参数解释
                    coords0(tensor[1,1,40,64,2]):创建的临时变量，最后一维度为(0,0)(1,0)(2,0)(3,0)...(63,0)(0,1)(1,1)(2,1)...(63,1)...
                    droid.filterx.fmap(tensor[2,128,40,64]):使用的是上一次旧数据
                    gmap(tensor[2,128,40,64]):创建的临时变量，使用的是由新的image通过转换成inputs,droid.filterx._feature_encoder(inputs)得到
                delta(tensor[1,1,40,64,2]):得到每个像素deltax,deltay的光流数值
                weight(tensor[1,1,40,64,2]):得到每个像素deltax,deltay的权重，但后续没有使用
``          droid.filterx(MotionFilter)``   更新的元素
                droid.filterx.count:一直在0和接近并包括1的数字变化
                droid.filterx.net(tensor[1,128,40,64]):通过droid.filterx.__context_encoder(inputs[:,[0]])得到，也就是通过新image
                droid.filterx.inp(tensor[1,128,40,64]):通过droid.filterx.__context_encoder(inputs[:,[0]])得到，也就是通过新image
                droid.filterx.fmap(tensor[2,128,40,64]):通过droid.filterx._feature_encoder(inputs)得到gmap,再赋值给fmapa
                *droid.video.append() 将droid.video.counter使用droid.video.get_lock()锁定,以下是参数列表解释*
                    index=droid.video.counter 这里counter还没有更新，还是老旧的数据
                    tstamp: 循环的t
                    image[0]取image的第0维的第0个，不要第一个
                    Id: None
                    depth: None
                    intrinsics / 8.0: tensor[4](32.,32.,32.,24)
                    gmap[2,128,40,64]: 值是满的并且混乱,来自于更新的droid.filterx.__feature_encoder(inputs)，包含了image[2,3,320,512]
                    net[0]:tensor[128,40,64]: 值是满的并且混乱,与更新的droid.filterx.net值相同
                    inp[0]:tensor[128,40,64]: 值是满的并且混,与更新的droid.filterx.inp值相同
``                      droid.video(DepthVideo)``   更新的元素
                            droid.video.counter -> droid.video.counter=index+1
                            droid.video.tstamp(tensor[512]) -> 第index项为循环的t，通过这个元素可以得到第几帧被作为关键帧
                                例如: droid.video.tstamp=[0,2,4,6 ...]，则第0，2，4，6帧被作为关键帧
                            droid.video.images(tensor[512,3,320,512]) -> 第[index,:,:,:]维被赋值image[0]
                            droid.video.poses(tensor[512,7],每一行为(0,0,0,0,0,0,1)) 没变化
                            droid.video.disps(tensor[512,40,64],全为1) 没变化
                            droid.video.disps_sens（tensor[512,40,64],全为0 没变化
                            droid.video.intrinsics(tensor[512,4]) -> 第[index,:]维被赋值 intrinsics / 8.0: tensor[4](32.,32.,32.,24)
                            droid.video.fmaps (tensor[512,2,128,40,64],全为0) -> 第[index,:,:,:,:]赋值gmap[2,128,40,64]: 值是满的并且混乱
                            droid.video.nets(tensor[512,128,40,64],全为0) -> 第[index,:,:,:]赋值net[0]:tensor[128,40,64]: 值是满的并且混乱
                            droid.video.inps(tensor[512,128,40,64],全为0) -> 第[index,:,:,:]赋值inp[0]:tensor[128,40,64]: 值是满的并且混乱
**初始化 initialize,t=29,droid.video.counter.value==warmup=15**
    *droid.frontend()->droid.frontend._initialize()*
``      droid.frontend(DroidFrontend)``
            droid.frontend.t0=0 没有变化
            droid.frontend.t1=0 -> droid.video.counter.value
            *droid.frontend.graph.add_neighborhoond_factors(droid.frontend.t0, droid.frontend.t1, r=3)*
                c=1 if self.video.stereo else 0
                ii(tensor[64]):临时变量值为(0,0,0,0,0,0,0,0,1,1,1,...,14)
                jj(tensor[64]):临时变量值为(0,1,2,3,4,5,6,7,8,9,10,...,14)
                *droid.frontend.graph.add_factors(ii[keep], jj[keep]) 以下是参数解释*
                    keep(tensor[64],bool):取ii中对应的jj元素的差值在(c,r]之间为True，否则为False
                    ii[keep](tensor[50]):以keep为指标过滤ii(tensor[64])，保留True的元素
                    jj[keep](tensor[50]):以keep为指标过滤jj(tensor[64])，保留True的元素
                        *ii,jj=droid.frontend.graph._filter_repeated_edges(ii[keep],jj[keep])去除和之前重复的边*
                        *判定是否需要remove多余的元素*
``                      droid.frontend.graph(FactorGraph)`` 更新的元素
                            droid.frontend.graph.net(None) -> droid.frontend.graph.net(tensor[1,,128,40,64])
                                .to(droid.frontend.graph.device)
                                来自于droid.video.nets[ii[keep]]的第[ii[keep],:,:,:]维度，满数据
                                第1维在net是None的时候会是50，后面会变成与之前的droid.frontend.graph.net在第1维的叠加
``                          droid.frontend.graph.corr(CorrBlock)`` None的对象被另一个CorrBlock类所创建
                                droid.frontend.graph.corr(None) -> droid.frontend.graph.corr(CorrBlock)
                                    更新通过直接cat添加
                                    droid.frontend.graph.corr.num_levels=4
                                    droid.frontend.graph.radius=3
                                    droid.frontend.graph.corr_pyramid 包含四个tensor对象
                                       每一个tenosr维度为分别为[50,40,64,40,64],[50,40,64,24,32],[50,40,64,12,16],[50,40,64,6,8]
                                       c 依次比较ii[keep],jj[keep]里面的元素，如果一样则为真，1，否则为假，0
                                       droid.video.fmaps1(tensor[1, 50, 128, 40, 64])来自于droid.video.fmaps(tensor[512,2,128,40,64])[ii[keep],0]
                                       droid.video.fmaps2(tensor[1, 50, 128, 40, 64])来自于droid.video.fmaps(tensor[512,2,128,40,64])[jj[keep],c]
                                       通过对droid.video.fmaps1和droid.video.fmaps2特征图相乘得到相关性，再进行压缩维度和平均池化得到四个corr_pyramid对象
                            droid.frontend.graph.inp(None) -> droid.frontend.graph.inp(tensor[1,50,128,40,64])
                                来自于droid.video.inps(tensor[512,128,40,64])[ii]，并且头部增加一维
                                第1维在droid.frontend.graph.inp是None的时候会是50，后面会变成与之前的droid.frontend.graph.inp在第1维的叠加
                            droid.frontend.graph.target(tensor[1,0,40,64,2],全0)
                            -> (tensor[1,50,40,64,2]),从ii[keep]投影到jj[keep]像素,后续沿着第一维度使用cat增加
                                *target, _=droid.video.reproject(ii[keep],jj[keep])*
                                *得到return,需要pops.projective_transform函数有关的值*
                                    *coords, valid_mask=import geom.projective_ops as pops.projective_transform*
                                    *(Gs,droid.video.disps[None],droid.video.intrinsics[None],ii,jj)*
                                    *得到return，需要与 iproj函数，actp函数，proj函数有关的值*
                                        Gs(tensor[1,512,7]):第一维每一行为[0,0,0,0,0,0,1]=poses                                      
                                        droid.video.disps(tensor[512,40,64],全为1)[None] -> tensor[1,512,40,64] = depths
                                        droid.video.intrinsics(tensor[512,4])[None] -> tensor[1,512,4] = intrinsics
                                        ii(tensor[50]):就是之前的ii[keep] 
                                        jj(tensor[50]):就是之前的jj[keep] 
                                            *X0, Jz = iproj(depths[:,ii], intrinsics[:,ii], jacobian=jacobian)*
                                                depths[:,ii](tensor[1,50,40,64]) = disps 
                                                intrinsics[:,ii](tensor[1,50,4]) = intrinsics
                                                    return Jz : None
                                                    return pts = X0:
                                                        y(tensor[40,64]):每一列从上到下0-39,即为disps的第二维
                                                        x(tensor[40,64]):每一行从左到右0-63,即为disps的第三维
                                                        i(tensor[1,50,40,64]):全是1
                                                        Y(tensor[1,50,40,64]):归一化计算，通过(y-cy) / fy,没有乘Z，否则是相机坐标系3D坐标
                                                        X(tensor[1,50,40,64]):归一化计算，通过(x-cx) / fx,没有乘Z，否则是相机坐标系3D坐标
                                                        pts(tensor[1,50,40,64,4]):最后一维分别是:X,Y,i,disps
                                            Gij(tensor[1,50,7])
                                                poses(tensor[1,512,7])[:,jj] * poses(tensor[1,512,7])[:,ii].inv() 
                                                If ii与jj中的元素完全一致，则将它们对应的Gij第一维值设置为[-0.1, 0.0, 0.0, ..., 1.0]
                                                    即把ii[keep]对应的poses与jj[keep]对应的poses进行处理，可以批量处理
                                                    注意！由于这里poses为SE3的矩阵，因此可以对[1,50,7]维度的tensor数据求逆，会自动转换成方阵
                                            *X1, Ja = actp(Gij, X0, jacobian=jacobian)*
                                                Gij(tensor[1,50,7]):临时变量得到的旋转矩阵
                                                X0(tensor[1,50,40,64,4]):iproj函数返回的临时变量pts
                                                    return Ja: None 
                                                    return X1(tensor[1,50,40,64,4]): 
                                                        = Gij[:,:,None,None](tensor[1,50,1,1,7]) * X0
                                                        即将变换矩阵Gij 乘以 包含归一化计算的X,Y和深度初始值的X0
                                            *x1, Jp = proj(X1, intrinsics[:,jj], jacobian=jacobian, return_depth=return_depth)*
                                                X1(tensor[1,50,40,64,4]) = Xs
                                                intrinsics[:,jj](tensor[1,50,4]) : 选择jj[keep]中的intrinsics 
                                                jacobian: False
                                                return_depth: False
                                                    return Jp : None
                                                    return coords(tensor[1,50,40,64,2]) = x1:投影到jj序列对应的像素坐标值
                                                        Z(tensor[1,50,40,64]) 全是1
                                                            就是上一步的i(tensor[1,50,40,64]):全是1,如果其中元素< MIN_DEPTH=0.2，设1
                                                        D(tensor[1,50,40,64])  
                                                            上面的depths[:,ii](tensor[1,50,40,64]) = disps
                                                            depths从droid.video.disps(tensor[512,40,64])中来
                                                        coords(tensor[1,50,40,64,2])
                                                            x(tensor[1, 50, 40, 64]) = fx * (X * d) + cx
                                                            y(tensor[1, 50, 40, 64]) = fy * (Y * d) + cy
                                                            coords沿着最后一维叠加
                                            valid(tensor[1,50,40,64,1])
                                                float类型的bool，要么0，要么1
                                                如果 X1(tensor[1,50,40,64,4]) 和 X0 (tensor[1,50,40,64,4]) 最后一维度的第3个元素，即相机坐标系下的Z，大于MIN_DEPTH=0.2，则为真
                                                这里全为1
                                            *return x1 = coords = add_factors函数中droid.frontend.graph.target, valid = valid_mask*
                                                x1(tensor[1,50,40,64,2])    
                                                    ！！！x1 = coords = add_factors函数中droid.frontend.graph.target 的原型
                                                    新投影的jj[keep]里的像素值，其实这里相比于ii[keept]设定的像素值没有变化，因为两个序列的位置参数都为单位参数，没有发生旋转
                                                valid(tensor[1,50,40,64,1])
                            droid.frontend.graph.weight(tensor[1,0,40,64,2],全为0) 
                            -> tensor[1,50,40,64,2],维度与droid.frontned.graph.targt相同，并且全为0
                            -> 之后沿着第一维使用cat更新
                            droid.frontend.graph.ii(tensor[50]) 
                            -> 经过*droid.frontend.graph._filter_repeated_edges(ii[keep],jj[keep])*去除和之前重复的边，不是去除{1,2},{2,1}..
                            -> 之后沿着第0维使用cat更新
                            droid.frontend.graph.jj(tensor[50]) 
                            -> 经过*droid.frontend.graph._filter_repeated_edges(ii[keep],jj[keep])*去除和之前重复的边，不是去除{1,2},{2,1}..
                            -> 之后沿着第0维使用cat更新
                            droid.frontend.graph.age(tensor[]) 
                            -> dorid.frontend.graph.age(tensor[50]),全为0，与dorid.frontend.graph.ii(tensor[50]) 维度相同
                            -> 之后沿着第0维使用cat更新
            *droid.frontend.graph.update(t0 = 1, t1 = None, itrs=2,use_inactive=True,EP=1e-7,motion_only=False)*
            *对droid.frontend.graph的元素进行循环8次循环,从而更新*
``              droid.frontend.graph(FactorGraph)`` 更新的元素
                    droid.frontend.graphcoords0(tensor[40,64,2],第:,:,0维度每一行从左往右0-47，第:,:,1维度每一列从上到下0-63) - 没有变化
                    droid.frontend.graph.inp(tensor[1,50,128,40,64])
                    - 来自于droid.video.inps(tensor[512,128,40,64])[ii]，并且头部增加一维
                    - 特殊！每次循环没有变化，但是被用在了很重要的droid.frontend.graph.update_op函数中
                    droid.frontend.graph.age(tensor[50]) -> 每次循环 数值+1
                    *droid.frontend.graph.net, delta, weight, damping, upmask = droid.frontend.graph.update_op*
                    *(droid.frontend.graph.net,droid.frontend.graph.inp,corr,motn,droid.frontend.graph.ii,droid.frontend.graph.jj) 主更新函数*
                        droid.frontend.graph.net(tensor[1,50,128,40,64]) : 来自于droid.video.nets[ii[keep]]的第[ii[keep],:,:,:]维度,满数据
                        droid.frontend.graph.inp(tensor[1,50,128,40,64]) : 来自于droid.video.inps(tensor[512,128,40,64])[ii]，且头部增一维
                        corr(tensor[1, 50, 196, 40, 64]):临时变量，来自于corr.py中的corr(coords1)函数
                            coords1(tensor[1, 50, 40, 64, 2])
                                来自于droid.video.reproject(droid.frontend.graph.ii,droid.frontend.graph.ii)函数
                                每次都会更新，因为每次循环的时候这个函数一个很重要的变量droid.video.poses(tensor[512,7])会变化，导致重投影结果变化
                        motn(tensor[1, 50, 4, 40, 64])
                            临时变量，与每次更新的coords1(tensor[1, 50, 40, 64, 2])，每次更新的droid.frontend.graph.target有关
                            再重新排列维度并且限制数值在(-64,64),变成(tensor[1,50,4,40,64])
                        droid.frontend.graph.ii(tensor[50]) 
                            经过*droid.frontend.graph._filter_repeated_edges(ii[keep],jj[keep])*去除和之前重复的边，不是去除{1,2},{2,1}..
                        droid.frontend.graph.jj(tensor[50]) 
                            经过*droid.frontend.graph._filter_repeated_edges(ii[keep],jj[keep])*去除和之前重复的边，不是去除{1,2},{2,1}..
                        delta(tensor[1, 50, 40, 64, 2]) : 目标跟踪光流像素的变化值，就是论文中的rij,通过网络预测得到,每次循环更新
                        weight(tensor[1, 50, 40, 64, 2]) : 目标跟踪光流像素预测值的权重，每次循环更新
                        damping(tensor[1,8,40,64])  
                            阻尼系数 (damping factor), 为了Levenberg-Marquardt算法的收敛性和稳定性，每次循环更新，神经网络得到的结果
``                      droid.frontend.graph(FactorGraph)`` 更新的元素
                            droid.frontend.graph.target(tensor[1,50,40,64,2]) 
                            -> coords1 + delta 维度(tensor[1,50,40,64,2]), dtype=torch.float
                            -> 很关键的更新元素，实现了论文中'pij*'变量
                            droid.frontend.graph.weight(tensor[1,0,40,64,2],全为0) -> 临时变量weight赋值给它，并且更新
                            droid.frontend.graph.damping(tensor[512,40,64],全为1e-6) 
                            -> 选取droid.frontend.graph.ii中不重复的元素成为droid.frontend.graph.damping第0维的index
                            -> 再用damping(tensor[1,8,40,64])赋值给它,当然了这里不重复的元素也有8个
                            -> 维度依然是tensor([512,40,64])
                            droid.frontend.graph.net(tensor[1,50,128,40,64]) -> 改变数值，不改变维度
                    *droid.frontend.graph.video.py_ba=*
                    *(target,weight,damping,ii,jj,t0,t1,itrs=itrs,lm=1e-4,motion_only=motion_only) 核心代码*
                        target(tensor[1,50,40,64,2]):
                            if use_inactive,沿第一维度给droid.frontend.graph.target(tensor[1,50,40,64,2])加部分droid.frontend.graph.target_inac(tensor[1,0,40,64,2]) 
                            否则直接是droid.frontend.graph.target(tensor[1,50,40,64,2])
                        weight(tensor[1,50,40,64,2])
                            if use_inactive,则沿第一维度给droid.frontend.graph.weight(tensor[1,50,40,64,2])加droid.frontend.graph.tweight_inac(tensor[1,0,40,64,2])
                        damping(tensor[8, 40, 64]):
                            找到ii[keep]序列中unique的数，以此为索引筛选 droid.frontend.graph.damping(tensor[512,40,64])第0维度对应的元素。最后乘0.2加上一个常数EP
                        ii(tensor[50])
                            if use_inactive,沿第0维度给droid.frontend.graph.ii(tensor[50])加部分droid.frontend.graph.ii_inac
                            否则直接是droid.frontend.graph.ii(tensor[50])
                        jj(tensor[50])
                            if use_inactive,沿第0维度给droid.frontend.graph.jj(tensor[50])加部分droid.frontend.graph.jj_inac
                            否则直接是droid.frontend.graph.jj(tensor[50])
                        t0:1
                        t1:None
                            *poses, disps = BA(target,weight,damping,poses,disps,self.intrinsics[None],ii,jj,fixedp=t0)*
                            *最核心BA代码,包含了公式组合和求解，循环两次*
                                target(tensor[1,50,40,64,2]):py_ba函数的参数
                                weight(tensor[1,50,40,64,2]):py_ba函数的参数
                                damping(tensor[8,40,64]):py_ba函数的参数
                                poses(tensor[1,15,7]):
                                    t1:ii[keep]和jj[keep]的最大维度再+1,这里为15
                                    使用droid.video.poses(tensor[512,7])创建一个lietorch.SE3对象，并且取前t1行，再增加维度
                                disps(tensor[1, 15, 40, 64]):取droid.video.disps(tensor[512,40,64])前t1行，再增加维度
                                intrinsics[None](tensor[1,512,4]):来自于droid.video.intrinsics(tensor[512,4])
                                ii(tensor[50]):py_ba函数的参数
                                jj(tensor[50]):py_ba函数的参数
                                fixedp=t0:py_ba函数的参数
                                    *coords, valid, (Ji, Jj, Jz) = pops.projective_transform(poses, disps, intrinsics, ii, jj, jacobian=True)*
                                    *构建Jacobian矩阵，求解变化量*
                                        自变量全为BA函数内传递的参数
                                            coords(tensor[1, 50, 40, 64, 2]):
                                                创造两个矩阵x,y，他们分别是每一列从0到63，每一行从0到47代表初始化图像的像素点坐标x,y，我称之为初始网格像素坐标，包含
                                                将初始网格像素坐标通过相机内参转换成归一化平面坐标上的坐标，并且设置Z为1，disps视差为1，这个部分共四维，分别为X,Y,Z,W,正如论文所说，称之为归一化坐标
                                                创造Gij，即poses[:,jj] * poses[:,ii].inv()。每次迭代由于poses更新了因此Gij都会更新
                                                将归一化坐标通过Gij转换成新的归一化坐标，再转换到更新网格像素坐标。注意，如果是立体视觉，更新网格像素坐标的第二维应该是更新后的视差坐标除以更新后的Z
                                                这个更新后的网格像素表格做就是coords
                                            valid(tensor[1, 50, 40, 64, 1])
                                                只有旧的和新的归一化坐标的第2维度，即Z都大于MIN_DEPTH,对应的值是float类型的真
                                            Ji(tensor[1,50,40,64,2,6])
                                                物理意义: p'对 了i 求导，从论文上看维度为[2,4]x[4,6]=[2,6]
                                            Jj(tensor[1,50,40,64,2,6])
                                                物理意义: p'对 了j 求导，从论文上看维度为[2,4]x[4,6]=[2,6]
                                            Jz(tensor[1,50,40,64,2,1])
                                                物理意义: p'对 d 求导，从论文上看维度为[2,4]x[4,1]=[2,1]
                                    r(tensor[1,50,5120,1])
                                        target(tensor[1,50,40,64,2])和经过函数pops.projective_transform处理后的coords(tensor[1, 50, 40, 64, 2])的差值
                                        重排列40x64x2
                                    w(tensor[1,50,5120,1])
                                        valid(tensor[1, 50, 40, 64, 1])和weight(tensor[1,50,40,64,2])乘积的结果
                                        缩小为原来的1/1000，并且重排列40x64x2
                                    Ji(tensor[1,50,5120,6]) <- 重排列 - (tensor[1,50,40,64,2,6])
                                    Ji(tensor[1,50,5120,6]) <- 重排列 - (tensor[1,50,40,64,2,6])
                                    Jz(tensor[1,50,2560,2]) <- 重排列 - (tensor[1,50,40,64,2,1])
                                    wJiT(tensor[1,50,6,5120])
                                        w(tensor[1,50,5120,1]) x Ji(tensor[1,50,40,64,2,6]) -> tensor[1,60,5120,6]
                                        .transpose转置结果的2,3维度
                                    wJjT(tensor[1,50,6,5120])
                                        w(tensor[1,50,5120,1]) x Jj(tensor[1,50,40,64,2,6]) -> tensor[1,60,5120,6]
                                        .transpose转置结果的2,3维度
                                    Hii, Hij, Hji, Hjj(tensor[1,50,6,6])
                                        通过维度相同的wJiT, wJjT与维度相同的Ji,Jj分别相乘得到的结果 tensor[1,50,6,5120] x tensor[1,50,5120,6]
                                    vi，vj(tensor[1,50,6])
                                        通过维度相同的wJiT,wJjT与r相乘得到的结果 tensor[1,50,6,5120] x tensor[1,50,5120,1], 再squeeze最后一维
                                    Ei,Ej(tensor[1,50,6,2560])
                                        wJiT.view(B,N,D,ht*wd,-1): 将wJiT的维度最后一维度重新塑,变为tensor[1,50,6,2560,2]
                                        Jz[:,:,None]: 这为Jz增加了一个新的维度，使其形状变为tensor[1,50,1,2560,2]
                                        相乘以上的两个元素，变成tensor[1,50,6,2560,2]，再沿着最后一个维度.sum求和
                                        Ej同理，换掉wJiT
                                    w,r(tensor[1,50,2560,2]) 
                                        还原w,r的维度
                                    wk(tensor[1,50,2560])
                                        维度相同的w,r,Jz(tensor[1,50,2560,2])对应元素相乘，再沿着最后一维度相加
                                    Ck(tensor[1,50,2560])
                                        维度相同的w,Jz,Jz(tensor[1,50,2560,2])对应元素相乘，再沿着最后一维度相加
                                    kx: 代表着ii中所有的唯一元素
                                    kk: 维度与ii相同，为ii中的每个元素提供在kx中的索引
                                    M: kx中元素的数量
                                    P,ii,jj: 值分别减去fixedp，也就是t0
                                    H(tensor[1,196,6,6]) 结合Hii, Hij, Hji, Hjj
                                    H(tensor[1,14,14,6,6]) 重排列
                                    E(tensor[1,210,6,2560]) 结合Ei, Ej
                                    E(tensor[1,14,15,6,2560]) 重排列
                                    v(tensor[1,14,6])
                                    C(tensor[1,15,2560])
                                    w(tensor[1,15,2560])
                                    *dx, dz = schur_solve(H, E, C, v, w) 使用schur求解*
                                        dx(tensor[1,14,6])
                                        dz(tensor[1,15,2560])
                                *BA* Return:
                                    poses(tensor[1,15,6])
                                    disps(tensor[1,15,40,64])
                                        保证disps在0-10之间，大于10为0
                        *py_ba* Return:
``                      droid.video(DepthVideo)`` 更新的元素
                            droid.video.poses(tensor[512,7])
                                [:,t1]的部分被更新了数据，不过注意，这里的第0个作为锁定帧并没有被修改数据
                            droid.video.disps(tensor[512,40,64])
                                [:,t1]的部分被更新了数据，全部都被修改了，包括第0个
                    *droid.frontend.graph.update* Return
``                  droid.frontend.graph(FactorGraph)`` 更新的元素                    
                        dorid.frontend.graph.age(tensor[50])->全体元素+1
            *添加边，并且设置这些边对应的结点距离为无穷大，以保证不通过距离比较近的条件，添加边*
            *droid.frontend.graph.add_proximity_factors(t0=0, t1=0, rad=2, nms=2, thresh=self.frontend_thresh==17.5, remove=False)*                
                t:等于droid.frontend.graph.video.counter.value
                ii:获取 t0 到 t 的值，并且与jj形成网格
                jj:获取 t1 到 t 的值，并且与ii形成网络
                d:使用cuda编程获得的ii,jj中的元素对应的结点之间的距离
                ！！！距离无穷大设置 第一部分规则
                  1）对于ii, jj形成的网格对应的照片
                        对于ii中元素对应的每张照片，将它之前对应的jj中元素对应的 rad 张照片(不包括rad) 设置d为 inf
                        例如:5属于ii, 1,2,3,4,5,6,7...属于jj，则d(5,4),d(5,5)设置为 inf, d(5,1),d(5,2),d(5,3),(5,6),(5,7)为原来的距离
                  1) 对于ii, jj形成的网格对应的照片
                        d>100, 则设置d为 inf
                  3）对于ii1, jj1形成的网格对应的照片
                        ii1: self.ii, self.ii_bad, self.ii_inac; 这里的self.ii是通过(r,c]条件筛选出来的
                        jj1: self.jj, self.jj_bad, self.jj_inac; 这里的self.jj是通过(r,c]条件筛选出来的
                        对与ii1, jj1中对应的照片组,如果它们附近结点的距离存在于d中，则将它们附近的结点（-nms,nms+1）的d也设置成无穷大
                        例如:(5,2),相差绝对值3,则 d(4,2),d(6,2),d(5,2),d(5,1),d(5,3)会被设置为 inf
                        例如:(5,3),相差绝对值2,则 d(5,3)会被设置为 inf
                ！！！给 es 添加边,并且设置距离无穷大 第二部分规则
                  1）对于ii中的每个元素i (t0 到 t)对应的照片
                        如果为stereo, 给 es 添加边 (i,i), 并且设置对应的d距离无穷大
                        令j为，i的前 rad 到 i-1张图片的序号，添加 双向边(i,j) 并且设置对应的d距离为无穷大
                  2) 对于所有的 d 小于 thresh 且这个时候 es 的长度小于 droid.frontned.graph.max_factors=48 的
                        给 es 添加 双向边(i,j), 如果它们附近结点的距离存在于d中，则将它们附近的结点（-nms,nms+1）的d也设置成无穷大
                *droid.frontend.graph.add_factors(ii,jj,remove=false)*
                    ii(tensor[43]), jj(tensor[43])
                    更新的变量
                        *droid.frontend.graph.add_factors._filter_repeated_edges(ii,jj)*
                            去掉self.ii+self.ii_inac 链接 self.jj+self.jj_inac 中包含的照片对
                        droid.frontend.graph.ii(tensor[93]) <- droid.frontend.graph.ii(tensor[50])
                        droid.frontend.graph.jj(tensor[93]) <- droid.frontend.graph.jj(tensor[50])
                        droid.frontend.graph.net(tensor[1,93,40,64,2]) <- droid.frontend.graph.net(tensor[1,50,40,64,2])
                            增加的部分为droid.frontend.graph.video.nets[ii]
                        droid.frontend.graph.inp(tensor[1,93,40,64,2]) <- droid.frontend.graph.inp(tensor[1,50,40,64,2])
                            增加的部分为droid.frontend.graph.video.inps[ii]
                        droid.frontend.graph.corr
                            droid.frontend.graph.corr.num_levels=4
                            droid.frontend.graph.radius=3
                            droid.frontend.graph.corr_pyramid 包含四个tensor对象
                                每一个tenosr维度为分别为[93,40,64,40,64],[93,40,64,24,32],[93,40,64,12,16],[93,40,64,6,8] 
                        droid.frontend.graph.age(tensor[93])
                                前50为8，后43为0
                        droid.frontend.graph.target(tensor[1,93,40,64,2]) <- droid.frontend.graph.target(tensor[1,50,40,64,2])
                            target是通过droid.frontend.video.reproject(ii,jj)得到
                            意义是ii,jj中的每对照片中的所有像素通过之前的Gs = lietorch.SE3(self.poses[None]) 重投影得到
                        droid.frontend.graph.weight(tensor[1,93,40,64,2]) <- droid.frontend.graph.weight(tensor[1,50,40,64,2]) 
                            增加的部分为 torch.zeros_like(target)，全为0
            *droid.frontend.graph.update(t0=1, use_inactive=True) 循环8次*
                coords1(tensor[1,93,40,64,2])
                    *droid.frontend.graph.video.reproject(droid.frontend.graph.ii(tensor[93]),droid.frontend.graph.jj(tensor[93]))*
                motn(tensor[1,93,4,40,64])
                    使用下面的数据得出结论
                    coords1
                    droid.frontend.graph.coords0(tensor[1,93,40,64,2])最原始的像素坐标
                    droid.frontend.graph.target(tensor[1,93,40,64,2]) 刚刚通过cat 43 50 得到的
                droid.frontend.graph.net(tensor[1,93,128,40,64])
                    通过 *droid.frontend.update_op*函数得到
                droid.frontent.graph.weight(tensor[1,93,128,40,64])
                    通过 *droid.frontend.update_op*函数得到
                delta(tensor[1,93,40,64,2]) weight(tensor[1,93,40,64,2])
                    通过 *droid.frontend.update_op*函数得到              
                更新 droid.frontend.graph.target(tensor[1,93,40,64,2])
                    通过coords1+delta
                *droid.frontend.graph.video.py_ba(target, weight, damping, ii, jj, t0, t1, itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)* 
                更新 前t1 个 droid.video.poses, droid.video.disps(不包括t1)
                使用在添加变之后的droid.frontend.graph.ii, droid.frontend.graph.jj进行BA iters次
                    t0=1, t1=15
                    target: 结合droid.frontend.graph.target 和 一部分 droid.frontend.graph.target_inac
                    weight: 结合droid.frontend.graph.weight 和 一部分 droid.frontend.graph.weight_inac
                    ii: 结合droid.frontend.graph.ii 和 一部分 droid.frontend.graph.ii_inac
                    jj: 结合droid.frontend.graph.jj 和 一部分 droid.frontend.graph.jj_inac
                更新 droid.frontend.graph.age +1
            *为下一次添加帧创造初始值*
                将t1-1时刻的 poses clone给 droid.video.poses[t1]
                将t1-4 到 t1-1时刻的 disps 的平均值 clone给 droid.video.disps[t1]
                将t1-1时刻的 droid.video.poses clone给 droid.frontend.last_pose
                将t1-1时刻的 droid.video.disps clone给 droid.frontend.last_disps
                将t1-1时刻的 droid.video.tstamp clone给 droid.frontend.last_time
                droid.video.drity[:droid.frontend.t1] -> true
            *droid.frontend.graph.rm_factors(droid.frontend.graph.ii<droid.frontend.warmup-4, store=true)*
                mask: droid.frontend.graph.ii < droid.frontend.warmup-4
                      即只保留 warmup的后四个帧，其它的元素移除
                droid.frontend.graph.ii_inac(tensor[71]) <- 空  结合mask对应的和之前的droid.frontend.graph.ii_inac
                droid.frontend.graph.jj_inac(tensor[71]) <- 空  结合mask对应的和之前的droid.frontend.graph.jj_inac
                droid.frontend.target_inac(tensor[1,71,40,64,2]) <- 空 结合mask对应的和之前的droid.frontend.graph.target_inac
                droid.frontend.weight_inac(tensor[1,71,40,64,2]) <- 空 结合mask对应的和之前的droid.frontend.graph.weight_inac
                去除老旧元素后的元素
                    droid.frontend.graph.ii(tensor[22])
                    droid.frontend.graph.jj(tensor[22])
                    droid.frontend.graph.age(tensor[22])
                    droid.frontend.graph.corr
                            droid.frontend.graph.corr.num_levels=4
                            droid.frontend.graph.radius=3
                            droid.frontend.graph.corr_pyramid 包含四个tensor对象
                                每一个tenosr维度为分别为[22,40,64,40,64],[22,40,64,24,32],[22,40,64,12,16],[22,40,64,6,8] 
                    droid.frontend.graph.net(tensor[1,22,128,40,64]) <- droid.frontend.graph.net(tensor[1,93,128,40,64])
                    droid.frontend.graph.inp(tensor[1,22,128,40,64]) <- droid.frontend.graph.inp(tensor[1,93,128,40,64])
                    droid.frontend.graph.target(tensor[1,22,40,64,2]) <- droid.frontend.graph.target(tensor[1,93,40,64,2])
                    droid.frontend.graph.weight(tensor[1,22,40,64,2]) <- droid.frontend.graph.weight(tensor[1,22,40,64,2])
**初始化结束**
**更新阶段**
    *droid.filterx.track(new image)*
    根据new image更新网络: 移动距离与droid.filterx.thresh=2.4相比,判定图像是否符合作为关键帧的条件，
        *delta=droid.filterx.update(droid.filterx.net, droid.filterx.inp, corr)*
            net:上一次的net
            inp:上一次的inp
            corr:新的corr
            如果delta > droid.filterx.thresh
                droid.filterx.net(tensor[1,128,40,64])
                droid.filterx.inp(tensor[1,128,40,64])
                droid.filterx.fmap(tensor[2,128,40,64])
                *droid.filterx.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0])*
                    自变量
                        index = 15 代表这是第多少个关键帧
                        tstamp= 32 代表读取了多少帧
                    改变量
                    droid.video.counter.value = index + 1 =16
                    droid.video.tstamp[index]: 包含关键帧 tstamp 的序列,其中index项改变
                    droid.video.images[index]: new image序列,其中index项改变
                    droid.video.poses[index] : 没有改变！之前已经给过初值
                    droid.video.disps[index] : 没有改变！之前已经给过初值
                    droid.video.disps_sens[index] : 没有改变！从来不存在
                    droid.video.intrinsics[index] : 新的 intrinsics
                    droid.video.fmaps[index] : 新的fmaps,其中index项改变 : 来自于droid.filterx.fmap(tensor[2,128,40,64])
                    droid.video.nets[index] : 新的nets,其中index项改变 : 来自于droid.filterx.net(tensor[1,128,40,64])
                    droid.video.inps[index] : 新的inps,其中index项改变 : 来自于droid.filterx.inp(tensor[2,128,40,64])
    *droid.frontend()* -> *droid.frontend._update()* 如果添加新关键帧成功
        droid.frontend.count 0 -> 1, 并且以后叠加
        droid.frontend.t1 15->16, 并且以后叠加
        如果droid.frontend.graph.corr is not None
            *droid.frontend.graph.rm_factors()*
                以 droid.frontend.graph.age > droid.frontend.max_age(25) 为判定条件
                将过老的droid.frontend.graph.ii, droid.frontend.graph.jj, droid.fronetnd.graph.target, droid.frontend.graph.weight保存到 _inac 中
                并且移除这些元素
            *droid.frontend.graph.add_proximity_factors(t0, t1, rad, nms, thresh, beta, remove=True)*
                自变量
                    t0 : droid.frontend.t1 - 5 
                    t1 : max(droid.frontend.t1 - droid.frontend.frontend_window(20), 0)
                    rad : droid.frontend.frontend_radius=2
                    nms : droid.frontend.frontend_nms=1
                    thresh : droid.frontend.frontend_thresh=17.5
                    beta : droid.frontend.beta=0.3
                    remove : True
                函数内部变化情况
                    t:droid.frontend.graph.video.counter.value=16
                    ii:获取 t0 到 t 的值，并且与jj形成网格 关键帧前5帧的范围
                    jj:获取 t1 到 t 的值，并且与ii形成网络 关键帧前20帧的范围
                    d:使用cuda编程获得的ii,jj中的元素对应的结点之间的距离
                    ！！！距离无穷大设置 第一部分规则
                        1）ii前rad张照片(不包括rad)，若对应d存在，则设置 inf
                        2）d>100 inf
                        3) 之前的ii,即self.ii, self.ii_bad, self.ii_inac合成;
                           和之前的jj,即self.jj, self.jj_bad, self.jj_inac合成;
                           临近 nms 的i,j对应的d inf
                    ！！！给 es 添加边,并且设置距离无穷大 第二部分规则
                        1） 对于ii中的每个元素i (t0 到 t)对应的照片
                                如果为stereo, 给 es 添加边 (i,i), 并且设置对应的d距离无穷大
                                令j为，i的前 rad 到 i-1张图片的序号，添加 双向边(i,j) 并且设置对应的d距离为无穷大
                        2） 对于所有的 d 小于 thresh 且这个时候 es 的长度小于 droid.frontned.graph.max_factors=48 的
                                给 es 添加 双向边(i,j), 如果它们附近结点的距离存在于d中，则将它们附近的结点（-nms,nms+1）的d也设置成无穷大
                *droid.frontend.graph.add_factors(ii,jj,remove=false)*
                    更新的变量
                        *droid.frontend.graph.add_factors._filter_repeated_edges(ii,jj)*
                            去掉self.ii+self.ii_inac 链接 self.jj+self.jj_inac 中包含的照片对
                        *droid.frontend.graph.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)*
                            保证新的ii加入以后不会超过 max_factors=48 的限制，如果超过了，则根据 age 排除最旧的因子
                        droid.frontend.graph.ii
                        droid.frontend.graph.jj
                        droid.frontend.graph.net
                        droid.frontend.graph.inp
                        droid.frontend.graph.corr
                        droid.frontend.graph.age
                        droid.frontend.graph.target: 意义是ii,jj中的每对照片中的所有像素通过之前的Gs = lietorch.SE3(self.poses[None]) 重投影得到
                        droid.frontend.graph.weight, 增加的部分为 torch.zeros_like(target)，全为0
            *检查droid.frontend.graph.disps_sens是否存在，这个版本不存在*
            *droid.frontend.graph.update 循环droid.frontend.iters1=4次，再内部循环2次*
                coords1(tensor[1,43,40,64,2])
                    *droid.frontend.graph.video.reproject(droid.frontend.graph.ii(tensor[43]),droid.frontend.graph.jj(tensor[43]))*
                motn(tensor[1,43,4,40,64])
                    使用下面的数据得出结论
                    coords0(tensor[40,64,2])
                    droid.frontend.graph.coords1(tensor[1,43,40,64,2])最原始的像素坐标
                    droid.frontend.graph.target(tensor[1,43,40,64,2]) 刚刚通过cat 21 22 得到的
                droid.frontend.graph.ii(tensor[43])
                droid.frontend.graph.jj(tensor[43])
                droid.frontend.graph.net(tensor[1,43,128,40,64])
                    通过 *droid.frontend.update_op*函数得到
                droid.frontent.graph.weight(tensor[1,43,128,40,64])
                    通过 *droid.frontend.update_op*函数得到
                delta(tensor[1,43,40,64,2]) weight(tensor[1,43,40,64,2])
                    通过 *droid.frontend.update_op*函数得到              
                更新 droid.frontend.graph.target(tensor[1,43,40,64,2])
                    通过coords1+delta
                *droid.frontend.graph.video.py_ba(target, weight, damping, ii, jj, t0, t1, itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)* 
                t0: max(1, self.ii.min().item()+1) 作用是确定 fixedp
                t1: max(ii.max().item(), jj.max().item()) + 1 作用是确定要更新的前t1个 droid.video.poses, droid.video.disps       
                使用在添加变之后的droid.frontend.graph.ii, droid.frontend.graph.jj进行BA iters次
                    t0=1, t1=16
                    target: 结合droid.frontend.graph.target 和 t0对应的前3帧 droid.frontend.graph.target_inac
                    weight: 结合droid.frontend.graph.weight 和 t0对应的前3帧 droid.frontend.graph.weight_inac
                    ii: 结合droid.frontend.graph.ii 和 t0对应的前3帧 droid.frontend.graph.ii_inac
                    jj: 结合droid.frontend.graph.jj 和 t0对应的前3帧 droid.frontend.graph.jj_inac
                更新 droid.frontend.graph.age +1
                更新 前t1 个 droid.video.poses, droid.video.disps(不包括t1)
            *d=droid.frontend.graph.video.distance(t1-3,t1-2)*
            计算新帧和新帧的上一帧之间的距离，如果小于droid.frontend.keyframe_thresh(3.5),则使用下列函数去除新帧
                *droid.frontend.graph.rm_keyframe(droid.frontend.t1-2)*
                    将新帧的以下各项赋值给新帧的上一帧
                        droid.video.images, droid.video.poses, droid.video.dips, droid.video.disps_sens, droid.video.intrinsics
                        droid.video.nets, droid.video.inps, droid.video.fmaps
                    以下的self 为 droid.frontend.graph
                    所有大于或等于 上一帧 的 ii_inac 和 jj_inac 索引都会减少1;如果任何 ii_inac 或 jj_inac 与 上一帧 相关，下列因子将被从列表中删除
                        self.ii_inac, self.jj_inac, self.target_inac, self.weight_inac
                    所有大于或等于 上一帧 的 ii 和 jj 索引都会减少1;如果任何 ii 或 jj 与 上一帧 相关，下列因子将被从列表中删除
                        self.ii, self.jj, self.age, self.corr, self.net, self.inp, self.target, self.weight
                droid.frontend.video.counter.value 减少 1
                droid.frontend.video.t1 减少 1
            如果新帧被保留了
                *droid.frontend.graph.update(None, None, use_inactive=True)* 再次更新droid.video.poses, droid.video.disps 2次
            *为下一次添加帧创造初始值*
                将t1-1时刻的 poses clone给 droid.video.poses[t1]
                将t1-1时刻的 disps 的平均值 clone给 droid.video.disps[t1]
            droid.frontend.video.dirty[self.graph.ii.min():self.t1] -> true














    






                


    
            
                        




                    

                            


                      
                    

                        
                    


                

                                    
                                    

                                        

                                            
                                            

                                            



                                




                        
                        
                        


                        

                    










                                                    
                                                        





                                                                                                           

                                                    




                                    


                            



                                                         
                                



                        





            



                            



                    











