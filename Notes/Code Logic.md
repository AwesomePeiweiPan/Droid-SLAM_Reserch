**输入image格式**
    image[2,3,384,512]
**建立droid实例**
``  args``
        backend_nms=3
        backend_radius=2
        backend_thresh=22.0
        beta=0.3
        buffer=512
        calib=None
        disable_vis=True
        filter_thresh=2.4
        frontend_nms=1
        frontend_radius=2
        frontend_thresh=16.0
        frontend_window=25
        image_size=[384, 512]
        imagedir='datasets/TartanAir/seasidetown/P000'
        keyframe_thresh=4.0
        reconstruction_path=None
        stereo=True
        stride=3
        t0=0
        upsample=False
        warmup=8
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
            w(Sequential)
                (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
                (1): ReLU(inplace=True)  
                (2): Conv2d(128, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
                (3): GradientClip()  
                (4): Sigmoid()
``  droid.video(DepthVideo)``
        ht=384
        wd=512
        counter(Value('i',0))
        ready(Value('i',0))
        dirty(tensor[512]，全为False)
        disps(tensor[512,48,64],全为1)
        disps_sens（tensor[512,48,64],全为0）
        disps_up（tensor[512,384,512],全为0）
        fmaps (tensor[512,2,128,48,64],全为0)
        images(tensor[512,3,384,512],全为0)
        inps(tensor[512,128,48,64],全为0)
        intrinsics(tensor[512,4],全为0)
        nets(tensor[512,128,48,64],全为0)
        poses(tensor[512,7],每一行为(0,0,0,0,0,0,1))
        red(tensor[512],全为False)
        tstamp(tensor[512],全为0)     
``  droid.filterx(MotionFilter)``
        cnet=droid.net.cnet
        fnet=droid.net.fnet
        update=droid.net.update
        video=droid.video
        thresh=2.5
        device="cuda:0"
        count=0
        MEAN(tensor[3,1,1],第0维分别为(0.485, 0.456, 0.406))
        STDV(tensor[3,1,1],第0维分别为(0.229, 0.224, 0.225))
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
        keyframe_thresh=args.keyframe_thresh=4.0
        frontend_window=args.frontend_window=25
        frontend_thresh=self.frontend.thresh=16.0
        frontend_radius=args.frontend_radius=2
        graph(FactorGraph)
            ht=48
            wd=64
            video=droid.video
            update_op=droid.net.update
            device="cuda:0"
            max_factors=48
            corr_impl='volume'
            upsample=False
            coords0(tensor[48,64,2],第:,:,0维度每一行从左往右0-47，第:,:,1维度每一列从上到下0-63)
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
            damping(tensor[512,48,64],全为1e-6)
            target(tensor[1,0,48,64,2],全为0)
            weight(tensor[1,0,48,64,2],全为0)
            target_inac(tensor[1,0,48,64,2],全为0)
            weight_inac(tensor[1,0,48,64,2],全为0)
``  droid.backend(DroidBackend)``
        t0=0
        t1=0
        upsample = args.upsample=False
        beta = args.beta=0.3
        backend_thresh = args.backend_thresh=22.0
        backend_radius = args.backend_radius=2
        backend_nms = args.backend_nms=3
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
        STDV(tensor[3,1,1],第0维分别为(0.229, 0.224, 0.225))
**积累关键帧到 counter=8=args.warmup, t=14**
    *droid.filterx.track(tstamp:循环t,image[2,3,384,512]:循环的image,depth:None,intrinsics: tensor[4],值为(256., 256., 256., 192.))*
        when counter=0
``          droid.filterx(MotionFilter)``   增加的元素
                droid.filterx.fmap(tensor[2,128,48,64])
                    通过将输入的 image 第一维改变成R,G,B通道，增加一维度, 归一化/255
                    经过droid.filterx.MEAN和STDV成为inputs[1,2,3,384,512]
                    并且经过droid.filterx._feature_encoder()函数处理得到gmap[2,128,48,64],
                    最后赋值给fmap: 值是满的并且混乱
                droid.filterx.inp(tensor[1,128,48,64])
                    通过将输入的 image 经过上个fmap的处理得到inputs后，取第0维
                    经过droid.filterx._context_encoder(函数处理得到)
                droid.filterx.net(tensor[1,128,48,64])
                    通过将输入的 image 经过上个fmap的处理得到inputs后，取第0维
                    经过droid.filterx._context_encoder(函数处理得到)
                *droid.video.append() 将droid.video.counter使用droid.video.get_lock()锁定,以下是参数列表解释*
                    tstamp: 循环的t
                    image[0]取image的第0维的第0个，不要第一个
                    Id: tensor(7) 只有最后一个值是1，其余为0
                    depth: None
                    intrinsics / 8.0: tensor[4](32.,32.,32.,24)
                    gmap[2,128,48,64]: 值是满的并且混乱,来自于droid.filterx.__feature_encoder(inputs)，包含了image[2,3,384,512]
                    net[0,0]:tensor[48,64]: 值是满的并且混乱,与droid.filterx.net一样
                    inp[0,0]:tensor[48,64]: 值是满的并且混乱,与droid.filterx.inp一样
``                      droid.video(DepthVideo)``   更新的元素
                            droid.video.counter=0 -> droid.video.counter=1
                            droid.video.tstamp(tensor[512],全为0) -> droid.tstamp[0]=0
                            droid.video.images(tensor[512,3,384,512],全为0) -> 第[0,:,:,:]维赋值image[0]
                            droid.video.poses(tensor[512,7],每一行为(0,0,0,0,0,0,1)) -> images.video.poses第[0,:]维度赋Id
                            droid.video.disps(tensor[512,48,64],全为1) -> 第[0,:,:]维赋值1
                            droid.video.disps_sens（tensor[512,48,64],全为0）没变化
                            droid.video.intrinsics(tensor[512,4],全为0) -> 第[0,:]维赋值(32,32,32,24)
                            droid.video.fmaps (tensor[512,2,128,48,64],全为0) -> 第[0,:,:,:,:]赋值gmap[2,128,48,64]: 值是满的并且混乱
                            droid.video.nets(tensor[512,128,48,64],全为0) -> 第[0,:,:,:]赋值net[0,0]:tensor[48,64]: 值是满的并且混乱
                            droid.video.inps(tensor[512,128,48,64],全为0) -> 第[0,:,:,:]赋值inp[0,0]:tensor[48,64]: 值是满的并且混乱
        when counter!=0 
            判定新加入的image的flow的norm(deltax,deltay).mean().item()是否大于droid.filterx.thresh=2.5
            如果大于,则进行以下代码，如果没有,则droid.filterx.count+1;但其实如果大于,droid.filterx.count会再次变成0
            *_, delta, weight = droid.filterx.update(droid.filterx.net[None], droid.filterx.inp[None], corr)    以下是参数及其返回值解释*
                droid.filterx.net(tensor[1,128,48,64]):使用的是上一次旧的数据
                droid.filterx.inp(tensor[1,128,48,64]):使用的是上一次旧的数据
                corr(tensor[1,1,196,48,64]): corr = CorrBlock(droid.filterx.fmap[None,[0]], gmap[None,[0]])(coords0)  以下是参数解释
                    coords0(tensor[1,1,48,64,2]):创建的临时变量，最后一维度为(0,0)(1,0)(2,0)(3,0)...(63,0)(0,1)(1,1)(2,1)...(63,1)...
                    droid.filterx.fmap(tensor[2,128,48,64]):使用的是上一次旧数据
                    gmap(tensor[2,128,48,64]):创建的临时变量，使用的是由新的image通过转换成inputs,droid.filterx._feature_encoder(inputs)得到
                delta(tensor[1,1,48,64,2]):得到每个像素deltax,deltay的光流数值
                weight(tensor[1,1,48,64,2]):得到每个像素deltax,deltay的权重，但后续没有使用
``          droid.filterx(MotionFilter)``   更新的元素
                droid.filterx.count:一直在0和接近并包括1的数字变化
                droid.filterx.net(tensor[1,128,48,64]):通过droid.filterx.__context_encoder(inputs[:,[0]])得到，也就是通过新image
                droid.filterx.inp(tensor[1,128,48,64]):通过droid.filterx.__context_encoder(inputs[:,[0]])得到，也就是通过新image
                droid.filterx.fmap(tensor[2,128,48,64]):通过droid.filterx._feature_encoder(inputs)得到gmap,再赋值给fmapa
                *droid.video.append() 将droid.video.counter使用droid.video.get_lock()锁定,以下是参数列表解释*
                    index=droid.video.counter 这里counter还没有更新，还是老旧的数据
                    tstamp: 循环的t
                    image[0]取image的第0维的第0个，不要第一个
                    Id: None
                    depth: None
                    intrinsics / 8.0: tensor[4](32.,32.,32.,24)
                    gmap[2,128,48,64]: 值是满的并且混乱,来自于更新的droid.filterx.__feature_encoder(inputs)，包含了image[2,3,384,512]
                    net[0]:tensor[128,48,64]: 值是满的并且混乱,与更新的droid.filterx.net值相同
                    inp[0]:tensor[128,48,64]: 值是满的并且混,与更新的droid.filterx.inp值相同
``                      droid.video(DepthVideo)``   更新的元素
                            droid.video.counter -> droid.video.counter=index+1
                            droid.video.tstamp(tensor[512]) -> 第index项为循环的t，通过这个元素可以得到第几帧被作为关键帧
                                例如: droid.video.tstamp=[0,2,4,6 ...]，则第0，2，4，6帧被作为关键帧
                            droid.video.images(tensor[512,3,384,512]) -> 第[index,:,:,:]维被赋值image[0]
                            droid.video.poses(tensor[512,7],每一行为(0,0,0,0,0,0,1)) 没变化
                            droid.video.disps(tensor[512,48,64],全为1) 没变化
                            droid.video.disps_sens（tensor[512,48,64],全为0 没变化
                            droid.video.intrinsics(tensor[512,4]) -> 第[index,:]维被赋值 intrinsics / 8.0: tensor[4](32.,32.,32.,24)
                            droid.video.fmaps (tensor[512,2,128,48,64],全为0) -> 第[index,:,:,:,:]赋值gmap[2,128,48,64]: 值是满的并且混乱
                            droid.video.nets(tensor[512,128,48,64],全为0) -> 第[index,:,:,:]赋值net[0]:tensor[128,48,64]: 值是满的并且混乱
                            droid.video.inps(tensor[512,128,48,64],全为0) -> 第[index,:,:,:]赋值inp[0]:tensor[128,48,64]: 值是满的并且混乱
**初始化 initialize,t=14,droid.video.counter.value==warmup=8**
    *droid.frontend()->droid.frontend._initialize()*
``      droid.frontend(DroidFrontend)``
            droid.frontend.t0=0 没有变化
            droid.frontend.t1=0 -> droid.video.counter.value
            *droid.frontend.graph.add_neighborhoond_factors(droid.frontend.t0, droid.frontend.t1, r=3)*
                c=1 if self.video.stereo else 0
                ii(tensor[64]):临时变量值为(0,0,0,0,0,0,0,0,1,1,1,...,7)
                jj(tensor[64]):临时变量值为(0,1,2,3,4,5,6,7,0,1,2,...,7)
                *droid.frontend.graph.add_factors(ii[keep], jj[keep]) 以下是参数解释*
                    keep(tensor[64],bool):取ii中对应的jj元素的差值在(c,r]之间为True，否则为False
                    ii[keep](tensor[22]):以keep为指标过滤ii(tensor[64])，保留True的元素
                    jj[keep](tensor[22]):以keep为指标过滤jj(tensor[64])，保留True的元素
                        *ii,jj=droid.frontend.graph._filter_repeated_edges(ii[keep],jj[keep])去除和之前重复的边，不是去除{1,2},{2,1}这种*
                        *判定是否需要remove多余的元素*
``                      droid.frontend.graph(FactorGraph)`` 更新的元素
                            droid.frontend.graph.net(None) -> droid.frontend.graph.net(tensor[1,22,128,48,64])
                                .to(droid.frontend.graph.device)
                                来自于droid.video.nets[ii[keep]]的第[ii[keep],:,:,:]维度，满数据
                                第1维在net是None的时候会是22，后面会变成与之前的droid.frontend.graph.net在第1维的叠加
``                          droid.frontend.graph.corr(CorrBlock)`` None的对象被另一个CorrBlock类所创建
                                droid.frontend.graph.corr(None) -> droid.frontend.graph.corr(CorrBlock)
                                    更新通过直接cat添加
                                    droid.frontend.graph.corr.num_levels=4
                                    droid.frontend.graph.radius=3
                                    droid.frontend.graph.corr_pyramid 包含四个tensor对象
                                       每一个tenosr维度为分别为[22,48,64,48,64],[22,48,64,24,32],[22,48,64,12,16],[22,48,64,6,8]
                                       c 依次比较ii[keep],jj[keep]里面的元素，如果一样则为真，1，否则为假，0
                                       droid.video.fmaps1(tensor[1, 22, 128, 48, 64])来自于droid.video.fmaps(tensor[512,2,128,48,64])[ii[keep],0]
                                       droid.video.fmaps2(tensor[1, 22, 128, 48, 64])来自于droid.video.fmaps(tensor[512,2,128,48,64])[jj[keep],c]
                                       通过对droid.video.fmaps1和droid.video.fmaps2特征图相乘得到相关性，再进行压缩维度和平均池化得到四个corr_pyramid对象
                            droid.frontend.graph.inp(None) -> droid.frontend.graph.inp(tensor[1,22,128,48,64])
                                来自于droid.video.inps(tensor[512,128,48,64])[ii]，并且头部增加一维
                                第1维在droid.frontend.graph.inp是None的时候会是22，后面会变成与之前的droid.frontend.graph.inp在第1维的叠加
                            droid.frontend.graph.target(tensor[1,0,48,64,2],全0)
                            -> (tensor[1,22,48,64,2]),从ii[keep]投影到jj[keep]像素,后续沿着第一维度使用cat增加
                                *target, _=droid.video.reproject(ii[keep],jj[keep])*
                                *得到return,需要pops.projective_transform函数有关的值*
                                    *coords, valid_mask=import geom.projective_ops as pops.projective_transform*
                                    *(Gs,droid.video.disps[None],droid.video.intrinsics[None],ii,jj)*
                                    *得到return，需要与 iproj函数，actp函数，proj函数有关的值*
                                        Gs(tensor[1,512,7]):第一维每一行为[0,0,0,0,0,0,1]=poses                                      
                                        droid.video.disps(tensor[512,48,64],全为1)[None] -> tensor[1,512,48,64] = depths
                                        droid.video.intrinsics(tensor[512,4])[None] -> tensor[1,512,4] = intrinsics
                                        ii(tensor[22]):就是之前的ii[keep] 
                                        jj(tensor[22]):就是之前的jj[keep] 
                                            *X0, Jz = iproj(depths[:,ii], intrinsics[:,ii], jacobian=jacobian)*
                                                depths[:,ii](tensor[1,22,48,64]) = disps 
                                                intrinsics[:,ii](tensor[1,22,4]) = intrinsics
                                                    return Jz : None
                                                    return pts = X0:
                                                        y(tensor[48,64]):每一列从上到下0-47,即为disps的第二维
                                                        x(tensor[48,64]):每一行从左到右0-63,即为disps的第三维
                                                        i(tensor[1,22,48,64]):全是1
                                                        Y(tensor[1,22,48,64]):归一化计算，通过(y-cy) / fy,没有乘Z，否则是相机坐标系3D坐标
                                                        X(tensor[1,22,48,64]):归一化计算，通过(x-cx) / fx,没有乘Z，否则是相机坐标系3D坐标
                                                        pts(tensor[1,22,48,64,4]):最后一维分别是:X,Y,i,disps
                                            Gij(tensor[1,22,7])
                                                poses(tensor[1,512,7])[:,jj] * poses(tensor[1,512,7])[:,ii].inv() 
                                                If ii与jj中的元素完全一致，则将它们对应的Gij第一维值设置为[-0.1, 0.0, 0.0, ..., 1.0]
                                                    即把ii[keep]对应的poses与jj[keep]对应的poses进行处理，可以批量处理
                                                    注意！由于这里poses为SE3的矩阵，因此可以对[1,22,7]维度的tensor数据求逆，会自动转换成方阵
                                            *X1, Ja = actp(Gij, X0, jacobian=jacobian)*
                                                Gij(tensor[1,22,7]):临时变量得到的旋转矩阵
                                                X0(tensor[1,22,48,64,4]):iproj函数返回的临时变量pts
                                                    return Ja: None 
                                                    return X1(tensor[1,22,48,64,4]): 
                                                        = Gij[:,:,None,None](tensor[1,22,7,1,1]) * X0
                                                        即将变换矩阵Gij 乘以 包含归一化计算的X,Y和深度初始值的X0
                                            *x1, Jp = proj(X1, intrinsics[:,jj], jacobian=jacobian, return_depth=return_depth)*
                                                X1(tensor[1,22,48,64,4]) = Xs
                                                intrinsics[:,jj](tensor[1,22,4]) : 选择jj[keep]中的intrinsics 
                                                jacobian: False
                                                return_depth: False
                                                    return Jp : None
                                                    return coords(tensor[1,22,48,64,2]) = x1:投影到jj序列对应的像素坐标值
                                                        Z(tensor[1,22,48,64]) 全是1
                                                            就是上一步的i(tensor[1,22,48,64]):全是1,如果其中元素< MIN_DEPTH=0.2，设1
                                                        D(tensor[1,22,48,64])  
                                                            上面的depths[:,ii](tensor[1,22,48,64]) = disps
                                                            depths从droid.video.disps(tensor[512,48,64])中来
                                                        coords(tensor[1,22,48,64,2])
                                                            x(tensor[1, 22, 48, 64]) = fx * (X * d) + cx
                                                            y(tensor[1, 22, 48, 64]) = fy * (Y * d) + cy
                                                            coords沿着最后一维叠加
                                            valid(tensor[1,22,48,64,1])
                                                float类型的bool，要么0，要么1
                                                如果 X1(tensor[1,22,48,64,4]) 和 X0 (tensor[1,22,48,64,4]) 最后一维度的第3个元素，即相机坐标系下的Z，大于MIN_DEPTH=0.2，则为真
                                                这里全为1
                                            *return x1 = coords = add_factors函数中droid.frontend.graph.target, valid = valid_mask*
                                                x1(tensor[1,22,48,64,2])    
                                                    ！！！x1 = coords = add_factors函数中droid.frontend.graph.target 的原型
                                                    新投影的jj[keep]里的像素值，其实这里相比于ii[keept]设定的像素值没有变化，因为两个序列的位置参数都为单位参数，没有发生旋转
                                                valid(tensor[1,22,48,64,1])
                            droid.frontend.graph.weight(tensor[1,0,48,64,2],全为0) 
                            -> tensor[1,22,48,64,2],维度与droid.frontned.graph.targt相同，并且全为0
                            -> 之后沿着第一维使用cat更新
                            droid.frontend.graph.ii(tensor[22]) 
                            -> 经过*droid.frontend.graph._filter_repeated_edges(ii[keep],jj[keep])*去除和之前重复的边，不是去除{1,2},{2,1}..
                            -> 之后沿着第0维使用cat更新
                            droid.frontend.graph.jj(tensor[22]) 
                            -> 经过*droid.frontend.graph._filter_repeated_edges(ii[keep],jj[keep])*去除和之前重复的边，不是去除{1,2},{2,1}..
                            -> 之后沿着第0维使用cat更新
                            droid.frontend.graph.age(tensor[]) 
                            -> dorid.frontend.graph.age(tensor[22]),全为0，与dorid.frontend.graph.ii(tensor[22]) 维度相同
                            -> 之后沿着第0维使用cat更新
            *droid.frontend.graph.update(t0 = 1, t1 = None, itrs=2,use_inactive=True,EP=1e-7,motion_only=False)*
            *对droid.frontend.graph的元素进行循环8次循环,从而更新*
``              droid.frontend.graph(FactorGraph)`` 更新的元素
                    droid.frontend.coords0(tensor[48,64,2],第:,:,0维度每一行从左往右0-47，第:,:,1维度每一列从上到下0-63) - 没有变化
                    droid.frontend.graph.inp(tensor[1,22,128,48,64])
                    - 来自于droid.video.inps(tensor[512,128,48,64])[ii]，并且头部增加一维
                    - 特殊！每次循环没有变化，但是被用在了很重要的droid.frontend.graph.update_op函数中
                    droid.frontend.graph.age(tensor[22]) -> 每次循环 数值+1
                    *droid.frontend.graph.net, delta, weight, damping, upmask = droid.frontend.graph.update_op*
                    *(droid.frontend.graph.net,droid.frontend.graph.inp,corr,motn,droid.frontend.graph.ii,droid.frontend.graph.jj) 主更新函数*
                        droid.frontend.graph.net(tensor[1,22,128,48,64]) : 来自于droid.video.nets[ii[keep]]的第[ii[keep],:,:,:]维度,满数据
                        droid.frontend.graph.inp(tensor[1,22,128,48,64]) : 来自于droid.video.inps(tensor[512,128,48,64])[ii]，且头部增一维
                        corr(tensor[1, 22, 196, 48, 64]):临时变量，来自于corr.py中的corr(coords1)函数
                            coords1(tensor[1, 22, 48, 64, 2])
                                来自于droid.video.reproject(droid.frontend.graph.ii,droid.frontend.graph.ii)函数
                                每次都会更新，因为每次循环的时候这个函数一个很重要的变量droid.video.poses(tensor[512,7])会变化，导致重投影结果变化
                        motn(tensor[1, 22, 4, 48, 64])
                            临时变量，与每次更新的coords1(tensor[1, 22, 48, 64, 2])，每次更新的droid.frontend.graph.target有关
                            再重新排列维度并且限制数值在(-64,64),变成(tensor[1,22,4,48,64])
                        droid.frontend.graph.ii(tensor[22]) 
                            经过*droid.frontend.graph._filter_repeated_edges(ii[keep],jj[keep])*去除和之前重复的边，不是去除{1,2},{2,1}..
                        droid.frontend.graph.jj(tensor[22]) 
                            经过*droid.frontend.graph._filter_repeated_edges(ii[keep],jj[keep])*去除和之前重复的边，不是去除{1,2},{2,1}..
                        delta(tensor[1, 22, 48, 64, 2]) : 目标跟踪光流像素的变化值，就是论文中的rij,通过网络预测得到,每次循环更新
                        weight(tensor[1, 22, 48, 64, 2]) : 目标跟踪光流像素预测值的权重，每次循环更新
                        damping(tensor[1,8,48,64])  
                            阻尼系数 (damping factor), 为了Levenberg-Marquardt算法的收敛性和稳定性，每次循环更新，神经网络得到的结果
``                      droid.frontend.graph(FactorGraph)`` 更新的元素
                            droid.frontend.graph.target(tensor[1,22,48,64,2]) 
                            -> coords1 + delta 维度(tensor[1,22,48,64,2]), dtype=torch.float
                            -> 很关键的更新元素，实现了论文中'pij*'变量
                            droid.frontend.graph.weight(tensor[1,0,48,64,2],全为0) -> 临时变量weight赋值给它，并且更新
                            droid.frontend.graph.damping(tensor[512,48,64],全为1e-6) 
                            -> 选取droid.frontend.graph.ii中不重复的元素成为droid.frontend.graph.damping第0维的index
                            -> 再用damping(tensor[1,8,48,64])赋值给它,当然了这里不重复的元素也有8个
                            -> 维度依然是tensor([512,48,64])
                            droid.frontend.graph.net(tensor[1,22,128,48,64]) -> 改变数值，不改变维度
                    *droid.frontend.graph.video.py_ba=*
                    *(target,weight,damping,ii,jj,t0,t1,itrs=itrs,lm=1e-4,motion_only=motion_only) 核心代码*
                        target(tensor[1,22,48,64,2]):
                            if use_inactive,沿第一维度给droid.frontend.graph.target(tensor[1,22,48,64,2])加部分droid.frontend.graph.target_inac(tensor[1,22,48,64,2]) 
                            否则直接是droid.frontend.graph.target(tensor[1,22,48,64,2])
                        weight(tensor[1,0,48,64,2])
                            if use_inactive,则沿第一维度给droid.frontend.graph.weight(tensor[1,0,48,64,2])加droid.frontend.graph.tweight_inac(tensor[1,22,48,64,2])
                        damping(tensor[8, 48, 64]):
                            找到ii[keep]序列中unique的数，以此为索引筛选 droid.frontend.graph.damping(tensor[512,48,64])第0维度对应的元素。最后乘0.2加上一个常数EP
                        ii(tensor[22])
                            if use_inactive,沿第0维度给droid.frontend.graph.ii(tensor[22])加部分droid.frontend.graph.ii_inac
                            否则直接是droid.frontend.graph.ii(tensor[22])
                        jj(tensor[22])
                            if use_inactive,沿第0维度给droid.frontend.graph.jj(tensor[22])加部分droid.frontend.graph.jj_inac
                            否则直接是droid.frontend.graph.jj(tensor[22])
                        t0:1
                        t1:None
                            *poses, disps = BA(target,weight,damping,poses,disps,self.intrinsics[None],ii,jj,fixedp=t0)*
                            *最核心BA代码,包含了公式组合和求解，循环两次*
                                target(tensor[1,22,48,64,2]):py_ba函数的参数
                                weight(tensor[1,0,48,64,2]):py_ba函数的参数
                                damping(tensor[8,48,64]):py_ba函数的参数
                                poses(tensor[1,8,7]):
                                    t1:ii[keep]和jj[keep]的最大维度再+1,这里为8
                                    使用droid.video.poses(tensor[512,7])创建一个lietorch.SE3对象，并且取前t1行，再增加维度
                                disps(tensor[1, 8, 48, 64]):取droid.video.disps(tensor[512,48,64])前t1行，再增加维度
                                intrinsics[None](tensor[1,512,4]):来自于droid.video.intrinsics(tensor[512,4])
                                ii(tensor[22]):py_ba函数的参数
                                jj(tensor[22]):py_ba函数的参数
                                fixedp=t0:py_ba函数的参数
                                    *coords, valid, (Ji, Jj, Jz) = pops.projective_transform(poses, disps, intrinsics, ii, jj, jacobian=True)*
                                    *构建Jacobian矩阵，求解变化量*
                                        自变量全为BA函数内传递的参数
                                            coords(tensor[1, 22, 48, 64, 2]):
                                                创造两个矩阵x,y，他们分别是每一列从0到63，每一行从0到47代表初始化图像的像素点坐标x,y，我称之为初始网格像素坐标，包含
                                                将初始网格像素坐标通过相机内参转换成归一化平面坐标上的坐标，并且设置Z为1，disps视差为1，这个部分共四维，分别为X,Y,Z,W,正如论文所说，称之为归一化坐标
                                                创造Gij，即poses[:,jj] * poses[:,ii].inv()。每次迭代由于poses更新了因此Gij都会更新
                                                将归一化坐标通过Gij转换成新的归一化坐标，再转换到更新网格像素坐标。注意，如果是立体视觉，更新网格像素坐标的第二维应该是更新后的视差坐标除以更新后的Z
                                                这个更新后的网格像素表格做就是coords
                                            valid(tensor[1, 22, 48, 64, 1])
                                                只有旧的和新的归一化坐标的第2维度，即Z都大于MIN_DEPTH,对应的值是float类型的真
                                            Ji, Jj, Jz
                                            

                                            



                                




                        
                        
                        


                        

                    










                                                    
                                                        





                                                                                                           

                                                    




                                    


                            



                                                         
                                



                        





            



                            



                    











