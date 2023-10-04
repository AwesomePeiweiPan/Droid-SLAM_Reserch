#####运行环境
使用 Seasidetown 的P000 作为输入图像，并采用Stereo
image[2,3,384,512]
#####Before Warmup
**demo.py** 
t=0
**motion_filter.py** 
self.video.append(...[2]=Id...[3]=1...) counter=1
*Success*

**demo.py** 
t=1
**motion_filter.py** 
delta.norm(dim=-1).eamn.item()<self.thresh
*Fail*

**demo.py** 
t=2
**motion_filter.py**
self.video.appen(...[2]=None...[3]=None) counter=2
*Success*

**简而言之**
t=3  Fail
t=4  Succ, counter=3 
t=5  Fail
t=6  Succ, counter=4
t=7  Fail
t=8  Succ, counter=5
t=9  Fail
t=10 Succ, counter=6
...
t=14 Succ, counter=8=warmup!!!!!!


#####重点关注问题
7.demo.py中，42行，D，Z代表的意思。我猜测Z是虚拟的给定的深度，D确实是真实深度，但是为什么后面需要D*d？jacobian也用了
8.parser.add_argument("--upsample", action="store_true")