## 课题来源及选题依据

自选课题，根据个人专长及兴趣，结合所学专业，通过查阅相关资料并在导师的指导

### 选题的目的与意义：

#### 研究目的：

伴随着计算机技术、信息处理技术和视觉通信技术的高速发展，人类进入了一个全新的信息化时代。人们所能能够获取的知识量呈爆炸式的增长，因此迫切的要求信息处理技术不断的完善和发展，以便能够为人们提供更加方便、快捷和多样化的服务。数字图像及其相关处理技术是信息处理技术的重要内容之一，在很多领域得到了越来越广泛的应用。对于数字图像在一些情况下一般要求是高分辨图像，如:医学图像要求能够显示出那些人眼不能辨别出的细微病灶;卫星地面要求卫星图像至少能够辨别出人的脸相甚至是证件;有些检测识别控制装置需要足够高分辨率的图像才能保证测量和控制的精度。因此提高图像分辨率是图像获取领域里追求的一个目标。

### 研究意义

1970年以来，CCD和CMOS图像传感器广泛的被用来获取数字图像，在很多的应用场合，需要获取高分辨图像，提高图像分辨率最直接的方法是提高成像装置的分辨力，但是受传感器阵列排列密度的限制，提高传感器的空间分辨率越来越难，通常采用的方法是减少单位像素的尺寸(即增加单位面积内的像素数量)，对于数字摄机，比如CCD，就是减少其传感单元的尺寸从而提高传感器的阵列密度，使其能够分辨出更多场景细节。但是这样将导致数字摄像机的价格大幅度提高。技术工艺的制约也限制了图像分辨率的进一步提高。事实上随着像素尺寸的减少，每个像素接收到的光照强度也随之降低，传感器自身的噪声将严重影响图像的质量，造成拍摄的影像信噪比不高，因此，像素尺寸不可能无限制的降低，而是有下限的，当CCD传感器阵列密度增加到一定程度时，图像的分辨率不但不会提高反而会下降。

减小传感器中的像素尺寸，提高阵列密度 

​         一方面技术工艺限制，另一方面当像素尺寸减小到一定程度时，加性噪声几乎维持不变，有效信号的能量将随传感器像素尺寸成比例减小，导致所形成图像的信噪比下降，退化反而加重

增大成像阵列芯片的面积 

​         应用这种尺寸较大的高精度光学传感器将会显著增加成本，电荷转移速率下降，给成像设备的普及带来重要阻碍 

​    那么还有其他方面的问题才导致我们选择超分辨率重建这条道路吗？

​       1：成像过程中各种噪声，场景运动，模糊等，太多的因素会使图像退化降质了，而SR软件的算法具有相当的灵活性，适应性强

​       2: 深刻分析了成像过程，成像模型的建立，使得超分辨率重建算法能够实现成为可能。

​    目前超分辨率重建的应用：

​       1：数字信号DTV转化为高清晰度电视HDTV

​       2：遥感军事应用，帮助气象检测、地理环境分析、军事保护等

​       3：医学成像上，帮助病情分析、病体定位等

​       4：视频监控，提高公共安全、协助破案等

### 与选题有关的国内外研究现状

图像超分辨率率(super resolution,SR)是指由一幅低分辨率图像(low resolution,LR)或图像序列恢复出高分辨率图像(high resolution,HR)。HR意味着图像具有高像素密度，可以提供更多的细节，这些细节往往在应用中起到关键作用。要获得高分辨率图像，最直接的办法是采用高分辨率图像传感器，但由于传感器和光学器件制造工艺和成本的限制，在很多场合和大规模部署中很难实现。因此，利用现有的设备，通过超分辨率技术获取HR图像具有重要的现实意义。

#### （一）国外研究现状

　　超分辨率概念最早出现在光学领域。在该领域中，超分辨率是指试图复原衍射极限以外数据的过程。Toraldo di Francia在1955年的雷达文献中关于光学成像第一次提出了超分辨率的概念。复原的概念最早是由J.L.Harris和J.w.Goodman分别于1964年和1965年提出一种称为Harris-Goodman频谱外推的方法。这些算法在某些假设条件下得到较好的仿真结果，但实际应用中效果并不理想。Tsai&Huang首先提出了基于序列或多帧图像的[超分辨率重建](https://so.csdn.net/so/search?from=pc_blog_highlight&q=超分辨率重建)问题。1982，D.C.C.Youla和H.Webb在总结前人的基础上，提出了凸集投影图像复原(Pocs)方法。1986年，S.E.Meinel提出了服从泊松分布的最大似然复原(泊松-ML)方法。1991年和1992年，B.R.Hunt和PJ.Sementilli在Bayes分析的基础上，提出了泊松最大后验概率复原(泊松-MAP)方法，并于1993年对超分辨率的定义和特性进行了分析，提出了图像超分辨率的能力取决于物体的空间限制、噪声和采样间隔。

　　近年来，图像超分辨率研究比较活跃，美国加州大学Milanfar等人提出的大量实用超分辨率图像复原算法， Chan等人从总变差正则方面，Zhao等人、Nagy等人从数学方法、多帧图像的去卷积和彩色图像的超分辨率增强方面，对超分辨率图像恢复进行了研究。Chan等人研究了超分辨率图像恢复的预处理迭代算法。此外，Elad等人对包含任意图像运动的超分辨率恢复进行了研究；Rajan和Wood等人分别从物理学和成像透镜散射的角度提出了新的超分辨率图像恢复方法；韩国Pohang理工大学对各向异性扩散用于超分辨率。Chung-Ang图像科学和多媒体与电影学院在基于融合的自适应正则超分辨率方面分别进行了研究。Yang等人提出了使用图形块的稀疏表示来实现超分辨率。他们从一些高分辨率图像中随机选取一些块组成一个过完备的词典，接着对于每一个测试块，通过线性规划的方法求得该测试块在这个过完备的词典下的稀疏表示，最后以这组系数加权重构出高分辨率的图像，这种方法克服了邻域嵌入方法中对于邻域大小的选择问题，即在求解稀疏表示的时候，无需指定重构所需要基的个数，其表示系数和基的个数将同时通过线性规划求解得到。然而，目前该方法的缺陷就在于过完备词典的选择，随机的选择只能实现特定领域的图像的超分辨率，对于通用图像的超分辨率效果较差。

#### （二）国内研究现状

　　国内许多科研院所和大学等对超分辨率图像恢复进行研究，其中部分是关于频谱外推、混叠效应的消除，其他主要是对国外超分辨率方法所进行的改进，包括对POCS算法和MAP算法的改进，对超分辨率插值方法的改进，基于小波域隐马尔可夫树(HMT)模型对彩色图像超分辨率方法的改进以及对超分辨率图像重构方法的改进。

　　2016年香港中文大学Dong等人将卷积神经网络应用于单张图像超分辨率重建上完成了深度学习在图像超分辨率重建问题的开山之作SRCNN(Super-Resolution Convolutional Neural Network)。SRCNN将深度学习与传统稀疏编码之间的关系作为依据，将3层网络划分为图像块提取(Patch extraction and representation)、非线性映射(Non-linear mapping)以及最终的重建(Reconstruction)。重建效果远远优于其他传统算法，利用SRCNN进行超分辨率图像重建与使用其他方法进行超分辨率重建的效果对比图如下图1所示。

## 研究方案与技术路线

### 1、 研究内容及关键技术

#### 1.1研究内容

#### 1.2关键技术

### 2 拟采取的研究方法、技术路线、实施方案及可行性分析