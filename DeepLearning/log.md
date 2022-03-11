论文

| 论文题目                                                     | 论文连接                                              | 注                                   |
| ------------------------------------------------------------ | ----------------------------------------------------- | ------------------------------------ |
| Generating Hyperspectral Skin Cancer Imagery using Generative Adversarial Neural Network | https://ieeexplore.ieee.org/abstract/document/9176292 | 用生成对抗网络生成皮肤癌高光谱图像。 |
|                                                              |                                                       |                                      |
|                                                              |                                                       |                                      |

gan的应用：

- 生成照片
- 图像转换
- 语义图像-图片转化
- 图片混合··
- 超分辨率
- 图片修复
- 服装转化
- 3D打印

ref：[link](https://cloud.tencent.com/developer/article/1528648)



准备试一下pix2pix。https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

colab [link](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb#scrollTo=mey7o6j-0368)

直接可以看效果



运行了gan网络，生成了手写字体。并具体的查看了它的网络结构，大量使用了全连接，并没有使用卷积。

运行了dcgan网络，生成了手写字体。效果确实比最初的gan好很多。



下载并查看了`indian Pines`数据，了解了高光谱图像的基础。



看了一篇利用生成对抗网络去给高光谱图像分类的论文，这篇论文提出了一种半监督学习的gan`RMGAN`，作者将它提出的`RMGAN`与另外一篇早前提出的高光谱图像分类gan（`HSGAN`）做了对比实验，他的改进之处是：`HSGAN`是基于`1DGAN`的，并没有充分利用空间信息，为了克服`HSGAN`的不足，作者借助了残差学习引入了更深的网络，以此来构成gan网络中的生成器。生成高分辨率的图像，便于鉴别器提取高光谱图像特征。在鉴别器中就可以借助更多尺度特征匹配层来聚合中高层信息，就可以提取更多的有用信息。



先学习普通高光谱图像的处理：可视化、印度松树HSI的可视化像素等等。



Generative Adversarial Networks for Hyperspectral Image Classification 没开源代码

nainaide , 好多代码都没开源。





这篇文章感觉还可以 [link](https://ieeexplore.ieee.org/document/8518681)

STATE–OF–THE–ART AND GAPS FOR DEEP LEARNING ON LIMITED TRAINING DATA IN REMOTE SENSING

深度学习通常在数量和多样性方面都需要大数据。然而，大多数遥感应用的训练数据有限，其中有一个小子集被标记。在此，我们回顾了应对这一挑战的深度学习的三种最先进的方法。第一个主题是转移学习，其中一个领域的某些方面，例如功能，被转移到另一个领域。接下来是无监督学习，例如自动编码器，它对无标签数据进行操作。最后是生成对抗网络，它可以生成逼真的数据，可以愚弄深度学习网络和人类等。本文旨在提高人们对这一困境的认识，引导读者了解现有工作，并突出当前需要解决的差距。



看到一篇综述：[link](https://ieeexplore.ieee.org/document/8738045)

# Deep Learning for Classification of Hyperspectral Data: A Comparative Review

摘要：

近年来，深度学习技术彻底改变了遥感数据的处理方式。高光谱数据的分类也不例外，但它具有内在的特殊性，这使得深度学习的应用不像其他光学数据那样简单。本文介绍了之前机器学习方法的最新技术，回顾了目前为高光谱分类提出的各种深度学习方法，并确定了为这项任务实施深度神经网络时出现的问题和困难。特别是，讨论了空间和光谱分辨率、数据量以及模型从多媒体图像传输到高光谱数据的问题。此外，还提供了对各种网络架构家族的比较研究，并公开发布了一个软件工具箱，以便对这些方法进行实验（https://github.com/nshaud/DeepHyperX）。本文既适用于对高光谱数据感兴趣的数据科学家，也适用于渴望将深度学习技术应用于自身数据的遥感专家



看到一篇文章csdn，讲的是高光谱图像分类

https://blog.csdn.net/qq_41683065/article/details/100748883



Hyperspectral_classfication



最后还是使用这个库进行实验。

https://github.com/nshaud/DeepHyperX



又是被这个库给卡住了。

ERROR:root:Error [Errno 54] Connection reset by peer while downloading https://unpkg.com/react-dom@16.2.0/umd/react-dom.production.min.js



看了这篇文献比较好

今日求助第一篇：

Generative Adversarial Networks and Conditional Random Fields for Hyperspectral Image Classification

**DOI:** 10.1109/TCYB.2019.2915094

但是不能下载



看了这个人的介绍与发表的论文，感觉在高光谱这一块研究的很深。



有一篇好文章不能下载

https://ieeexplore.ieee.org/document/8661744

Classification of Hyperspectral Images Based on Multiclass Spatial–Spectral Generative Adversarial Networks

**DOI:** [10.1109/TGRS.2019.2899057](https://doi.org/10.1109/TGRS.2019.2899057)





<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-6954409989927492"
     crossorigin="anonymous"></script>