### Abstract

生成对抗网络可以在狭窄的区域生成接近真实照片的图像。这个狭窄的区域指的是图像比较单一，比如这些图像都是人脸照片。



本文从和密度估计技术中得到启发，引入了一种非参数方法来建模复杂数据集的分布。

我们将数据流形划分为由数据点及其最近邻描述的重叠邻域的混合物，并引入一个称为实例条件GAN（IC-GAN）的模型，该模型学习每个数据点周围的分布。



 Experimental results on ImageNet and COCO-Stuff show that
IC-GAN significantly improves over unconditional models and unsupervised data
partitioning baselines.

此外，我们还表明，IC-GAN可以通过简单地改变条件实例，轻松地传输到训练过程中看不到的数据集 ，仍然可以生成逼真的图像。 最后，我们将IC-GAN扩展到类条件情况



### Introduction

生成性对抗网络（GANs）[18]在无条件图像生成方面取得了令人印象深刻的成果 

GAN存在优化困难，可能会出现模式崩溃，导致生成器无法获得良好的分布覆盖率，并经常生成质量差和/或多样性低的生成样本。  GANs present optimization difficulties and can suffer from mode collapse,resulting in the generator not being able to obtain a good distribution coverage, and often producing poor quality and/or low diversity generated samples.



Classconditional  GANs[5,38,39,55]通过在类标签上设置条件，有效地划分数据，简化了学习数据分布的任务。尽管它们提供的样品质量高于无条件样品，但它们需要有标签的数据，这些数据可能无法获得，或者获取成本高昂。