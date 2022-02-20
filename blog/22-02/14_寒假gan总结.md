论文：

- Generative Adversarial Nets
- DCGan
- acgan
- A Review on Generative Adversarial Networks
- 2021_ProteinGAN



视频：

看了李宏毅的gan教程，做了相关的笔记。主要是对基础知识的巩固。



蛋白质gan：

目前看的是google的alphafold，这个网络比较大，对硬件的要求比较高，所以也不能自己实现，目前还知识停留在理论阶段。有点难，看的一知半解。主要用到了下面的技术：

- MSA：多序列对齐，这个技术用于从一个大的数据库中抽取和输入氨基酸序列相近的序列，并且顺便进行对齐。
- 特征构造：通过同源序列和模板表示成深度学习可以作为输入的矩阵结构
- 广泛运用了Attention架构。一个二维的表可以横着做再竖着做attention
- 训练神经网络来对regression target进行逐步迭代精化



一些gan项目的运行

https://github.com/eriklindernoren/PyTorch-GAN

将这个仓库里面的代码自己运行了下。只带了公司的电脑回家，所以没有运行环境。在云服务器上面运行的，修改代码不方便所以进展不大。

运行了两个网络。一个是acgan，一个是dcgan，数据也都是用的手写字体。