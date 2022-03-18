Abstract

在本文中，我们介绍了用于改进图像合成的生成对抗网络 (GAN) 训练的新方法。 我们构建了一个使用标签条件的 GAN 变体，该变体导致 128 × 128 分辨率的图像样本表现出全局相干性。 我们扩展了先前的图像质量评估工作，以提供两种新的分析方法来评估来自类条件图像合成模型的样本的可辨别性和多样性。 这些分析表明，高分辨率样本提供了低分辨率样本中不存在的类别信息。 在 1000 个 ImageNet 类中，128 × 128 样本的可识别性是人工调整大小的 32 × 32 样本的两倍多。 此外，84.7% 的类别的样本表现出与真实 ImageNet 数据相当的多样性。

Introduction

Characterizing the structure of natural images has been a rich research endeavor. Natural images obey intrinsic invariances and exhibit multi-scale statistical structures that have historically been difficult to quantify (Simoncelli & Olshausen, 2001). Recent advances in machine learning offer an opportunity to substantially improve the quality of image models. Improved image models advance the state-of-the-art in image denoising (Balle ́ et al., 2015), compression (Toderici et al., 2016), inpainting (van den Oord et al., 2016a), and super-resolution (Ledig et al., 2016). Better models of natural images also improve performance in semi-supervised learning tasks (Kingma et al., 2014; Springenberg, 2015; Odena, 2016; Salimans et al., 2016) and reinforcement learning problems (Blundell et al., 2016).

表征自然图像的结构一直是一项丰富的研究工作。 自然图像遵循内在不变性，并表现出历来难以量化的多尺度统计结构（Simoncelli & Olshausen，2001）。 机器学习的最新进展为大幅提高图像模型的质量提供了机会。 改进的图像模型在图像去噪 (Balle ́ et al., 2015)、压缩 (Toderici et al., 2016)、修复 (van den Oord et al., 2016a) 和超 分辨率（Ledig 等人，2016 年）。 更好的自然图像模型还可以提高半监督学习任务（Kingma 等人，2014；Springenberg，2015；Odena，2016；Salimans 等人，2016）和强化学习问题（Blundell 等人，2016）的性能。





One method for understanding natural image statistics is to build a system that synthesizes images de novo. There are several promising approaches for building image synthesis models. Variational autoencoders (VAEs) maximize a variational lower bound on the log-likelihood of the training data (Kingma & Welling, 2013; Rezende et al., 2014). VAEs are straightforward to train but introduce potentially restrictive assumptions about the approximate posterior distribution (but see (Rezende & Mohamed, 2015; Kingma et al., 2016)). Autoregressive models dispense with latent variables and directly model the conditional distribution over pixels (van den Oord et al., 2016a;b). These models produce convincing samples but are costly to sample from and do not provide a latent representation. Invertible density estimators transform latent variables directly using a series of parameterized functions constrained to be invertible (Dinh et al., 2016). This technique allows for exact log-likelihood computation and exact inference, but the invertibility constraint is restrictive.

理解自然图像统计的一种方法是构建一个从头合成图像的系统。有几种有前途的方法可用于构建图像合成模型。变分自动编码器 (VAE) 最大化训练数据对数似然的变分下限（Kingma & Welling, 2013; Rezende et al., 2014）。 VAE 易于训练，但引入了关于近似后验分布的潜在限制性假设（但请参阅 (Rezende & Mohamed, 2015; Kingma et al., 2016)）。自回归模型无需潜在变量，而是直接对像素上的条件分布进行建模（van den Oord et al., 2016a;b）。这些模型产生了令人信服的样本，但采样成本很高，并且不提供潜在的表示。可逆密度估计器使用一系列限制为可逆的参数化函数直接转换潜在变量（Dinh 等人，2016 年）。这种技术允许精确的对数似然计算和精确的推断，但可逆性约束是有限制的。



Generative adversarial networks (GANs) offer a distinct and promising approach that focuses on a game-theoretic formulation for training an image synthesis model (Good- fellow et al., 2014). Recent work has shown that GANs can produce convincing image samples on datasets with low variability and low resolution (Denton et al., 2015; Radford et al., 2015). However, GANs struggle to generate glob-ally coherent, high resolution samples - particularly from datasets with high variability. Moreover, a theoretical understanding of GANs is an on-going research topic (Uehara et al., 2016; Mohamed & Lakshminarayanan, 2016).

生成对抗网络 (GAN) 提供了一种独特且有前途的方法，该方法专注于用于训练图像合成模型的博弈论公式（Good- Fellow et al., 2014）。 最近的工作表明，GAN 可以在具有低可变性和低分辨率的数据集上生成令人信服的图像样本（Denton 等人，2015；Radford 等人，2015）。然而，GAN 难以生成全局连贯的高分辨率样本——尤其是来自具有高可变性的数据集。 此外，对 GAN 的理论理解是一个持续的研究课题（Uehara et al., 2016; Mohamed & Lakshminarayanan, 2016）。





In this work we demonstrate that that adding more structure to the GAN latent space along with a specialized cost function results in higher quality samples. We exhibit 128×128 pixel samples from all classes of the ImageNet dataset (Russakovsky et al., 2015) with increased global coherence (Figure 1). Importantly, we demonstrate quantitatively that our high resolution samples are not just naive resizings of low resolution samples. In particular, downsampling our 128 × 128 samples to 32 × 32 leads to a 50% decrease in visual discriminability. We also introduce a new metric for assessing the variability across image samples and employ this metric to demonstrate that our synthesized images exhibit diversity comparable to training data for a large fraction (84.7%) of ImageNet classes. In more detail, this work is the first to:

在这项工作中，我们证明了向 GAN 潜在空间添加更多结构以及专门的成本函数会产生更高质量的样本。 我们展示了来自 ImageNet 数据集 (Russakovsky et al., 2015) 所有类别的 128×128 像素样本，具有增强的全局一致性（图 1）。 重要的是，我们定量地证明了我们的高分辨率样本不仅仅是低分辨率样本的简单调整大小。 特别是，将我们的 128 × 128 样本下采样到 32 × 32 会导致视觉可辨别性降低 50%。 我们还引入了一个新的度量标准来评估图像样本的可变性，并使用这个度量标准来证明我们的合成图像表现出与大部分（84.7%）ImageNet 类的训练数据相当的多样性。 更详细地说，这项工作是第一个：

![image-20220318143545727](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220318143549.png)

- Demonstrate an image synthesis model for all 1000 ImageNet classes at a 128x128 spatial resolution (or any spatial resolution - see Section 3).

  以128x128的空间分辨率（或任何空间分辨率-见第3节）演示所有1000个ImageNet类的图像合成模型。

- Measure how much an image synthesis model actually uses its output resolution (Section 4.1).

  测量图像合成模型实际使用其输出分辨率的程度（第 4.1 节）。

- 
  Measure perceptual variability and ’collapsing’ be- havior in a GAN with a fast, easy-to-compute metric (Section 4.2).

  使用快速、易于计算的指标测量GAN中的感知可变性和“折叠”行为（第4.2节）。

- Highlight that a high number of classes is what makes ImageNet synthesis difficult for GANs and provide an explicit solution (Section 4.6).

  强调大量类使GAN难以进行ImageNet合成，并提供显式解决方案（第4.6节）。

- Demonstrate experimentally that GANs that perform well perceptually are not those that memorize a small number of examples (Section 4.3).
  通过实验证明，感知性能良好的GAN不是那些记住少量示例的GAN（第4.3节）。

- Achieve state of the art on the Inception score metric when trained on CIFAR-10 without using any of the techniques from (Salimans et al., 2016) (Section 4.4).
  在CIFAR-10上训练时，无需使用任何技术（Salimans等人，2016年）（第4.4节）。

### Background

生成对抗网络（GAN）由两个相互对立的神经网络组成。生成器G以随机噪声矢量z作为输入，并输出图像Xfake = G(z)。鉴别器D接收来自生成器的训练图像或合成图像作为输入，并在可能的图像源上输出概率分布P(S | X) = D(X)。经过训练，鉴别器可以最大限度地提高分配给正确来源的日志可能性：

生成器经过训练，以最小化方程1中的第二项。

基本的GAN框架可以使用侧面信息进行增强。一种策略是为发电机和鉴别器提供类标签，以生成类条件样品（Mirza和Osindero，2014年）。类条件合成可以显著提高生成样品的质量（van den Oord等人，2016年b）。更丰富的侧面信息，如图像说明和边界盒低钙化，可能会进一步提高样品质量（Reed等人，2016a；b）。

可以要求鉴别器重建侧面形成，而不是向鉴别器提供侧面信息。这是通过修改鉴别器以包含一个辅助解码器网络1来实现的，该网络输出训练数据的类标签（Odena，2016年；Salimans等人，2016年）或生成样本的潜在变量的子集（Chen等人，2016年）。众所周知，强迫模型执行其他任务可以改善原始任务的性能（例如（Sutskever等人，2014年；Szegedy等人，2014年；Ramsundar等人，2016年）。此外，辅助解码器可以利用预先训练的反犯罪者（例如图像分类器）来进一步改进合成图像（Nguyen等人，2016年）。在这些考虑因素的激励下，我们引入了一种模型，结合了利用侧面信息的两种策略。也就是说，下面提出的模型是类条件的，但有一个辅助解码器，负责重建类标签。



### AC-GANs

我们提出了GAN架构的变体，我们称之为辅助分类器GAN（或AC-GAN）。在AC-GAN中，除了噪声z外，每个生成的样本都有一个相应的类label，c∼pc。G同时用于生成图像Xf ake = G(c, z)。鉴别器给出了源上的概率分布和类标签的概率分布，P(S | X), P(C | X) = D(X)。目标函数分为两部分：正确源LS的对数似然和正确类LC的对数似然。

LS表示是真实图像还是虚假图像，LC表示是哪个类别。

![image-20220318152505756](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220318152509.png)

D被训练以最大化LS + LC，而G被训练以最大化LC-LS。AC-GAN学习与类标签无关的z表示。

从结构上讲，这个模型与现有模型没有太大区别。然而，对stan-dard GAN配方的这种修改产生了出色的效果和ap-梨来稳定训练。此外，我们认为AC-GAN模型只是这项工作技术贡献的一部分，以及我们提出的测量模型利用其给定输出分辨率的程度的方法，测量模型样本感知变异性的方法，以及从所有1000个ImageNet类中创建128×128个样本的图像生成模型的彻底实验分析。

早期的实验表明，在固定模型的同时增加受训班级的数量会降低模型输出的质量。AC-GAN模型的结构允许按类将大型数据集分隔为子集，并为每个子集训练生成器和鉴别器。所有ImageNet实验都使用100个AC-GAN的合奏进行，每个组合都经过10个班级的分组训练。