### 博客笔记：

我们知道深度学习中对图像处理应用最好的模型是CNN，那么如何把CNN与GAN结合？DCGAN是这方面最好的尝试之一

它只是把上述的G和D换成了两个卷积神经网络（CNN）。但不是直接换就可以了，DCGAN对卷积神经网络的结构做了一些改变，以提高样本的质量和收敛的速度，这些改变有：

- 取消所有pooling层。G网络中使用转置卷积（transposed convolutional layer）进行上采样，D网络中用加入stride的卷积代替pooling。
- 在D和G中均使用batch normalization
- 去掉FC层，使网络变为全卷积网络
- G网络中使用ReLU作为激活函数，最后一层使用tanh
- D网络中使用LeakyReLU作为激活函数





### 论文原文详读：

### ABSTRACT

近年来，卷积网络（CNN）的监督学习在计算机视觉应用中得到了广泛采用。相比之下，CNN的无监督学习受到的关注较少。在这项工作中，我们希望帮助弥合CNN监督式学习和非超级学习成功之间的差距。我们引入了一类名为深度卷积生成对抗网络（DCGAN）的CNN，它们具有一定的架构约束，并证明它们是无监督学习的有力候选者。在各种图像数据集的培训中，我们展示了令人信服的证据，证明我们深层卷曲对抗对在生成器和鉴别器中学习了从对象部分到场景的表示层次结构。此外，我们将学到的功能用于新任务——证明它们作为一般图像表示的适用性。

### INTRODUCTION

从大型无标签数据集学习可重用特征表示一直是积极研究的一个领域。在计算机视觉方面，人们可以利用几乎无限量的无标签图像和视频来学习良好的中间表示，然后可用于图像分类等各种监督学习任务。我们建议，构建良好图像表示的一种方法是培训生成对抗网络（GAN）（Goodfellow等人，2014年），然后将生成器和鉴别器网络的部分内容重用为监督任务的特征提取器。GAN为最大可能性技术提供了一个有吸引力的替代方案。人们还可以争辩说，他们的学习过程和缺乏启发式成本函数（如像素独立的均方误差）对表示学习很有吸引力。众所周知，GAN在训练上不稳定，往往导致生成器产生无意义的输出。在试图理解和可视化GAN学到的东西以及多层GAN的中间表示方面，发表的研究非常有限。

In this paper, we make the following contributions

- 我们提出并评估了一套关于卷积GAN架构拓扑的约束，使其在大多数情况下都能进行训练。我们为这类架构命名深度卷积GAN（DCGAN）
- 我们将训练有素的鉴别器用于图像分类任务，显示出与其他无监督算法的竞争性能。
- 我们可视化了GAN学到的过滤器，并从经验上表明特定过滤器已经学会了绘制特定对象。
- 我们表明，发电机具有有趣的矢量算法特性，可以轻松操作生成样本的许多语义质量。



### RELATED WORK

#### 从无标签数据中学习表示

无监督表示学习是一般计算机视觉研究中研究得相当好的问题，以及图像的上下文。无监督表示学习的经典方法是对数据进行聚类 `for example using K-means`，并利用集群来提高分类分数。在图像上下文中，您可以对图像补丁进行分层聚类去学习强大的图像表示，另一种流行的方法是训练自动编码器分离代码的内容和位置组件，将图像编码为紧凑的代码，并解码代码以尽可能准确地重建图像，这些方法还被证明可以从图像像素中学习良好的特征表示，事实证明，在学习分层表示方面深度置信网络也很有效。



#### GENERATING NATURAL IMAGES

生成图像模型研究得很好，分为两类：参数和非参数。

非参数模型通常从现有图像数据库进行匹配，通常匹配图像补丁，并用于纹理合成，超分辨率。

生成图像的参数模型已被广泛探索（例如MNIST数字或纹理合成），然而，直到最近，生成现实世界的自然图像才取得多大成功，生成图像的变分采样方法（Kingma和Welling，2013年）取得了一些成功，但样本往往模糊不清，

另一种方法使用迭代正向扩散过程生成图像（Sohl-Dickstein等人，2015年）。生成对抗网络（Goodfellow等人，2014年）生成了噪音和难以理解的图像。这种方法的拉普拉西亚金字塔延伸（Denton等人，2015年）显示了更高质量的图像，但由于在链式多个模型中引入的噪音，它们仍然受到物体看起来摇晃的影响。循环网络方法（Gregor等人，2015年）和反卷积网络方法（Dosovitskiy等人，2014年）最近在生成自然图像方面也取得了一些成功。然而，他们没有利用发电机执行监督任务。



#### VISUALIZING THE INTERNALS OF CNNS

对使用神经网络的一个持续批评是，它们是黑盒方法，对网络以简单的人类消耗算法的形式做什么知之甚少。在CNN的背景下，Zeiler等人。（Zeiler & Fergus，2014年）表明，通过使用反卷积和过滤最大激活，可以找到网络中每个卷积过滤器的近似目的。同样，在输入上使用梯度下降可以让我们检查激活过滤器某些子集的理想图像（Mordvintsev等人）。



### APPROACH AND MODEL ARCHITECTURE !!

使用CNN来建模图像来扩展GAN的历史尝试没有成功。这促使LAPGAN（Denton等人，2015年）的作者开发了一种替代方法——可以更可靠地建模的高档低分辨率生成图像。我们还在尝试使用受监督文献中常用的CNN架构扩展GAN时遇到了困难。

然而，经过广泛的模型探索，我们确定了一系列架构，这些架构可以在一系列数据集上进行稳定的训练，并允许训练更高分辨率和更深入的生成模型。我们方法的核心是采用和修改最近展示的对 CNN 架构的三个更改。



第一个是全卷积网络（Springenberg et al., 2014），它用跨步卷积代替确定性空间池化函数（例如 maxpooling），允许网络学习自己的空间下采样。 我们在生成器中使用这种方法，使其能够学习自己的空间上采样和鉴别器。



其次是在卷积特征之上消除全连接层的趋势。 最有力的例子是全局平均池化，它已被用于最先进的图像分类模型（Mordvintsev 等人）。 我们发现全局平均池化提高了模型稳定性，但损害了收敛速度。 将最高卷积特征分别直接连接到生成器和鉴别器的输入和输出的中间地带效果很好。 GAN 的第一层以均匀的噪声分布 Z 作为输入，可以称为全连接，因为它只是一个矩阵乘法，但结果被重新整形为 4 维张量并用作卷积堆栈的开始 . 对于鉴别器，最后一个卷积层被展平，然后馈入单个 sigmoid 输出。 有关示例模型架构的可视化，请参见图 1。



第三是批量归一化（Ioffe & Szegedy，2015），它通过将每个单元的输入归一化以具有零均值和单元方差来稳定学习。 这有助于处理由于初始化不良而出现的训练问题，并有助于梯度在更深层次的模型中流动。 事实证明，这对于让深度生成器开始学习至关重要，防止生成器将所有样本崩溃到一个点，这是 GAN 中观察到的常见故障模式。 然而，直接将 batchnorm 应用于所有层会导致样本振荡和模型不稳定。 这通过不对生成器输出层和鉴别器输入层应用 batchnorm 来避免。



ReLU激活（Nair和Hinton，2010年）用于发电机，但使用Tanh函数的输出层除外。我们观察到，使用有界激活使模型能够更快地学习，以饱和和覆盖训练分布的颜色空间。在识别器中，我们发现泄漏的整流激活（Maas等人，2013年）（Xu等人，2015年）效果良好，特别是对于更高分辨率的建模。这与使用最大激活的原始GAN论文形成鲜明对比（Goodfellow等人，2013年）。



稳定深度卷积GAN的体系结构指南：

- 用大步卷积（鉴别器）和分数条纹卷积（发电机）取代任何池层。
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.



### DETAILS OF ADVERSARIAL TRAINING

我们在三个数据集上培训了DCGAN，即大规模场景理解（LSUN）（Yu等人，2015年）、Imagenet-1k和新组装的Faces数据集。有关每个数据集使用情况的详细信息如下。

除了缩放到tanh激活函数的范围[-1, 1]外，没有对训练图像进行预处理。所有模型都采用迷你批次随机梯度下降（SGD）进行了训练，迷你批次尺寸为128。所有权重都是从标准偏差0.02的零中心正态分布初始化的。在LeakyReLU中，所有型号的泄漏斜率都设置为0.2。虽然之前的GAN工作使用动量来加速训练，但我们使用带有调谐超参数的Adam优化器（Kingma和Ba，2014年）。我们发现建议的0.001学习率太高了，改用0.0002。另外，我们发现将动量项 β1 保留在建议值 0.9 会导致训练振荡和不稳定，而将其降低到 0.5 有助于稳定训练。

![image-20220313210503496](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220313210503496.png)

LSUN

随着生成图像模型样本视觉质量的提高，对训练样本过度拟合和记忆的担忧加剧。为了演示我们的模型如何通过更多数据和更高的分辨率生成进行扩展，我们在LSUN卧室数据集上训练了一个模型，其中包含300多万个训练示例。最近的分析表明，模型学习速度与其泛化性能之间存在直接联系（Hardt等人，2015年）。除了收敛后的样本（图3），我们还展示了一个训练时代的样本（图2），模仿在线学习，以证明我们的模型不是仅仅通过过度拟合/记忆训练示例来产生高质量的样本的机会。没有对图像应用数据增强。



DEDUPLICATION

为了进一步降低生成器记住输入示例的可能性（图2），我们执行了一个简单的图像重复数据删除过程。我们在32x32下采样中心作物培训示例上安装了3072-128-3072去噪辍学常规RELU自动编码器。然后，由此产生的代码层激活通过阈值ReLU激活进行二值化，ReLU激活已被证明是一种有效的信息保存技术（Srivastava等人，2014年），并提供一种方便的语义散列形式，允许线性时间重复数据删除。对散列碰撞的目视检查显示精度很高，估计假阳性率不到100分之一。此外，该技术检测并删除了大约275,000个接近重复项，这表明召回率很高。



FACES

我们从对人名的随机网络图像查询中刮取了包含人脸的图像。这些人的名字是从dbpedia获得的，标准是他们出生在现代。这个数据集有来自1万人的3M图像。我们对这些图像运行OpenCV人脸检测器，保持足够高分辨率的检测，这给了我们大约35万个人脸盒。我们使用这些面罩盒进行培训。没有对图像应用数据增强。



IMAGENET-1K

我们使用 Imagenet-1k (Deng et al., 2009) 作为无监督训练的自然图像来源。 我们训练 32 × 32 分钟调整大小的中心作物。 没有对图像应用数据增强。



### EMPIRICAL VALIDATION OF DCGANS CAPABILITIES

#### CLASSIFYING CIFAR-10 USING GANS AS A FEATURE EXTRACTOR

评估非监督表示学习算法rithms质量的一种常见技术是将它们用作受监督数据集上的特征提取器，并评估这些特征之上的线性模型的性能。

在CIFAR-10数据集上，使用K均值作为特征学习算法的调谐良好的单层特征提取管道证明了非常强大的基线性能。当使用大量特征地图（4800）时，该技术实现了80.6%的准确性。基本算法的非监督多层扩展达到82.0%的准确性（Coates & Ng，2011年）。为了评估DCGAN为监督任务学习的表示的质量，我们在Imagenet-1k上进行培训，然后使用鉴别器从所有层的卷积特征，最大限度地将每个层表示形式集合起来，以生成一个4×4的空间网格。然后将这些特征展平并连接起来，形成一个28672维矢量，并在它们上面训练一个正则化的线性L2-SVM分类器。这实现了82.8%的准确性，超过了所有基于K均值的方法。值得注意的是，与基于K均值的技术相比，鉴别器的特征映射要少得多（最高层为512），但由于4×4空间位置的许多层，特征矢量大小确实更大。DCGANs的性能仍然低于Exemplar CNNs（Dosovitskiy等人，2015年），这是一种以不受监督的方式训练普通歧视性CNN的技术，以区分源数据集中专门选择的、积极增强的示例样本。通过微调歧视者的陈述，可以做出进一步的改进，但我们将此留待未来工作。此外，由于我们的DCGAN从未接受过CIFAR-10的培训，该实验还展示了所学功能的域鲁棒性。



#### CLASSIFYING SVHN DIGITS USING GANS AS A FEATURE EXTRACTOR

在StreetView House Numbers数据集（SVHN）（Netzer等人，2011年）上，当标记数据稀缺时，我们将DCGAN鉴别器的功能用于监督目的。按照与CIFAR-10实验类似的数据集准备规则，我们从非额外集中拆分了10,000个示例的验证集，并将其用于所有超参数和模型选择。随机选择1000个统一类分布式训练示例，并用于在用于CIFAR-10的相同特征提取管道上训练正则化线性L2-SVM分类器。这达到了22.48%的测试误差（使用1000个标签进行分类），改进了旨在利用未标签数据的CNN的另一项修改（Zhao等人，2015年）。此外，我们验证DCGAN中使用的CNN架构不是模型性能的关键因素，方法是在相同的数据上训练具有相同架构的纯监督CNN，并通过64次超参数试验的随机搜索优化该模型（Bergstra和Bengio，2012年）。它实现了明显更高的28.87%的验证错误。



INVESTIGATING AND VISUALIZING THE INTERNALS OF THE NETWORKS

我们以多种方式研究训练有素的生成器和判别器。 我们不对训练集进行任何类型的最近邻搜索。 像素或特征空间中的最近邻被小图像变换简单地愚弄（Theis et al., 2015）。 我们也没有使用对数似然指标来定量评估模型，因为它是一个很差的指标（Theis et al., 2015）。



WALKING IN THE LATENT SPACE

我们做的第一个实验是了解潜在空间的景观。 走在所学的流形上通常可以告诉我们记忆的迹象（如果有急剧的转变）以及空间分层折叠的方式。 如果在这个潜在空间中行走会导致图像生成的语义发生变化（例如添加和删除对象），我们可以推断该模型已经学习了相关且有趣的表示。 结果如图4所示。



VISUALIZING THE DISCRIMINATOR FEATURES

之前的工作表明，对CNN进行大型图像数据集的监督培训会产生非常强大的学习功能（Zeiler和Fergus，2014年）。此外，接受现场分类培训的受监督CNN学习物体探测器（Oquab等人，2014年）。我们证明，在大型图像数据集上训练的无监督DCGAN也可以学习有趣的功能层次结构。使用（Springenberg等人，2014年）提议的引导式反向传播，我们在图5中显示，鉴别器学到的功能在卧室的典型部分激活，如床和窗户。为了进行比较，在同一图中，我们给出了随机初始化功能的基线，这些功能没有在语义上相关或有趣的任何内容上激活。



MANIPULATING THE GENERATOR REPRESENTATION

FORGETTING TO DRAW CERTAIN OBJECTS

除了鉴别者学到的表示外，还有一个生成器学到什么表示的问题。样本的质量表明，发电机学习了床、窗户、灯具、门和杂项家具等主要场景组件的特定对象表示。为了探索这些表示形式，我们进行了一项实验，试图从生成器中完全删除窗口。

在150个样本中，手工绘制了52个窗口边界框。在第二高的卷积层特征上，逻辑回归适合预测特征激活是否在窗口上，方法是使用绘制的边界框内的激活是正面的，同一图像中的随机样本是负数的标准。使用这个简单的模型，所有权重大于零（共200）的特征地图都从所有空间位置删除。然后，随机生成新样本，无论是否删除特征图。

图6显示了带或不带窗口掉线的生成图像，有趣的是，网络大多忘记在卧室里画窗户，用其他对象替换它们。



VECTOR ARITHMETIC ON FACE SAMPLES

在评估学习的单词表示（Mikolov等人，2013年）的背景下，表明简单的算术运算在表示空间中显示了丰富的线性结构。一个canoni-cal示例表明，矢量（“国王”）-矢量（“男人”）+矢量（“女人”）产生了一个矢量，其最近的邻居是女王的矢量。我们调查了发电机的Z表示中是否出现了类似的结构。我们对视觉概念示例样本集的Z矢量进行了类似的算法。每个概念只对单个样本进行的实验不稳定，但三个样本的平均Z矢量显示，在语义上遵守算法的几代一致且稳定。除了如图7所示的对象操作外，我们还演示了面部姿势也在Z空间线性建模（图8）。

这些演示表明，可以使用我们模型学到的Z表示来开发有趣的应用程序。此前已经证明，条件生成模型可以学习令人信服地建模标度、旋转和位置等对象属性（Dosovitskiy等人，2014年）。据我们所知，这是在纯粹无人监督模型的情况下发生的第一次证明，进一步探索和发展上述矢量算法可以大幅减少复杂图像分布的条件生成建模所需的数据量。



CONCLUSION AND FUTURE WORK

我们提出了一套更稳定的体系结构来训练生成对抗网络，并证明对抗网络学习了良好的图像表示，用于监督学习和生成建模。仍然存在一些形式的模型不稳定性——我们注意到，随着模型训练时间更长，它们有时会将滤波器子集折叠到单个振荡模式。

需要进一步的工作来解决这个不稳定的问题。 我们认为将这个框架扩展到其他领域，例如视频（用于帧预测）和音频（用于语音合成的预训练特征）应该非常有趣。 对学习到的潜在空间属性的进一步研究也会很有趣。



code：

```python
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

```

 其中需要注意的是：

网络结构：

```txt
Generator(
  (l1): Sequential(
    (0): Linear(in_features=100, out_features=8192, bias=True)
  )
  (conv_blocks): Sequential(
    (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Upsample(scale_factor=2.0, mode=nearest)
    (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(128, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Upsample(scale_factor=2.0, mode=nearest)
    (6): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
    (8): LeakyReLU(negative_slope=0.2, inplace=True)
    (9): Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (10): Tanh()
  )
)

Discriminator(
  (model): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Dropout2d(p=0.25, inplace=False)
    (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Dropout2d(p=0.25, inplace=False)
    (6): BatchNorm2d(32, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
    (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (8): LeakyReLU(negative_slope=0.2, inplace=True)
    (9): Dropout2d(p=0.25, inplace=False)
    (10): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
    (11): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (12): LeakyReLU(negative_slope=0.2, inplace=True)
    (13): Dropout2d(p=0.25, inplace=False)
    (14): BatchNorm2d(128, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
  )
  (adv_layer): Sequential(
    (0): Linear(in_features=512, out_features=1, bias=True)
    (1): Sigmoid()
  )
)

```

