Generative Adversarial Nets

### Abstract

我们提出了一个通过对抗过程估计生成模型的新框架，其中我们同时训练两个模型：一个是捕获数据分布的生成模型G，另一个是估计样本来自训练数据而不是G的概率的判别模型D。G的训练程序是最大限度地提高D出错的可能性。这个框架对应于minimax双人游戏。在任意函数G和D的空间中，存在一个独特的解决方案，G恢复训练数据分布，D在任何地方都等于21。在G和D由多层感知器定义的情况下，整个系统可以通过反向传播进行训练。在培训或生成样本期间，不需要任何马尔科夫链或展开的近似推理网工作。实验通过对生成的样本进行定性和定量评估，证明了该框架的潜力。

### Introduction

深度学习的前景是发现丰富的分层模型[2]，这些模型代表了人工智能应用程序中遇到的数据类型的概率分布，如自然图像、包含语音的音频波形和自然语言语料库中的符号。到目前为止，深度学习中最显著的成功涉及鉴别模型，通常是那些将高维度、丰富的感官输入映射到类标签的模型[14、22]。这些显著的成功主要基于反向传播和辍学算法，使用具有特别良好梯度的分段线性单位[19、9、10]。由于难以近似在最大似然估计和相关策略中产生的许多棘手概率计算，以及难以在生成上下文中利用分段线性单元的好处，深度生成模型的影响较小。我们提出了一种新的生成模型估计程序来绕过这些困难。1

在拟议的对抗网框架中，生成模型与对手对立：一种学习确定样本是来自模型分布还是数据分布的鉴别模型。生成模型可以被视为类似于一组伪造者，试图生产假币并在不被发现的情况下使用它，而歧视模型类似于警察，试图检测假币。本场比赛的竞争促使两支球队改进方法，直到假货与正版物品无关。



该框架可以为多种模型和优化算法提供特定的训练算法。本文探讨了生成模型通过多层感知器传递随机噪声生成样本的特殊情况，判别模型也是一个多层感知器。我们把这个特殊情况称为对抗网。在这种情况下，我们只能使用非常成功的反向传播和退出算法[17]来训练这两个模型，并且只能使用正向传播从生成模型中采样。不需要近似推断或马尔可夫链。

### Related work

具有潜在变量的定向图形模型的替代方案是具有潜在变量的无向图形模型，如受限博尔兹曼机器（RBMs）[27、16]、深度博尔兹曼机器（DBM）[26]及其众多变体。这些模型中的相互作用表示为非归一化势函数的乘积，由随机变量所有状态的全局求和/积分归一化。这个数量（分区函数）及其梯度对除最琐碎的实例外的所有实例都是棘手的，尽管它们可以通过马尔科夫链蒙特卡洛（MCMC）方法进行估计。混合对依赖MC[3,5]的学习算法构成了重大问题。

深度信仰网络（DBN）[16]是包含单个无向层和sev-eral定向层的混合模型。虽然存在快速近似分层训练标准，但DBN会遇到与无向和有向模型相关的计算困难。

还提出了不近似或约束对数似然的替代标准，如分数匹配[18]和噪声对比估计（NCE）[13]。两者都要求从解析上指定学到的概率密度，直到归一化常数。请注意，在许多具有几层潜在变量（如DBN和DBBM）的有趣生成模型中，甚至不可能导出可处理的非归一化概率密度。一些模型，如去噪自动编码器[30]和收缩自动编码器，其学习规则与适用于RBM的分数匹配非常相似。在NCE中，与这项工作一样，使用判别训练标准来拟合生成模型。然而，生成模型本身不是适合单独的判别模型，而是用于区分生成的数据和固定噪声分布的样本。由于NCE使用固定的噪声分布，因此在模型在观测变量的一小部分上学习甚至大致正确的分布后，学习速度会大幅放缓。

最后，一些技术不涉及明确定义概率分布，而是训练生成机器从所需的分布中提取样本。这种方法的优点是，这些机器可以通过反向传播进行训练。该领域最近的突出工作包括生成随机网络（GSN）框架[5]，该框架扩展了广义去噪自动编码器[4]：两者都可以被视为定义参数化的马尔可夫链，即学习执行生成马尔可夫链一步的机器的参数。与GSN相比，对抗网框架不需要马尔可夫链进行采样。由于对抗网在生成过程中不需要反馈回路，它们能够更好地利用分段线性单元[19、9、10]，这提高了反向传播的性能，但在反馈循环中使用时存在无界激活问题。通过反向传播训练生成机器的最新示例包括最近在自动编码变分贝叶斯[20]和随机反向传播[24]方面的工作。

### Adversarial nets



### Advantages and disadvantages

与之前的建模框架相比，这个新框架具有优缺点。缺点主要是pg（x）没有显式表示，并且D在训练期间必须与G很好地同步（特别是，在不更新D的情况下，G不得进行太多训练，以避免G折叠太多z值到x的相同值的“Helvetica场景”，以至于有足够的多样性来建模pdata），就像Boltzmann机器的负链必须在学习步骤之间保持最新一样。优点是从来不需要马尔科夫链，只使用反向道具来获得梯度，学习期间不需要推断，模型中可以将各种各样的功能纳入模型。表2总结了生成对抗网与其他生成建模方法的比较。



上述优势主要是计算性的。对抗模型也可能从发电机网络中获得一些统计优势，因为发电机网络没有直接更新数据检查，而只是随着梯度流过鉴别器。这意味着输入的组件不会直接复制到生成器的参数中。对抗网络作品的另一个优势是，它们可以代表非常锐利甚至退化的分布，而基于马尔科夫链的方法要求分布有点模糊，以便链能够在模式之间混合。

### Conclusions and future work

这个框架允许许多直接的扩展：

- 通过在G和D中同时添加c作为输入，可以获得条件生成模型p(x/c)。
- 学习的近似推理可以通过训练辅助网络来预测给定x的z来执行。这类似于唤醒睡眠算法[15]训练的推理网，但优点是推理网可以在发电机之后为固定发电机网训练Net已经完成训练。
-  通过训练一组共享参数的条件模型，可以大致建模所有条件p(xS | x ⁇ S)，其中S是x指数的子集。本质上，人们可以使用对抗网来实现确定性MP-DBM[11]的随机扩展。
- 半监督学习：当标签数据有限时，来自鉴别器或推理网的功能可以改善分类器的性能。
- 提高效率：通过划分更好的协调G和D的方法或确定培训期间对z样本的更好分布，可以大大加快培训速度。



### code:

```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
```



`nn.Linear(in_feat, out_feat)`

Applies a linear transformation to the incoming data: y=xAT+b*y*=*x**A**T*+*b*

这个就是一个线性函数。





```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

```



网络结构：G

```
Namespace(b1=0.5, b2=0.999, batch_size=64, channels=1, img_size=28, latent_dim=100, lr=0.0002, n_cpu=8, n_epochs=200, sample_interval=400)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Sequential: 1-1                        --
|    └─Linear: 2-1                       12,928
|    └─LeakyReLU: 2-2                    --
|    └─Linear: 2-3                       33,024
|    └─BatchNorm1d: 2-4                  512
|    └─LeakyReLU: 2-5                    --
|    └─Linear: 2-6                       131,584
|    └─BatchNorm1d: 2-7                  1,024
|    └─LeakyReLU: 2-8                    --
|    └─Linear: 2-9                       525,312
|    └─BatchNorm1d: 2-10                 2,048
|    └─LeakyReLU: 2-11                   --
|    └─Linear: 2-12                      803,600
|    └─Tanh: 2-13                        --
=================================================================
Total params: 1,510,032
Trainable params: 1,510,032
Non-trainable params: 0
=================================================================
```



D:

```
Namespace(b1=0.5, b2=0.999, batch_size=64, channels=1, img_size=28, latent_dim=100, lr=0.0002, n_cpu=8, n_epochs=200, sample_interval=400)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Sequential: 1-1                        --
|    └─Linear: 2-1                       401,920
|    └─LeakyReLU: 2-2                    --
|    └─Linear: 2-3                       131,328
|    └─LeakyReLU: 2-4                    --
|    └─Linear: 2-5                       257
|    └─Sigmoid: 2-6                      --
=================================================================
Total params: 533,505
Trainable params: 533,505
Non-trainable params: 0
=================================================================
```



在epoch中：gen使用了随机噪声来生成图片，使用了这个参数：`opt.latent_dim`，和图像的长一起作为生成随机参数的参数。

```python
# Sample noise as generator inpu`t
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

# Generate a batch of images
gen_imgs = generator(z)
```



训练过程：

```python
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # 将真实的图片放入device中
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------
				# 将优化器的梯度置零
        optimizer_G.zero_grad()

        # z为64*100的形状
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # 生成一个批次的图片 64*1*28*28 
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        # 这里首先使用D网络测试G网络生成的图片。也就是说G网络的参数更新是根据D网络的性能来进行更新的
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
				
        # 反向传播。优化网络的节点。
        g_loss.backward()
        # 
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # 首先需要置零梯度
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        # D网络首先检测真实图片，再检测生成图片。利用两个损失来更新D网络的参数。
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
            # 保存前25张图片
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
```

