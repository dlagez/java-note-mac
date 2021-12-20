## 使用残差网络与wgan制作二次元人物头像

ref：https://blog.csdn.net/qq_41103479/article/details/119352714



我复现的项目链接：https://github.com/dlagez/gan_resnet



参考的wgan链接：https://github.com/martinarjovsky/WassersteinGAN



主要有两个地方需要注意，一个使用了wgan的损失函数。

```python
class Wasserstein(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = -pred_real.mean()
            loss_fake = pred_fake.mean()
            loss = loss_real + loss_fake
            return loss
        else:
            loss = -pred_real.mean()
            return loss
```



一个是使用了残差网络作为gan的主体

```python
class BasicBlock(nn.Module):
    def __init__(self, in1):  # in1为输入的channel大小，BasicBlock输出等于输入channel大小
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in1, in1 * 2, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 =nn.BatchNorm2d(in1*2)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(in1*2, in1, kernel_size=3,
                        stride=1, padding=1, bias=False)
        self.bn2 =nn.BatchNorm2d(in1)
        self.relu2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
      # out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
      # out = self.bn2(out)
        out = self.relu2(out)

        out = out + residual
        return out

class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3 ,stride=1, padding=1)


        self.layer1 = nn.Sequential(
            BasicBlock(64),
            nn.AvgPool2d(3, 2),
            BasicBlock(64),
        )

        self.layer2 = nn.Sequential(
            nn.AvgPool2d(3,2),
            BasicBlock(64)
        )

        self.layer3 = nn.Sequential(
            nn.AvgPool2d(3, 2),
            BasicBlock(64),
        )

        self.layer4 = nn.Sequential(
            nn.AvgPool2d(3, 2),
            BasicBlock(64),
        )

        self.layer5 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
          nn.Linear(576, 1),
        )
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out
class netG(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(nz, 64*2*2)
        self.layer1 = nn.Sequential(
            BasicBlock(64),
            nn.UpsamplingNearest2d(scale_factor=2),
            BasicBlock(64),
            nn.UpsamplingNearest2d(scale_factor=2),

        )
        self.layer2 = nn.Sequential(
            BasicBlock(64),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(64),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(64),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
        self.Conv = nn.Sequential(
            BasicBlock(64),
            nn.BatchNorm2d(64),
          #  nn.LayerNorm([64,96,96]),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=1),
            nn.Tanh()
        )


    def forward(self, z):
        x = self.linear(z)
        x = x.view(batch_size,64,2,2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.Conv(x)
        return x
```



使用方法：

1.将我的项目clone到本地：

2.下载数据集到文件夹：data

3.执行下面的命令 --data_path 表示下载的数据集的位置

```
python /content/gan_resnet/train.py --epoch 20 --batchSize 8 --data_path /content/gan_resnet/data/ --outf /content/drive/MyDrive/data/gan_resnet/v5/
```

### 训练效果：

训练效果感觉比较差，尤其是当训练的次数多了之后，感觉生成的图片的质量变得差了一点。这里我还没有找到原因。



这是训练9个epochs的效果

![image-20211220092740260](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211220092740260.png)



这是训练20个epochs的效果

![image-20211220092835906](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211220092835906.png)

这是训练100个epochs的效果

![image-20211220092902576](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211220092902576.png)



感觉训练是有问题的，我先是在google colab里面运行了10个epochs。然后再转到实验室的3090里面运行了一晚上。但是效果还是很差。不知道是不是分了两次训练的问题。