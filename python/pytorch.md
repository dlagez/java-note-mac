

### 网络训练流程：

S下载数据集操作：注意`root`的路径，放到`data`文件夹下面。

```python
training_data = datasets.FashionMNIST(
    root='/Volumes/roczhang/data/',
    train=True,
    download=True,
    transform=ToTensor(),
)
```

定义和使用dataloader

```python
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape if y: ", y.shape, y.dtype)
    break
```

使用GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```

定义网络：

```python
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

定义损失函数和优化方法

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 这应该是10的-3次方
```

定义训练流程：

```python
def train(dataloader, model, loss_fn, oprimizer):
    size = len(dataloader.dataset)
    model.train()  # 将模型设置为训练模式
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # computer prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        oprimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 优化参数

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
```

定义测试流程：

```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)  # 使用模型预测
            test_loss += loss_fn(pred, y).item()  # 将真实值和预测值进行对比计算出损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

定义epochs并训练

```python
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n -------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

保存模型

```python
torch.save(model.state_dict(), "/Volumes/roczhang/data/model/model.pth")
```



加载模型并使用模型去预测

```python
# define a new network
mode_new = NeuralNetwork()
# load pth file
mode_new.load_state_dict(torch.load("/Volumes/roczhang/data/model/model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

mode_new.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```



图像经过网络的细节：

```python
# 生成一张随机的图片输入到网络中
X = torch.rand(1, 28, 28, device=device)
logits = model(X)  # 丢进网络进行预测
pred_probab = nn.Softmax(dim=1)(logits)  # 将结果转化成0-1范围
y_pred = pred_probab.argmax()  
print(f"Predicted class: {y_pred}")
```

定义三张图片，送到一个Sequential里面，首先将每张图片变成一维的数据。在经过全链接层in_feature由784变成20。20变成10.然后输出。

```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
seq_modules = nn.Sequential(
    flatten,  # 会将3*28*28的图像变成3*784 ，相当于拉直了图片
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```

使用dataset显示图片

```python
training_data = datasets.FashionMNIST(
    root="/Volumes/roczhang/data/",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="/Volumes/roczhang/data/",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```





### 查看网络结构：

```python
import torchvision.models as models
from torchsummary import summary
import torch
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.vgg19().to(device)
 
summary(vgg, (3, 224, 224))

```

 

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
from torchsummary import summary

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


G = Generator().to(device)
D = Discriminator().to(device)
summary(G)
```



### 关于变量





### pytorch method

#### 序列解包

`*block(opt.latent_dim, 128, normalize=False)`

block前面带一个星号表示，block个方法，需要将方法执行完成后返回值用到`nn.Sequential`函数里面。*表示序列解包。

#### nn.Linear [link](http://www.sharetechnote.com/html/Python_PyTorch_nn_Linear_01.html)

**Inputs and 1 output (1 neuron)**

net = torch.nn.Linear(2,1);

This creates a network as shown below. Weight and Bias is set automatically.

- ![img](https://cdn.jsdelivr.net/gh/dlagez/img@master/Python_Pytorch_nn_Linear_i2_o1_01.png)

**Inputs and 2 outputs (2 neuron)

net = torch.nn.Linear(2,2);

This creates a network as shown below. Weight and Bias is set automatically.

- ![img](https://cdn.jsdelivr.net/gh/dlagez/img@master/Python_Pytorch_nn_Linear_i2_o2_01.png)



**2 Inputs and 3 output (3 neuron)**

net = torch.nn.Linear(2,3);

This creates a network as shown below. Weight and Bias is set automatically.

- ![img](https://cdn.jsdelivr.net/gh/dlagez/img@master/Python_Pytorch_nn_Linear_i2_o3_01.png)



#### nn.Sequential [link](http://www.sharetechnote.com/html/Python_PyTorch_nn_Sequential_01.html)

**2 Inputs , 1 outputs and Activation Function**

```
net = torch.nn.Sequential(
                         torch.nn.Linear(2,1),
                         torch.nn.Sigmoid()
                         );
```

- ![img](https://cdn.jsdelivr.net/gh/dlagez/img@master/Python_Pytorch_nn_Sequential_i2_o1_sigmoid_01.png)



#### Activation Function

[link](https://machinelearningknowledge.ai/pytorch-activation-functions-relu-leaky-relu-sigmoid-tanh-and-softmax/)

看图即可，激活函数为f，然后f作用于整个输入参数。

![Untitled6](https://cdn.jsdelivr.net/gh/dlagez/img@master/Untitled6.png)

线性函数应该具有的性质：

- **Non-Linearity –** 激活函数应该能够在神经网络中添加非线性，尤其是在隐藏层的神经元中。 这是因为您很少会看到任何可以用线性关系解释的真实场景。
- **Differentiable** – 激活函数应该是可微的。 在训练阶段，神经网络通过将误差从输出层反向传播到隐藏层来学习。 反向传播算法利用隐藏层神经元激活函数的导数来调整它们的权重，从而减少下一个训练时期的误差。



#### Sigmoid [link](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)

Sigmoid函数是一个在生物学中常见的S型函数，也称为S型生长曲线。 [1] 在信息科学中，由于其单增以及反函数单增等性质，Sigmoid函数常被用作神经网络的阈值函数，将变量映射到0,1之间。

#### ReLU

是一种分段线性函数。

- ReLu 激活函数的计算速度很快，因此它能够更快地收敛神经网络的训练阶段。
- 非线性和可微分都是激活函数的良好特性。
- ReLU 不会像其他激活函数那样遭受梯度消失问题的困扰。 因此，在大型神经网络的隐藏层中它是一个不错的选择。

ReLU激活函数的缺点

- ReLU函数的主要缺点是它会导致死亡节点的问题。 每当输入为负时，它的导数变为零，因此无法执行反向传播，并且该神经元可能不会进行学习并且它会消失。

#### view

他只是改变了tensor的形状，并没有在内存中copy数据。这里的b和t共享底层数组。他们只是数据的两种不同展现形式。

Typically a PyTorch op returns a new tensor as output, e.g. [`add()`](https://pytorch.org/docs/stable/generated/torch.Tensor.add.html#torch.Tensor.add). But in case of view ops, outputs are views of input tensors to avoid unncessary data copy

```
>>> t = torch.rand(4, 4)
>>> b = t.view(2, 8)
>>> t.storage().data_ptr() == b.storage().data_ptr()  # `t` and `b` share the same underlying data.
True
# Modifying view tensor changes base tensor as well.
>>> b[0][0] = 3.14
>>> t[0][0]
tensor(3.14)
```

#### np.random.normal

从正太高斯分布中随机抽取样本

```
np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))
```

第一个参数表示：中心的位置

第二个参数表示：宽度

第三个参数表示：样本的形状 （行，列）



#### BatchNorm2d  [link](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)

作用：使我们的一批feature map满足均值为0，方差为1的分布规律。

参数：

- num_features：一般输入参数为`batch_size*num_features*height*width`，即为其中特征的数量，即为输入BN层的通道数。
- eps：分母中添加的一个值，目的是为了计算的稳定性。
- momentum：一个用于运行过程中均值和方差的一个估计参数
- affine：当设为true时，会给定可以学习的系数矩阵gamma和beta

机器学习中，进行模型训练之前，需对数据做归一化处理，使其分布一致。在深度神经网络训练过程中，通常一次训练是一个batch，而非全体数据。每个batch具有不同的分布产生了internal covarivate shift问题——在训练过程中，数据分布会发生变化，对下一层网络的学习带来困难。Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，一方面使得数据分布一致，另一方面避免梯度消失。



example:

BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

表示输入的通道数量为128层。即一个输入例子的形状为`128*hight*width`

example

```python
import torch
from torch import nn

torch.manual_seed(123)

a = torch.rand(3,2,3,3)
print(a)

print(nn.BatchNorm2d(2)(a))


```

```
tensor([[[[0.2961, 0.5166, 0.2517],
          [0.6886, 0.0740, 0.8665],
          [0.1366, 0.1025, 0.1841]],

         [[0.7264, 0.3153, 0.6871],
          [0.0756, 0.1966, 0.3164],
          [0.4017, 0.1186, 0.8274]]],


        [[[0.3821, 0.6605, 0.8536],
          [0.5932, 0.6367, 0.9826],
          [0.2745, 0.6584, 0.2775]],

         [[0.8573, 0.8993, 0.0390],
          [0.9268, 0.7388, 0.7179],
          [0.7058, 0.9156, 0.4340]]],


        [[[0.0772, 0.3565, 0.1479],
          [0.5331, 0.4066, 0.2318],
          [0.4545, 0.9737, 0.4606]],

         [[0.5159, 0.4220, 0.5786],
          [0.9455, 0.8057, 0.6775],
          [0.6087, 0.6179, 0.6932]]]])
tensor([[[[-0.5621,  0.2574, -0.7273],
          [ 0.8968, -1.3879,  1.5584],
          [-1.1552, -1.2819, -0.9787]],

         [[ 0.5369, -1.0117,  0.3888],
          [-1.9141, -1.4584, -1.0073],
          [-0.6859, -1.7524,  0.9171]]],


        [[[-0.2425,  0.7925,  1.5103],
          [ 0.5422,  0.7042,  1.9901],
          [-0.6425,  0.7846, -0.6311]],

         [[ 1.0298,  1.1880, -2.0520],
          [ 1.2915,  0.5833,  0.5047],
          [ 0.4593,  1.2495, -0.5645]]],


        [[[-1.3761, -0.3375, -1.1132],
          [ 0.3187, -0.1512, -0.8011],
          [ 0.0269,  1.9569,  0.0493]],

         [[-0.2561, -0.6096, -0.0199],
          [ 1.3619,  0.8356,  0.3525],
          [ 0.0933,  0.1281,  0.4116]]]], grad_fn=<NativeBatchNormBackward>)
```



#### Upsample [link](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html#upsample) 

Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

Parameters：

- **scale_factor** ([*float*](https://docs.python.org/3/library/functions.html#float) *or* *Tuple**[*[*float*](https://docs.python.org/3/library/functions.html#float)*] or* *Tuple**[*[*float*](https://docs.python.org/3/library/functions.html#float)*,* [*float*](https://docs.python.org/3/library/functions.html#float)*] or* *Tuple**[*[*float*](https://docs.python.org/3/library/functions.html#float)*,* [*float*](https://docs.python.org/3/library/functions.html#float)*,* [*float*](https://docs.python.org/3/library/functions.html#float)*]**,* *optional*) – multiplier for spatial size. Has to match input size if it is a tuple. 简单来说就是扩大多少倍，长宽都乘以这个倍数。

#### Conv2d [link](https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)

对由多个输入平面组成的输入信号进行二维卷积

- kernel_size ：卷积核大小
- stride：卷积的跨度
- padding：填充物已添加到输入的所有四边

example

```
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
```





#### Variable

[torch](https://so.csdn.net/so/search?q=torch&spm=1001.2101.3001.7020).autograd.Variable是Autograd的核心类，**它封装了Tensor**，并整合了**反向传播**的相关实现([tensor](https://so.csdn.net/so/search?q=tensor&spm=1001.2101.3001.7020)变成variable之后才能进行反向传播求梯度?用变量.backward()进行反向传播之后,var.grad中保存了var的梯度)

```python
x = Variable(tensor, requires_grad = True)
```

Varibale包含三个属性：

- data：存储了Tensor，是本体的数据
- grad：保存了data的梯度，本事是个Variable而非Tensor，与data形状一致
- grad_fn：指向Function对象，用于反向传播的梯度计算之用



```python
import torch
from torch.autograd import Variable

x = Variable(torch.one(2,2), requires_grad = True)
print(x)#其实查询的是x.data,是个tensor
```





#### Tensor