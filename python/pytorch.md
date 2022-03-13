## plt

### 图相关

一个背景上画两个图。

```python
fig, axs = plt.subplots(2, 1)  # 参数为行列数，这里是两行一列。
```

设置标题

```
ax.set_title('Score by group and gender')
```

设置右上角的图例

```python
ax.legend()  # Place a legend on the Axes.
```

显示网格：

```
axs[0].grid(True)
```

调整子图之间和周围的填充。

```python
fig.tight_layout()
```

### Y轴相关的设置

设置y轴名：

```python
fig, ax = plt.subplots()
ax.set_ylabel('Scores')set_ylabel
```

### X轴相关的设置

设置x轴的标签

```python
ax.set_xticks(x)  # Set the xaxis' tick locations.
ax.set_xticklabels(labels)  # Set the xaxis' labels with list of string labels.
```

设置x轴试图限制

```python
axs[0].set_xlim(0, 2)
```

设置x轴的名字

```python
axs[0].set_xlabel('time')
```

设置柱形图上的标记 padding=3表示叫数字抬高一点，以免数字和柱形图重叠了。

```python
ax.bar_label(rects1, padding=3)  # 柱形图上面的数字
ax.bar_label(rects2, padding=3)
```



### example

画柱形图：[doc](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html#matplotlib.axes.Axes.bar)

```python
import matplotlib.pyplot as plt
import numpy as np

labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels)) # 标签的位置
width = 0.35 # 柱的宽度

fig, ax = plt.subplots()
# x - width/2: 柱坐标开始的地方, width: 柱的宽度, 
rects1 = ax.bar(x - width/2, men_means, width, label='Men')
rects2 = ax.bar(x + width/2, women_means, width, label='Women')

ax.set_ylabel('Scores')
ax.set_title('Score by group and gender')
ax.set_xticks(x)  # 用来显示x的标签
ax.set_xticklabels(labels)
ax.legend()  # 和上面的rects的labels对应。用来显示labels。


ax.bar_label(rects1, padding=3)  # 柱形图上面的数字
ax.bar_label(rects2, padding=3)


fig.tight_layout()
plt.show()
```

## pytorch

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

 

### pytorch method

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

- **Non-Linearity –** Activation function should be able to add nonlinearity in neural networks especially in the neurons of hidden layers. This is because rarely you will see any real-world scenarios that can be explained with linear relationships.
- **Differentiable** – The activation function should be differentiable. During the training phase, the neural network learns by back-propagating error from the output layer to hidden layers. The backpropagation algorithm uses the derivative of the activation function of neurons in hidden layers, to adjust their weights so that error in the next training epoch can be reduced.

#### ReLU

The **ReLU or Rectified Linear Activation Function** is a type of piecewise linear function.

**Advantages of ReLU Activation Function**

- ReLu activation function is computationally fast hence it enables faster convergence of the training phase of the neural networks.
- It is both non-linear and differentiable which are good characteristics for activation function.
- ReLU does not suffer from the issue of Vanishing Gradient issue like other activation functions. Hence it is a good choice in hidden layers of large neural networks.

**Disadvantages** **of ReLU Activation Function**

- The main disadvantage of the ReLU function is that it can cause the problem of **Dying Neurons**. Whenever the inputs are negative, its derivative becomes zero, therefore backpropagation cannot be performed and learning may not take place for that neuron and it dies out.



```
m = nn.ReLU()
input = torch.randn(5)
output = m(input)

print("This is the input:",input)
print("This is the output:",output)
```

Output:

```
This is the input: tensor([ 1.0720, -1.4033, -0.6637,  1.2851, -0.5382])
This is the output: tensor([1.0720, 0.0000, 0.0000, 1.2851, 0.0000])
```



#### view function

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



#### Upsample [link](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html#upsample) 

Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

Parameters：

- **scale_factor** ([*float*](https://docs.python.org/3/library/functions.html#float) *or* *Tuple**[*[*float*](https://docs.python.org/3/library/functions.html#float)*] or* *Tuple**[*[*float*](https://docs.python.org/3/library/functions.html#float)*,* [*float*](https://docs.python.org/3/library/functions.html#float)*] or* *Tuple**[*[*float*](https://docs.python.org/3/library/functions.html#float)*,* [*float*](https://docs.python.org/3/library/functions.html#float)*,* [*float*](https://docs.python.org/3/library/functions.html#float)*]**,* *optional*) – multiplier for spatial size. Has to match input size if it is a tuple. 简单来说就是扩大多少倍，长宽都乘以这个倍数。