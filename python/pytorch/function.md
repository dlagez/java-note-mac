官网链接：https://pytorch.org



### [torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d) (Python class, in Conv2d)

Applies a 2D convolution over an input signal composed of several input planes.

- **in_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of channels in the input image
- **out_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of channels produced by the convolution
- **kernel_size** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)) – Size of the convolving kernel
- **stride** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)*,* *optional*) – Stride of the convolution. Default: 1
- **padding** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)*,* *optional*) – Padding added to all four sides of the input. 

#### example:

```python
 self.conv1 = nn.Conv2d(in1, in1 * 2, kernel_size=1, stride=1, padding=0, bias=False)
```

#### My understanding is:

输入的层数是in1，输出的层数是in1 * 2，所以上面这句代码的作用是将输入的图片层数增加一倍。





### [torch.nn.BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html?highlight=batchnorm2d#torch.nn.BatchNorm2d) (Python class, in BatchNorm2d)

Applies Batch Normalization over a 4D input

- num_features – C from an expected input of size (N,C,H,W)

example:

```python
self.bn1 =nn.BatchNorm2d(in1*2)
```

#### My understanding is:

在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理。这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。输入和输出的形状是一样的。

- num_features：一般输入参数为batch_size * num_features * height * width，即为其中特征的数量





### [torch.nn.LeakyReLU](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html?highlight=leakyrelu#torch.nn.LeakyReLU) (Python class, in LeakyReLU)

### 作用：

信号从一个神经元进入，经过非线性的激活函数，传入到下一个神经元。正是由于这些非线性函数的反复叠加， 才使得神经网络有足够的capacity来抓取复杂的pattern。也就是激活函数将神经网络由线性变成非线形的。

LeakyReLU函数长的像下面这个样子。

<img src="https://pytorch.org/docs/stable/_images/LeakyReLU.png" alt="../_images/LeakyReLU.png" style="zoom: 67%;" />



### [torch.Tensor.detach](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html?highlight=detach#torch.Tensor.detach) (Python method, in torch.Tensor.detach)

返回一个新的tensor，新的tensor和原来的tensor共享数据内存，但不涉及梯度计算，即requires_grad=False。修改其中一个tensor的值，另一个也会改变，因为是共享同一块内存，但如果对其中一个tensor执行某些内置操作





