## 训练过程详解：

项目地址：https://github.com/dlagez/gan_resnet

我们进入到代码中解析

首先这个网络由三部分文件组成。一个train，一个model，一个loss。

整体的文件布局：

![image-20211220124233928](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211220124233928.png)

### 设置debug

使用debug来详细查看网络运行的情况。如下设置即可进行debug

![image-20211220124441758](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211220124441758.png)

### 打断点

只需要在这几个地方打上断点即可查看网络运行的流程：

![image-20211220124640223](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211220124640223.png)

### 一批次的图像

前面的参数调式的时候查看一下就行，这里有几个重要的参数需要记录一下。

```
imgs = (8, 3, 64, 64)
```

把这一批次的图像送进D网络，D网络会输出一个二维的数组。

![image-20211220125518742](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211220125518742.png)



### D网络

表示一个批次有8张64*64的RGB图像。这八张图像经过D网络之后就变成了8行一列的二维数组了。里面具体的流程是怎么样的看下一篇笔记。



![image-20211220125518742](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211220125820290.png)



### G网络

```python
noise = torch.randn(opt.batchSize, opt.nz)
noise = noise.to(device)
fake = netG(noise)  # 生成假图
```

这里是使用了`batchSize`生成了一个批量的噪声。一个`nz`（噪声）如下图所示，是一个一维、含有100个点的噪声，我们生成八个噪声丢进`G`网络，`G`网络就会生成对应的八张图片（fake），每张图片的大小为`3*64*64`。



![image-20211220134037973](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211220134037973.png)



### 更新参数

```python
lossD = criterion(outputreal, outputfake)
lossD.backward()
optimizerD.step()
```

`fake`与真实图片丢进D网络得到的数组`outputfake,outputreal`。损失函数使用他们进行梯度的更新。











