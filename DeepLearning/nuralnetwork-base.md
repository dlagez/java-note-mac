### 激活函数

激活函数可以引入非线性因素，如果不使用激活函数，则输出信号仅仅是一个简单的线性函数。线性函数的复杂度有限。常见的有：

sigmoid 激活函数、tanh激活函，Relu激活函数

<img src="https://cdn.jsdelivr.net/gh/dlagez/img@master/3-26.png" alt="img" style="zoom:50%;" />

### Softmax 定义及作用

可以把输入处理称0-1之间，并且能够把输出归一化到和为 1



### 神经网络越深越好吗？

理论上，在训练集上，即越深的网络不会比浅层的网络效果差

但是随着层数的增多，训练集上的效果变差？这被称为**退化问题（degradation problem）**，原因是随着网络越来越深，训练变得原来越难，网络的优化变得越来越难。



ref：

- https://github.com/scutan90/DeepLearning-500-questions