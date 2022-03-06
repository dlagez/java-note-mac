查看网络结构：

```python
import torchvision.models as models
from torchsummary import summary
import torch
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.vgg19().to(device)
 
summary(vgg, (3, 224, 224))
```

 

### nn.Linear [link](http://www.sharetechnote.com/html/Python_PyTorch_nn_Linear_01.html)

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



### nn.Sequential [link](http://www.sharetechnote.com/html/Python_PyTorch_nn_Sequential_01.html)

**2 Inputs , 1 outputs and Activation Function**

```
net = torch.nn.Sequential(
                         torch.nn.Linear(2,1),
                         torch.nn.Sigmoid()
                         );
```

- ![img](https://cdn.jsdelivr.net/gh/dlagez/img@master/Python_Pytorch_nn_Sequential_i2_o1_sigmoid_01.png)

### Activation Function

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



### view function

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

### np.random.normal

从正太高斯分布中随机抽取样本

```
np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))
```

第一个参数表示：中心的位置

第二个参数表示：宽度

第三个参数表示：样本的形状 （行，列）