项目地址https://github.com/dlagez/gan_resnet

由于项目中的resnet方法需要修改一下。

残差块：

```python
class BasicBlock(nn.Module):
    def __init__(self, in1):  # in1为输入的channel大小，BasicBlock输出等于输入channel大小
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in1, in1 * 2, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in1*2)
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
```

残差块：

 

在这个残差块的最后使用`out = out + residual`来将两个层合并，这个直接相加是不会改变输出通道的层数的。相当于是把两个层的数值相加之后输出。



现在的需求是



TORCH.CAT 使用方法：https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat

