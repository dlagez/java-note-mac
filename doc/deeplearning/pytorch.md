## 整个网络定义训练的流程：

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

