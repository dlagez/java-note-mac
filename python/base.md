### conda

- `conda --version` #查看conda版本，验证是否安装

- `conda update conda` #更新至最新版本，也会更新其它相关包

- `conda create -n package_name` #创建名为`env_name`的新环境，并在该环境下安装名为`package_name` 的包，可以指定新环境的版本号，

  例如：`conda create -n python2 python=python2.7 numpy pandas`，创建了`python2`环境，`python`版本为2.7，同时还安装了`numpy pandas`包

- `source activate env_name` #切换至`env_name`环境

- `source deactivate` #退出环境

- `conda info -e` #显示所有已经创建的环境

- `conda remove --name env_name –all` #删除环境

- `conda list` #查看所有已经安装的包

- `conda install matplotlib` 安装库



查看安装包的版本

```python
python -m pip show scikit-learn  # to see which version and where scikit-learn is installed
python -m pip freeze  # to see all packages installed in the active virtualenv
python -c "import sklearn; sklearn.show_versions()"
```





不常用：

- `conda config --set auto_activate_base false：`可以通过配置`auto_activate_base`关闭自动进入`conda`基础环境：`

换源：

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes 



### matplotlib

`pandas`格式画散点图

```python
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据，Molecular_Descriptor.xlsx为自变量，ERα_activity.xlsx为因变量
data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=0)
label = pd.read_excel("ERα_activity.xlsx", sheet_name=0)

# 由于需要画出一个自变量和因变量的散点图，需要把他两合并。
frames = [data, label]
result = pd.concat(frames, axis=1)

# 选出两列来画散点图。SsLi代表一列自变量， pIC50代表因变量
result.plot.scatter(x='SsLi', y='pIC50')
```

折线图：

```python
index = [i for i in range(474)]
plt.plot(index)
plt.show()
```



### numpy

`ndarray`是`numpy`的对象。

- `np.random.randn(10, 10) ` 创建一个`10*10`的数组 

#### 切片：

删除第一列。逗号前面的是操作行，逗号后面的操作列。

```
data_numpy[:, 1:]
```

#### 数据的持久化

```
np.save('data', data)
data = np.load('data.npy')
```

将`pandas`数据编程`numpy`数据：

```python
import pandas as pd
data_numpy = data.values
```



选取前1500行数据

```python
train_data = data_num[:1500, :]
```



反转列表

方法1：`list(reversed(a))        reversed(a)`返回的是迭代器，所以前面加个list转换为list

方法2：`sorted(a,reverse=True)`

方法3：`a[: :-1]`       其中`[::-1]`代表从后向前取值，每次步进值为1

### pandas

选择指定索引的列

```python
index = [585, 406, 659, 508, 652, 11, 525, 3, 718, 96, 293, 97, 4, 43, 98, 587, 44, 720, 728, 594]

data = pd.read_excel("Molecular_Descriptor.xlsx", sheet_name=0, index_col='SMILES').astype(float)
label = pd.read_excel("ERα_activity.xlsx", sheet_name=0, index_col='SMILES').astype(float)

# 选取指定的一列，第五列
data_4 = data.iloc[:, [4]]

# 选取两列
data_45 = data.iloc[:, [3, 4]]

# 选取一个列表里面的列。列表里面的数据为索引，注意，索引以开始。
data_20 = data.iloc[:, index_new]
```



### Jupyter Notebook

安装与使用：

```
pip install notebook
jupyter notebook
```



将环境添加到`notebook`的选择列表里面

```
python -m ipykernel install --user --name pytorch --display-name "pytorch"
```



总结：按h即可查看帮助。

shortcuts ref: [link](https://towardsdatascience.com/jypyter-notebook-shortcuts-bf0101a98330)

- `Shift + Enter` run the current cell, select below
- `Ctrl + Enter` run selected cells
- `Alt + Enter` run the current cell, insert below
- `Ctrl + S` save and checkpoint



While in command mode (press `Esc` to activate):

- `Enter` take you into edit mode
- `H` show all shortcuts
- `Up` select cell above
- `Down` select cell below
- `Shift + Up` extend selected cells above
- `Shift + Down` extend selected cells below
- `A` insert cell above
- `B` insert cell below
- `X` cut selected cells
- `C` copy selected cells
- `V` paste cells below
- `Shift + V` paste cells above
- `D, D (press the key twice)` delete selected cells
- `Z` undo cell deletion
- `S` Save and Checkpoint
- `Y` change the cell type to *Code*
- `M` change the cell type to *Markdown*
- `P` open the command palette. 
- `Shift + Space` scroll notebook up
- `Space` scroll notebook down



### 常用操作

#### 加载`mat`类型的数据：

```
import scipy.io as scio
scio.loadmat("地址")
```

#### 加载`xlsx`文件：

```python
import pandas as pd
import scipy.io as scio
data = pd.read_excel("ADMET.xlsx", sheet_name=0)
data_numpy = data.values
print(data_numpy)
```



#### 归一化：

其中`X`时`（112，448）`的形状

```python
from sklearn import preprocessing
# 用于规范化每个非零采样（或每个非零特征，如果轴为0）的范数。
# axis=0 竖着规范每个特征
X_norm = preprocessing.normalize(X, norm='l2', axis=0)
```

ref：https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html?highlight=normalize#sklearn.preprocessing.normalize



#### 主成分分析：

```python
import sklearn.decomposition as dp
pca = dp.PCA(n_components=3)
pca.fit(X)
print(pca.explained_variance_ratio_)
# [0.98318212 0.00850037 0.00831751]
# 可以看出投影后三个特征维度的方差比例大约为98.3%：0.8%：0.8%。投影后第一个特征占了绝大多数的主成分比例。
print(pca.explained_variance_)
X_new = pca.transform(X)
fig = plt.figure()
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o')
plt.show()
```



#### 画3d图

```python
# 8.画图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 将类别为1的数据画到figure上面
ax.scatter(one_norm[:, 0], one_norm[:, 1], one_norm[:, 2], c='r', marker='o')
# 将类别为0的数据画到figure上面
ax.scatter(zero_norm[:, 0], zero_norm[:, 1], zero_norm[:, 2], c='b', marker='^')

# 9.设置xyz轴
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# 10.显示图像
plt.show()
```



#### 读取文件夹：

```python
import os
dir = '/Users/roczhang/Downloads/2016/'

files = os.listdir(dir)
for file in files:
    print(file)
```



#### 读取`txt`文件：

读取一个文件夹的文件：

```python
import os
dir = '/Users/roczhang/Downloads/2016/'

files = os.listdir(dir)
strs = []
for file in files:
    with open(dir+file, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取
            if not lines:  # 如果读到txt文件结尾了，读下一个文件
                break
            lines = lines.strip('\n')  # 去掉文件结尾的换行符号
            if len(lines) == 0:  # 如果读到空行，不记录
                continue
            strs.append(lines)
print(str)
```



#### 字符串操作：

拼接两个字符串：`str1+str2`

只保留字符串里面的汉字： 

```python
lines = re.sub('[^\u4e00-\u9fa5]+', '', lines)
```



#### 读取和写入ndarray

```
np.save('data', data)
data = np.load('data.npy')
```



### python

`@staticmethod`

`@staticmethod`是一个内置的装饰器，在`Python`的类中定义静态方法。静态方法不会收到任何引用参数，无论是由类的实例还是由类本身调用。

- 在类中声明静态方法。

- 它不能有cls或自参数。

- 静态方法无法访问类属性或实例属性。

- 可以使用`ClassName.MethodName()`和`Object.MethodName()`调用静态方法。

- 它可以返回类的对象。

ref：[link](https://www.tutorialsteacher.com/python/staticmethod-decorator)