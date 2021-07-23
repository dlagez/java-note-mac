### Python3 基本数据类型

Python 中的变量不需要声明。每个变量在使用前都必须赋值，变量赋值以后该变量才会被创建。

在 Python 中，变量就是变量，它没有类型，我们所说的"类型"是变量所指的内存中对象的类型。

```python
counter = 100          # 整型变量
miles   = 1000.0       # 浮点型变量
name    = "runoob"     # 字符串
```

### String（字符串）

Python中的字符串用单引号 **'** 或双引号 **"** 括起来

```python
str = 'Runoob'

print (str)          # 输出字符串
print (str[0:-1])    # 输出第一个到倒数第二个的所有字符
print (str[0])       # 输出字符串第一个字符
print (str[2:5])     # 输出从第三个开始到第五个的字符
print (str[2:])      # 输出从第三个开始的后的所有字符
print (str * 2)      # 输出字符串两次，也可以写成 print (2 * str) 
print (str + "TEST") # 连接字符串

输出的结果
Runoob
Runoo
R
noo
noob
RunoobRunoob
RunoobTEST
```

### List（列表）

List（列表） 是 Python 中使用最频繁的数据类型。

列表可以完成大多数集合类的数据结构实现。列表中元素的类型可以不相同，它支持数字，字符串甚至可以包含列表（所谓嵌套）。

```python
list = [ 'abcd', 786 , 2.23, 'runoob', 70.2 ]
tinylist = [123, 'runoob']

print (list)            # 输出完整列表
print (list[0])         # 输出列表第一个元素
print (list[1:3])       # 从第二个开始输出到第三个元素
print (list[2:])        # 输出从第三个元素开始的所有元素
print (tinylist * 2)    # 输出两次列表
print (list + tinylist) # 连接列表

['abcd', 786, 2.23, 'runoob', 70.2]
abcd
[786, 2.23]
[2.23, 'runoob', 70.2]
[123, 'runoob', 123, 'runoob']
['abcd', 786, 2.23, 'runoob', 70.2, 123, 'runoob']
```

与Python字符串不一样的是，列表中的元素是可以改变的：

```python
>>> a = [1, 2, 3, 4, 5, 6]
>>> a[0] = 9
>>> a[2:5] = [13, 14, 15]
>>> a
[9, 2, 13, 14, 15, 6]
>>> a[2:5] = []   # 将对应的元素值设置为 [] 
>>> a
[9, 2, 6]
```

### Tuple（元组）

元组（tuple）与列表类似，不同之处在于元组的元素不能修改。元组写在小括号 **()** 里，元素之间用逗号隔开。

```python
tuple = ( 'abcd', 786 , 2.23, 'runoob', 70.2 )
tinytuple = (123, 'runoob')

**print** (tuple)       # 输出完整元组
**print** (tuple[0])      # 输出元组的第一个元素
**print** (tuple[1:3])     # 输出从第二个元素开始到第三个元素
**print** (tuple[2:])     # 输出从第三个元素开始的所有元素
**print** (tinytuple * 2)   # 输出两次元组
**print** (tuple + tinytuple) # 连接元组
```

### Set（集合）

集合（set）是由一个或数个形态各异的大小整体组成的，构成集合的事物或对象称作元素或是成员。

基本功能是进行成员关系测试和删除重复元素。

```python
sites = {'Google', 'Taobao', 'Runoob', 'Facebook', 'Zhihu', 'Baidu'}

print(sites)   # 输出集合，重复的元素被自动去掉

# 成员测试
if 'Runoob' in sites :
    print('Runoob 在集合中')
else :
    print('Runoob 不在集合中')


# set可以进行集合运算
a = set('abracadabra')
b = set('alacazam')

print(a)

print(a - b)     # a 和 b 的差集

print(a | b)     # a 和 b 的并集

print(a & b)     # a 和 b 的交集

print(a ^ b)     # a 和 b 中不同时存在的元素
```

Dictionary（字典）

```python
dict = {}
dict['one'] = "1 - 菜鸟教程"
dict[2]     = "2 - 菜鸟工具"

tinydict = {'name': 'runoob','code':1, 'site': 'www.runoob.com'}


print (dict['one'])       # 输出键为 'one' 的值
print (dict[2])           # 输出键为 2 的值
print (tinydict)          # 输出完整的字典
print (tinydict.keys())   # 输出所有键
print (tinydict.values()) # 输出所有值

# 结果
1 - 菜鸟教程
2 - 菜鸟工具
{'name': 'runoob', 'code': 1, 'site': 'www.runoob.com'}
dict_keys(['name', 'code', 'site'])
dict_values(['runoob', 1, 'www.runoob.com'])
```

### 循环语句

Python for 循环可以遍历任何可迭代对象，如一个列表或者一个字符串。

```python
for循环的一般格式如下：
for <variable> in <sequence>:
    <statements>
else:
    <statements>
```

```python
>>>languages = ["C", "C++", "Perl", "Python"] 
>>> for x in languages:
...     print (x)

C
C++
Perl
Python
```



### range函数

如果你需要遍历数字序列，可以使用内置range()函数。它会生成数列，例如:

```python
>>>for i in range(5):
...     print(i)
...
0
1
2
3
4
```



### 分类：

```python
# 进行随机下采样，并使用MLP分类
import scipy
import scipy.io as scio
import numpy as np
import os
# 读取训练和测试数据
data_root = '/data/file/classification_data/tmp/years/sum_2016'
trainX_norm = scio.loadmat(os.path.join(data_root, 'train_sumX_2016_norm.mat'))['X']
testX_norm = scio.loadmat(os.path.join(data_root, 'test_sumX_2016_norm.mat'))['X']

trainlabel = scio.loadmat(os.path.join(data_root, 'train_sumy_2016_label.mat'))['y']
testlabel = scio.loadmat(os.path.join(data_root, 'test_sumy_2016_label.mat'))['y']

# 导入采样方法
from imblearn.over_sampling import ADASYN 
ada = ADASYN(random_state=42)
X_re, y_re = ada.fit_resample(trainX_norm, trainlabel)
X_re
y_re = y_re.reshape(-1, 1)

# 导入分类的方法
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC
clf = make_pipeline(StandardScaler(), NuSVC())
clf.fit(X_re, y_re)

clf.score(testX_norm, testlabel)
```

