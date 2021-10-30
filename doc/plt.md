## 图相关

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

## Y轴相关的设置

设置y轴名：

```python
fig, ax = plt.subplots()
ax.set_ylabel('Scores')set_ylabel
```

## X轴相关的设置

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



## example

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