- np.random.randn(10, 10)  创建一个10*10的数组 

### 切片：

删除第一列。逗号前面的是操作行，逗号后面的操作列。

```
data_numpy[:, 1:]
```



将pandas数据编程numpy数据：

```python
import pandas as pd
data_numpy = data.values
```



选取前1500行数据

```python
train_data = data_num[:1500, :]
```



反转列表

方法1：list(reversed(a))        reversed(a)返回的是迭代器，所以前面加个list转换为list

方法2：sorted(a,reverse=True)

方法3：a[: :-1]       其中[::-1]代表从后向前取值，每次步进值为1
