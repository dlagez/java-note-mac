pandas格式画散点图

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

