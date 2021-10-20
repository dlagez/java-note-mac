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

