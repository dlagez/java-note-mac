merge a

### fold txt to merged txt

将一个文件夹的txt文本合并到一个文本里面。code

```python
import os
path = "/Volumes/roczhang/WHPU/zen/text2"
file_list = os.listdir(path)
f = open('/Volumes/roczhang/WHPU/zen/merged.txt', 'w')
for filename in file_list:
    filepath = path + '/' + filename
    for line in open(filepath):
        f.writelines(line)
f.close()
```



### list to txt or text to list

把列表以txt形式的保存与读取.[code](https://github.com/dlagez/bigdata/blob/master/demo3_/test.py)

```python
key_list = ['环境保护', '环保']

with open('/Volumes/roczhang/temp/list.txt', 'w') as f:
    for i in key_list:
        f.write(i+'\n')

file = open('/Volumes/roczhang/temp/list.txt')
lines = file.readlines()
for line in lines:
    line = line.strip('\n')
    print(line)

read_line = []
for line in lines:
    line = line.strip('\n')
    read_line.append(line)
```



{} dict to txt

```

```



### word to txt

把一个文件夹的word文档转换成相应的txt文件。只会读取文字。[code](https://github.com/dlagez/bigdata/blob/master/demo3_/word_to_txt.py)

```python
from docx import Document
import os
# 将一个文件夹的word文件转换称txt文件
path = '/Volumes/roczhang/WHPU/zen/政策文本'
path_txt = '/Volumes/roczhang/WHPU/zen/test'  # 这个文件夹用来装txt文件
file_list = os.listdir(path)  # 读取出docx文件夹所有的文件名字

for file in file_list:
    file_path = path + '/' + file
    # print(file_path)
    doc = Document(file_path)  # 读取docx文件
    f = open(path_txt + '/' + file.split('.')[0] + '.txt', 'a')  # 把.docx的后缀改成txt，并创建txt文件。
    print(file_path + '\n')
    for paragraph in doc.paragraphs:
        f.write(paragraph.text)  # 将docx段落写入txt文件
        print(paragraph.text + '\n')
    f.close()  # txt文件使用完成后关闭

# 更好的写法
for file in file_list:
    file_path = path + '/' + file
    # print(file_path)
    doc = Document(file_path)  # 读取docx文件
    with open(path_txt + '/' + file.split('.')[0] + '.txt', 'a') as f:
        for paragraph in doc.paragraphs:
            f.write(paragraph.text)  # 将docx段落写入txt文件
            print(paragraph.text + '\n')
```



### 二维list to txt

将二维list写入文件

```python
output = open('/Volumes/roczhang/WHPU/zen/result.txt', 'w')
for i in range(len(result)):
    output.write('第' + str(i) + '篇文章!' + '\n')  # 在每篇（每一行开始的时候做一个标记）
    for j in range(len(result[i])):
        # print(result[i][j])
        output.write(str(result[i][j]))  # 写入一行数据
        output.write(' ')
    output.write('\n\n')  # 每写完一行数据之后按两次回撤键
output.close()
```

效果是这样的

![image-20220107162556713](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220107162556713.png)

### list 统计元素个数

```
from collections import Counter
a = [1, 2, 3, 1, 1, 2]
result = Counter(a)
```



### {} sorted

按照key排序

```
my_dict = {'lilee':25, 'age':24, 'phone':12}
sorted(my_dict.keys())
输出结果为

['age', 'lilee', 'phone']
```

key使用lambda匿名函数取value进行排序

```

d = {'lilee':25, 'wangyan':21, 'liqun':32, 'age':19}
sorted(d.items(), key=lambda item:item[1])
输出结果为

[('age',19),('wangyan',21),('lilee',25),('liqun',32)]
如果需要倒序则

sorted(d.items(), key=lambda item:item[1], reverse=True)
得到的结果就会是

[('liqun',32),('lilee',25),('wangyan',21),('age',19)]
```

