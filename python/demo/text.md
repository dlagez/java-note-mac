merge a

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



把列表以txt形式的保存与读取.[code](https://github.com/dlagez/bigdata/blob/master/demo3_/test.py)

```python
key_list = ['环境保护', '环保', '污染', '生态', '生态环境', '生态文明', '污染', '减排', '排污', '能耗', '水耗', '污水处理', '污水治理', '污染防治', '节水',
            '水土保持', '再利用', '节能', '节约', '可持续发展', '新能源', '低碳', '绿色', '绿化', '绿色发展', '空气', '饮水安全', '水质', '化学需氧量', '氨氮',
            '二氧化硫', '二氧化碳', 'PM10', 'PM2.5', '自然资源', '土地资源', '耕地', '水资源', '矿山', '森林', '海洋', '草原', '土壤', '蓝天', '碧水',
            '净土', '农业面污染防治', '自然资源资产离任审计', '自然资源资产负债表', '河长制', '湖长制', '中央环境保护督察']

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



word to txt

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