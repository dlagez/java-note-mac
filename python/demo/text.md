merge a

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