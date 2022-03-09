### 执行命令：

```
python main.py --model SVM --dataset IndianPines --training_sample 0.3
```

--training_sample 0.3: using 30% of the samples for training and the rest for testing

可选参数：

- --epoch
- --restore  使用checkpoint，接着上次的训练
- --lr
- --batch_size
- --folder 指定数据集文件夹，一般不使用



### 模型的种类：

深度学习类型：

- nn
- hamida
- lee
- chen
- li
- hu
- he
- luo
- sharma
- liu
- boulch
- mou

传统方法：

- SVM_grid
- SVM
- SGD
- nearest



数据类型：

- Botswana
- DFC2018_HSI
- IndianPines
- KSC
- PaviaC
- PaviaU