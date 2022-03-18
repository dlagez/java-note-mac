### 执行命令：

```
python main.py --model SVM --dataset IndianPines --training_sample 0.09 --epoch 300 --cuda 0
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



### 实验结果总结：

目前只做了IndianPines数据集的实验

| method | epochs   | 训练集百分比 | 准确率  |      |
| ------ | -------- | ------------ | ------- | ---- |
| SVM    | 300      | 0.09         | 50.509% |      |
|        | 300      | 0.30         | 52.432% |      |
|        | 300      | 0.40         | 53.236% |      |
|        | 300      | 0.75         | 56.067% |      |
| nn     | 300      | 0.09         | 77.592% |      |
|        | 300      | 0.20         | 83.415% |      |
|        | 500      | 0.40         | 90.780% |      |
| hu     | 300      | 0.09         | 42.243% |      |
|        | 300      | 0.20         | 46.549% |      |
|        | 500      | 0.50         | 57.580% |      |
|        | 1000     | 0.60         | 76.122% |      |
| hamida | 300      | 0.09         | 66.870% |      |
|        | 500      | 0.20         | 84.037% |      |
|        | 1000     | 0.40         | 92.797% |      |
| lee    | 300      | 0.09         | 59.033% |      |
|        | 400      | 0.20         | 78.780% |      |
|        | 1000     | 0.40         | 93.675% |      |
| chen   | 300      | 0.09         | 47.325% |      |
|        | 400      | 0.30         | 72.697% |      |
| li     | 500      | 0.40         | 72.697% |      |
|        | 600      | 0.50         | 93.385% |      |
| he     | 600      | 0.30         | 92.920% |      |
| luo    | 1000     | 0.40         | 45.707% |      |
| liu    | 代码报错 |              |         |      |
|        |          |              |         |      |
|        |          |              |         |      |

后面的几个3d网络运行的太慢了，1000个epochs跑了一上午还没跑完。



实验室电脑上运行：

### SVM  

先看下他的默认参数

models.py

```
kwargs.setdefault("epoch", 100)
kwargs.setdefault("batch_size", 100)
```

#### 一：

百分之九的训练集。

```
python main.py --model SVM --dataset IndianPines --training_sample 0.09 --epoch 300 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model SVM --dataset IndianPines --trai
ning_sample 0.09 --epoch 300 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is dep
recated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.
array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is dep
recated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.
array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
922 samples selected (over 10249)
Running an experiment with the SVM model run 1/1
Saving model params in 2022_03_16_09_32_44
D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    0    0   41    0    0    1    0    0
     0    0    0]
 [   0    0  365    0    0    0    5    0    1    0    0  928    0    0
     0    0    0]
 [   0    0   75    0    0    0    1    0    0    0    0  679    0    0
     0    0    0]
 [   0    0   88    0    0    0   27    0    1    0    0  100    0    0
     0    0    0]
 [   0    0    0    0    0    0  112    0   21    0    0   20    0    0
   287    0    0]
 [   0    0    0    0    0    1  611    0    0    0    0    8    0    0
    44    0    0]
 [   0    0    0    0    0    0    0    0   18    0    0    7    0    0
     0    0    0]
 [   0    0    0    0    0    0    5    0  430    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0   18    0    0    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    3    0    3    0    0  879    0    0
     0    0    0]
 [   0    0   47    0    0    0   18    0    6    0    0 2163    0    0
     0    0    0]
 [   0    0   64    0    0    0    3    0    0    0    0  473    0    0
     0    0    0]
 [   0    0    0    0    0    0  186    0    0    0    0    1    0    0
     0    0    0]
 [   0    0    0    0    0    1    8    0    0    0    0    0    0    0
  1142    0    0]
 [   0    0    0    0    0    0  192    0    0    0    0   18    1    0
   140    0    0]
 [   0    0   15    0    0    0    0    0    0    0    0   70    0    0
     0    0    0]]---
Accuracy : 50.509%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.000
        Corn-notill: 0.374
        Corn-mintill: 0.000
        Corn: 0.000
        Grass-pasture: 0.000
        Grass-trees: 0.659
        Grass-pasture-mowed: 0.000
        Hay-windrowed: 0.900
        Oats: 0.000
        Soybean-notill: 0.000
        Soybean-mintill: 0.571
        Soybean-clean: 0.000
        Wheat: 0.000
        Woods: 0.826
        Buildings-Grass-Trees-Drives: 0.000
        Stone-Steel-Towers: 0.000
---
Kappa: 0.396

```

#### 二：

百分之三十的训练集

```
python main.py --model SVM --dataset IndianPines --training_sample 0.3 --epoch 300 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model SVM --dataset IndianPines --trai
ning_sample 0.3 --epoch 300 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is dep
recated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.
array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is dep
recated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.
array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
3074 samples selected (over 10249)
Running an experiment with the SVM model run 1/1
Saving model params in 2022_03_16_09_34_06
D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    0    0   31    0    0    1    0    0
     0    0    0]
 [   0    0  341    0    0    0    3    0    2    0    0  654    0    0
     0    0    0]
 [   0    0   71    0    0    0    2    0    0    0    0  508    0    0
     0    0    0]
 [   0    0   79    0    0    0   24    0    0    0    0   63    0    0
     0    0    0]
 [   0    0    0    0    0    5   66    0   23    0    0   17    0    0
   227    0    0]
 [   0    0    0    0    0   11  488    0    1    0    0    2    0    0
     9    0    0]
 [   0    0    0    0    0    0    0    0   18    0    0    2    0    0
     0    0    0]
 [   0    0    0    0    0    1    1    0  333    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0   14    0    0    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    1    0    2    0    0  677    0    0
     0    0    0]
 [   0    0   49    0    0    0   13    0    7    0    0 1650    0    0
     0    0    0]
 [   0    0   63    0    0    0    1    0    0    0    0  351    0    0
     0    0    0]
 [   0    0    0    0    0    0  117    0    0    0    0    0    0    3
     0   23    0]
 [   0    0    0    0    0    2    6    0    0    0    0    0    0    1
   876    1    0]
 [   0    0    0    0    0    3  137    0    1    0    0   13    1    3
   100   12    0]
 [   0    0    5    0    0    0    0    0    0    0    1    5    0    0
     0    0   54]]---
Accuracy : 52.432%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.000
        Corn-notill: 0.424
        Corn-mintill: 0.000
        Corn: 0.000
        Grass-pasture: 0.028
        Grass-trees: 0.705
        Grass-pasture-mowed: 0.000
        Hay-windrowed: 0.884
        Oats: 0.000
        Soybean-notill: 0.000
        Soybean-mintill: 0.583
        Soybean-clean: 0.000
        Wheat: 0.040
        Woods: 0.835
        Buildings-Grass-Trees-Drives: 0.078
        Stone-Steel-Towers: 0.908
---
Kappa: 0.423
```

#### 三：

百分之四十训练集

```
python main.py --model SVM --dataset IndianPines --training_sample 0.4 --epoch 300 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model SVM --dataset IndianPines --trai
ning_sample 0.4 --epoch 300 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is dep
recated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.
array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is dep
recated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.
array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
4099 samples selected (over 10249)
Running an experiment with the SVM model run 1/1
Saving model params in 2022_03_16_09_35_19
D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    0    0   26    0    0    1    0    0
     0    0    0]
 [   0    0  284    0    0    0    3    0    2    0    0  568    0    0
     0    0    0]
 [   0    0   66    0    0    0    2    0    0    0    0  430    0    0
     0    0    0]
 [   0    0   63    0    0    0   26    0    1    0    0   52    0    0
     0    0    0]
 [   0    0    2    0    0    5   59    0   22    0    0   13    0    0
   189    0    0]
 [   0    0    0    0    0    9  419    0    0    0    0    2    0    0
     7    1    0]
 [   0    0    0    0    0    0    0    0   15    0    0    2    0    0
     0    0    0]
 [   0    0    0    0    0    0    2    0  285    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0   12    0    0    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    0    0    3    0    0  580    0    0
     0    0    0]
 [   0    0   55    0    0    0    9    0    7    0    0 1402    0    0
     0    0    0]
 [   0    0   59    0    0    0    1    0    0    0    0  296    0    0
     0    0    0]
 [   0    0    0    0    0    0   47    0    0    0    0    1    0   69
     0    6    0]
 [   0    0    0    0    0    2    2    0    0    0    0    0    0    0
   752    3    0]
 [   0    0    0    0    0    7  112    0    0    0    0   10    3    7
    81   12    0]
 [   0    0    6    0    0    0    0    0    0    0    1    3    0    0
     0    0   46]]---
Accuracy : 53.236%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.000
        Corn-notill: 0.408
        Corn-mintill: 0.000
        Corn: 0.000
        Grass-pasture: 0.032
        Grass-trees: 0.740
        Grass-pasture-mowed: 0.000
        Hay-windrowed: 0.880
        Oats: 0.000
        Soybean-notill: 0.000
        Soybean-mintill: 0.580
        Soybean-clean: 0.000
        Wheat: 0.693
        Woods: 0.841
        Buildings-Grass-Trees-Drives: 0.094
        Stone-Steel-Towers: 0.902
---
Kappa: 0.433
```

#### 四：

百分之七十五训练集：

```
python main.py --model SVM --dataset IndianPines --training_sample 0.75 --epoch 300 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model SVM --dataset IndianPines --trai
ning_sample 0.75 --epoch 300 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is dep
recated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.
array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is dep
recated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.
array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
7686 samples selected (over 10249)
Running an experiment with the SVM model run 1/1
Saving model params in 2022_03_16_09_37_01
D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  11   0   0   0   0   0   0   0   0]
 [  0   0 141   0   0   0   1   0   0   0   1 214   0   0   0   0   0]
 [  0   0  23   0   0   0   1   0   0   0   0 184   0   0   0   0   0]
 [  0   0  24   0   1   0   9   0   1   0   0  24   0   0   0   0   0]
 [  0   0   1   0   0  21   5   0   4   0   0   7   0   0  83   0   0]
 [  0   0   0   0   0   3 178   0   0   0   0   0   0   0   2   0   0]
 [  0   0   0   0   0   0   0   0   6   0   0   1   0   0   0   0   0]
 [  0   0   0   0   0   1   0   0 119   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   5   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   2   0   1   0   1 239   0   0   0   0   0]
 [  0   0  18   0   0   2   5   0   1   0   0 588   0   0   0   0   0]
 [  0   0  25   0   0   0   1   0   0   0   1 121   0   0   0   0   0]
 [  0   0   0   0   0   0   2   0   0   0   0   0   0  49   0   0   0]
 [  0   0   0   0   0   0   2   0   0   0   0   0   0   0 314   0   0]
 [  0   0   0   0   0   4  38   0   0   0   0   4   0   8  38   5   0]
 [  0   0   3   0   0   0   0   0   0   0   0   0   0   0   0   0  20]]---
Accuracy : 56.067%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.000
        Corn-notill: 0.476
        Corn-mintill: 0.000
        Corn: 0.033
        Grass-pasture: 0.276
        Grass-trees: 0.824
        Grass-pasture-mowed: 0.000
        Hay-windrowed: 0.905
        Oats: 0.000
        Soybean-notill: 0.008
        Soybean-mintill: 0.589
        Soybean-clean: 0.000
        Wheat: 0.907
        Woods: 0.834
        Buildings-Grass-Trees-Drives: 0.098
        Stone-Steel-Towers: 0.930
---
Kappa: 0.468
```



### nn

#### 一：

只用百分之九的训练集。

看了下训练的损失和准确率，感觉应该还是可以多训练几个epochs。下轮先提高一下训练集的数量。

![image-20220316100157641](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316100200.png)

```
python main.py --model nn --dataset IndianPines --training_sample 0.09 --epoch 300 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model nn --dataset IndianPines --training_sample 0.09 --epoch 300 --cuda 0
922 samples selected (over 10249)
{'dataset': 'IndianPines', 'model': 'nn', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.09, 'sampling_mode': 'random',
 'epoch': 300, 'class_balancing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentatio
n': False, 'with_exploration': False, 'n_classes': 17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weigh
ts': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       device='cuda:0'), 'patch_size': 1, 'dropout': False, 'learning_rate': 0.0001, 'batch_size': 100, 'scheduler': <torch.optim.lr_schedul
er.ReduceLROnPlateau object at 0x0000016B89DAFB08>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                 [-1, 2048]         411,648
            Linear-2                 [-1, 4096]       8,392,704
            Linear-3                 [-1, 2048]       8,390,656
            Linear-4                   [-1, 17]          34,833
================================================================
Total params: 17,229,841
Trainable params: 17,229,841
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.06
Params size (MB): 65.73
Estimated Total Size (MB): 65.79
----------------------------------------------------------------

Inference on the image: 211it [00:01, 169.50it/s]
D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0   13    0    0    0    1    0    0   28    0    0    0    0    0
     0    0    0]
 [   0    0  845  113   64    2    2    0    0    0   68  126   78    0
     0    1    0]
 [   0    0   69  535   18    0    1    0    0    0    7   74   51    0
     0    0    0]
 [   0    0   34   16  131    0   11    0    4    0    1    2   17    0
     0    0    0]
 [   0    1    2    1    6  379    5   10    2    0    1   11    9    0
     8    5    0]
 [   0    0    0    0    0    0  625    0    0    5   10    1    0    0
     6   17    0]
 [   0    0    0    0    0    1    0   13   10    0    1    0    0    0
     0    0    0]
 [   0    4    0    0    0    3    0    0  428    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    6    0    0    3    0    0    0    0
     0    9    0]
 [   0    0   24   31   19    0    1    1    0    0  653  134   22    0
     0    0    0]
 [   0    0  153  124   36    6    1    1    5    0  154 1703   43    0
     0    8    0]
 [   0    0   39   25   14    0    0    0    1    0   32   22  405    0
     0    1    1]
 [   0    0    0    0    0    0    1    0    0    0    0    1    0  177
     0    8    0]
 [   0    0    0    0    0   29    3    0    0    0    0    0    0    2
  1096   21    0]
 [   0    0    0    1    4   11   38    0    0    1    0    0    3   38
    95  160    0]
 [   0    0    3    2    0    0    0    0    0    0    0    9    0    0
     0    0   71]]---
Accuracy : 77.592%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.433
        Corn-notill: 0.685
        Corn-mintill: 0.667
        Corn: 0.516
        Grass-pasture: 0.869
        Grass-trees: 0.920
        Grass-pasture-mowed: 0.520
        Hay-windrowed: 0.938
        Oats: 0.222
        Soybean-notill: 0.721
        Soybean-mintill: 0.789
        Soybean-clean: 0.693
        Wheat: 0.876
        Woods: 0.930
        Buildings-Grass-Trees-Drives: 0.551
        Stone-Steel-Towers: 0.904
---
Kappa: 0.745
```

#### 二：

用百分之二十的训练集。看了下迭代的曲线，感觉是应该可以再多训练几个epoch。准确率到了Accuracy : 83.415%，和一些论文里面的结果相近了。

![image-20220316100744921](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316100747.png)

```
python main.py --model nn --dataset IndianPines --training_sample 0.2 --epoch 300 --cuda 0
```

```
2049 samples selected (over 10249)
{'dataset': 'IndianPines', 'model': 'nn', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.2, 'sampling_mode': 'random', 'epoch': 300, 'class_bala
ncing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n_classes': 
17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1.],
       device='cuda:0'), 'patch_size': 1, 'dropout': False, 'learning_rate': 0.0001, 'batch_size': 100, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau obje
ct at 0x000002966527A508>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                 [-1, 2048]         411,648
            Linear-2                 [-1, 4096]       8,392,704
            Linear-3                 [-1, 2048]       8,390,656
            Linear-4                   [-1, 17]          34,833
================================================================
Total params: 17,229,841
Trainable params: 17,229,841
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.06
Params size (MB): 65.73
Estimated Total Size (MB): 65.79
----------------------------------------------------------------

D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0   29    0    0    0    1    0    0    7    0    0    0    0    0
     0    0    0]
 [   0    0  748   60   62    1    2    0    0    0   18  185   65    0
     0    0    2]
 [   0    0    7  483   38    1    1    0    0    0    0  105   29    0
     0    0    0]
 [   0    0    0   10  167    0    1    0    1    0    1    4    6    0
     0    0    0]
 [   0    1    0    0    1  356    5    0    0    0    0    6    7    0
    10    0    0]
 [   0    0    0    0    1    0  578    0    0    0    0    1    0    0
     3    1    0]
 [   0    0    0    0    0    1    0   20    1    0    0    0    0    0
     0    0    0]
 [   0    3    0    0    0    3    0    1  375    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    2    0    0   12    0    0    0    0
     0    2    0]
 [   0    0   57   10    2    2    5    0    0    0  511  154   36    0
     0    1    0]
 [   0    1   18   44   12    9    3    1    0    0   25 1808   36    0
     0    7    0]
 [   0    2    2   21   20    3    0    0    0    0    1   16  407    0
     0    1    2]
 [   0    0    0    0    0    1    1    0    0    0    0    0    0  162
     0    0    0]
 [   0    0    0    0    1    5    2    0    0    0    0    0    0    0
   992   12    0]
 [   0    1    0    1    5   11   40    0    1    2    0    1    0    6
   106  131    4]
 [   0    0    0    2    0    0    0    0    0    0    0   11    0    0
     0    0   61]]---
Accuracy : 83.415%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.784
        Corn-notill: 0.757
        Corn-mintill: 0.746
        Corn: 0.669
        Grass-pasture: 0.913
        Grass-trees: 0.944
        Grass-pasture-mowed: 0.909
        Hay-windrowed: 0.978
        Oats: 0.800
        Soybean-notill: 0.766
        Soybean-mintill: 0.850
        Soybean-clean: 0.767
        Wheat: 0.976
        Woods: 0.935
        Buildings-Grass-Trees-Drives: 0.565
        Stone-Steel-Towers: 0.853
---
Kappa: 0.810

```

#### 三：

使用百分之四十的数据作为训练集，迭代500轮。训练准确率为90.78%

![image-20220316103011008](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316103013.png)

```
python main.py --model nn --dataset IndianPines --training_sample 0.4 --epoch 500 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model nn --dataset IndianPines --training_sample 0.4 --epoch 500 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
4099 samples selected (over 10249)
Running an experiment with the nn model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'nn', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.4, 'sampling_mode': 'random', 'epoch': 500, 'class_bala
ncing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n_classes': 
17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1.],
       device='cuda:0'), 'patch_size': 1, 'dropout': False, 'learning_rate': 0.0001, 'batch_size': 100, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau obje
ct at 0x00000266D9F23E48>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                 [-1, 2048]         411,648
            Linear-2                 [-1, 4096]       8,392,704
            Linear-3                 [-1, 2048]       8,390,656
            Linear-4                   [-1, 17]          34,833
================================================================
Total params: 17,229,841
Trainable params: 17,229,841
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.06
Params size (MB): 65.73
Estimated Total Size (MB): 65.79
----------------------------------------------------------------

D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0   26    0    0    0    0    0    0    0    0    0    1    0    0
     0    0    0]
 [   0    0  744   14    3    0    0    0    0    0   27   59   10    0
     0    0    0]
 [   0    0    8  419   20    0    0    0    0    1    3   35   11    0
     0    1    0]
 [   0    0    3   16  114    2    0    0    1    0    4    1    1    0
     0    0    0]
 [   0    1    0    0    1  274    4    0    0    0    1    2    3    0
     2    2    0]
 [   0    0    0    0    0    0  430    0    0    0    1    0    0    0
     0    7    0]
 [   0    0    0    0    0    1    0   14    2    0    0    0    0    0
     0    0    0]
 [   0    2    0    0    0    0    0    0  285    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    0    0    0    8    0    0    0    0
     0    4    0]
 [   0    0   22    1    1    0    3    0    0    0  511   42    3    0
     0    0    0]
 [   0    0   38   16    4    4    1    0    0    0   33 1358   17    0
     0    2    0]
 [   0    0    3    8    5    2    0    0    0    0    4    9  323    0
     0    2    0]
 [   0    0    0    0    0    0    0    0    0    0    0    1    0  122
     0    0    0]
 [   0    0    0    0    0    4    0    0    0    0    0    0    0    1
   747    7    0]
 [   0    0    0    0    0    3    8    0    1    2    1    3    4    4
    52  153    1]
 [   0    0    1    0    0    0    0    0    0    0    0    0    0    0
     0    0   55]]---
Accuracy : 90.780%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.929
        Corn-notill: 0.888
        Corn-mintill: 0.862
        Corn: 0.786
        Grass-pasture: 0.945
        Grass-trees: 0.973
        Grass-pasture-mowed: 0.903
        Hay-windrowed: 0.990
        Oats: 0.696
        Soybean-notill: 0.875
        Soybean-mintill: 0.910
        Soybean-clean: 0.887
        Wheat: 0.976
        Woods: 0.958
        Buildings-Grass-Trees-Drives: 0.746
        Stone-Steel-Towers: 0.982
---
Kappa: 0.895

```





### hu

#### 一：

可以看出来训练并不是很稳定。下一步提升训练数据的百分比。

![image-20220316103429188](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316103431.png)

```
python main.py --model hu --dataset IndianPines --training_sample 0.09 --epoch 300 --cuda 0
```

```
922 samples selected (over 10249)
Running an experiment with the hu model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'hu', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.09, 'sampling_mode': 'random', 'epoch': 300, 'class_bal
ancing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n_classes':
 17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
, 1.],
       device='cuda:0'), 'patch_size': 1, 'learning_rate': 0.01, 'batch_size': 100, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x00000132433D
3E48>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1              [-1, 20, 178]             480
         MaxPool1d-2               [-1, 20, 35]               0
            Linear-3                  [-1, 100]          70,100
            Linear-4                   [-1, 17]           1,717
================================================================
Total params: 72,297
Trainable params: 72,297
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.03
Params size (MB): 0.28
Estimated Total Size (MB): 0.31
D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0   41    0    0    0    0    1    0    0
     0    0    0]
 [   0    0    0    0    0    0    7    0    1    0    0 1290    0    0
     1    0    0]
 [   0    0    0    0    0    0    2    0    0    0    0  753    0    0
     0    0    0]
 [   0    0    0    0    0    0   33    0    5    0    0  178    0    0
     0    0    0]
 [   0    0    0    0    0    0  121    0    3    0    0   26    0    0
   290    0    0]
 [   0    0    0    0    0    0  469    0    4    0    0    3    0    0
   188    0    0]
 [   0    0    0    0    0    0    2    0    3    0    0   20    0    0
     0    0    0]
 [   0    0    0    0    0    0  240    0  111    0    0   84    0    0
     0    0    0]
 [   0    0    0    0    0    0   18    0    0    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    1    0    4    0    0  880    0    0
     0    0    0]
 [   0    0    0    0    0    0   16    0    3    0    0 2211    0    0
     4    0    0]
 [   0    0    0    0    0    0    4    0    2    0    0  534    0    0
     0    0    0]
 [   0    0    0    0    0    0  160    0    0    0    0    1    0    0
    26    0    0]
 [   0    0    0    0    0    0    2    0    0    0    0    0    0    0
  1149    0    0]
 [   0    0    0    0    0    0  144    0    1    0    0   14    0    0
   192    0    0]
 [   0    0    0    0    0    0    0    0    0    0    0   85    0    0
     0    0    0]]---
Accuracy : 42.243%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.000
        Corn-notill: 0.000
        Corn-mintill: 0.000
        Corn: 0.000
        Grass-pasture: 0.000
        Grass-trees: 0.488
        Grass-pasture-mowed: 0.000
        Hay-windrowed: 0.388
        Oats: 0.000
        Soybean-notill: 0.000
        Soybean-mintill: 0.532
        Soybean-clean: 0.000
        Wheat: 0.000
        Woods: 0.766
        Buildings-Grass-Trees-Drives: 0.000
        Stone-Steel-Towers: 0.000
---
Kappa: 0.286
```



#### 二：

发现提升了训练集并没有帮助网络准确率的提升

```
python main.py --model hu --dataset IndianPines --training_sample 0.2 --epoch 300 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model hu --dataset IndianPines --training_sample 0.2 --epoch 300 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
2049 samples selected (over 10249)
Running an experiment with the hu model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'hu', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.2, 'sampling_mode': 'random', 'epoch': 300, 'class_bala
ncing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n_classes': 
17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1.],
       device='cuda:0'), 'patch_size': 1, 'learning_rate': 0.01, 'batch_size': 100, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x00000239A64F
43C8>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1              [-1, 20, 178]             480
         MaxPool1d-2               [-1, 20, 35]               0
            Linear-3                  [-1, 100]          70,100
            Linear-4                   [-1, 17]           1,717
================================================================
Total params: 72,297
Trainable params: 72,297
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.03
Params size (MB): 0.28
Estimated Total Size (MB): 0.31

nference on the image: 211it [00:00, 282.63it/s]
D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    8    0   28    0    0    1    0    0
     0    0    0]
 [   0    0    0    0    0    0    5    0   12    0    0 1126    0    0
     0    0    0]
 [   0    0    0    0    0    0    0    0    3    0    0  661    0    0
     0    0    0]
 [   0    0    0    0    0    0   16    0   39    0    0  135    0    0
     0    0    0]
 [   0    0    0    0    0    0   90    0   33    0    0    6    0    0
   257    0    0]
 [   0    0    0    0    0    3  525    0   29    0    0    0    0    0
    27    0    0]
 [   0    0    0    0    0    0    1    0   21    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    7    0  375    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0   16    0    0    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    1    0   14    0    0  763    0    0
     0    0    0]
 [   0    0    0    0    0    0   12    0   35    0    0 1917    0    0
     0    0    0]
 [   0    0    0    0    0    0    1    0   17    0    0  457    0    0
     0    0    0]
 [   0    0    0    0    0    0  163    0    0    0    0    0    0    0
     1    0    0]
 [   0    0    0    0    0    0   12    0    0    0    0    0    0    0
  1000    0    0]
 [   0    0    0    0    0    1  162    0    5    0    2    5    1    0
   133    0    0]
 [   0    0    0    0    0    0    0    0    0    0    0   74    0    0
     0    0    0]]---
Accuracy : 46.549%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.000
        Corn-notill: 0.000
        Corn-mintill: 0.000
        Corn: 0.000
        Grass-pasture: 0.000
        Grass-trees: 0.655
        Grass-pasture-mowed: 0.000
        Hay-windrowed: 0.755
        Oats: 0.000
        Soybean-notill: 0.000
        Soybean-mintill: 0.539
        Soybean-clean: 0.000
        Wheat: 0.000
        Woods: 0.823
        Buildings-Grass-Trees-Drives: 0.000
        Stone-Steel-Towers: 0.000
---
Kappa: 0.345
```



#### 三：

进一步提升数据集比例，并提高epochs。应该还可以提高训练epochs来提升性能。

![image-20220316105107300](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316105109.png)



```
python main.py --model hu --dataset IndianPines --training_sample 0.5 --epoch 500 --cuda 0
```

```
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
5124 samples selected (over 10249)
Running an experiment with the hu model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'hu', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.5, 'sampling_mode': 'random', 'epoch': 500, 'class_bala
ncing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n_classes': 
17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1.],
       device='cuda:0'), 'patch_size': 1, 'learning_rate': 0.01, 'batch_size': 100, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x000001FA9501
3B08>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1              [-1, 20, 178]             480
         MaxPool1d-2               [-1, 20, 35]               0
            Linear-3                  [-1, 100]          70,100
            Linear-4                   [-1, 17]           1,717
================================================================
Total params: 72,297
Trainable params: 72,297
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.03
Params size (MB): 0.28
Estimated Total Size (MB): 0.31
----------------------------------------------------------------

Inference on the image: 211it [00:00, 287.24it/s]
D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  23   0   0   0   0   0   0   0   0]
 [  0   0 427   9  10   0   0   0   2   0   3 257   6   0   0   0   0]
 [  0   0 102  98   1   0   0   0   0   0  40 151  23   0   0   0   0]
 [  0   0  74   3   7   0  12   0   5   0   0   7   9   1   0   0   0]
 [  0   0   2   4   1  32  31   0  12   0   0   1   3   0 156   0   0]
 [  0   0   2   0   7   6 347   0   1   0   0   1   0   0   0   1   0]
 [  0   0   0   0   1   0   0   0  13   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0 239   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0  10   0   0   0   0   0   0   0   0   0   0]
 [  0   0  19  11   1   0   0   0   3   0  78 372   2   0   0   0   0]
 [  0   0 150  18   3   4   9   0   6   0  42 993   3   0   0   0   0]
 [  0   0 155  14   1   0   0   0   0   0   7  89  30   0   0   1   0]
 [  0   0   0   1   0   0  18   0   0   0   0   0   0  83   0   0   0]
 [  0   0   0   0   0  50   7   0   0   0   0   0   0   3 570   3   0]
 [  0   0   0   4   1  16  86   0   0   0   0   0   2  22  55   7   0]
 [  0   0   3   0   0   0   0   0   0   0   0   3   0   0   0   0  40]]---
Accuracy : 57.580%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.000
        Corn-notill: 0.518
        Corn-mintill: 0.340
        Corn: 0.093
        Grass-pasture: 0.183
        Grass-trees: 0.784
        Grass-pasture-mowed: 0.000
        Hay-windrowed: 0.880
        Oats: 0.000
        Soybean-notill: 0.238
        Soybean-mintill: 0.640
        Soybean-clean: 0.160
        Wheat: 0.787
        Woods: 0.806
        Buildings-Grass-Trees-Drives: 0.068
        Stone-Steel-Towers: 0.930
---
Kappa: 0.501
```

#### 四：

发现震荡的幅度有点大。

![image-20220316125844947](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316125847.png)

```
python main.py --model hu --dataset IndianPines --training_sample 0.5 --epoch 1000 --cuda 0
```



#### 五：

准确率76%

![image-20220316131650963](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316131652.png)

```
python main.py --model hu --dataset IndianPines --training_sample 0.6 --epoch 1000 --cuda 0
```



```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model hu --dataset IndianPines --training_sample 0.6 --epoch 1000 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
6149 samples selected (over 10249)
Running an experiment with the hu model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `ar
r[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'hu', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.6, 'sampling_mode': 'random', 'epoch': 1000, 'class_bal
ancing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n_classes':
 17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
, 1.],
       device='cuda:0'), 'patch_size': 1, 'learning_rate': 0.01, 'batch_size': 100, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x000002453CEE
3648>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1              [-1, 20, 178]             480
         MaxPool1d-2               [-1, 20, 35]               0
            Linear-3                  [-1, 100]          70,100
            Linear-4                   [-1, 17]           1,717
================================================================
Total params: 72,297
Trainable params: 72,297
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.03
Params size (MB): 0.28
Estimated Total Size (MB): 0.31
----------------------------------------------------------------

Inference on the image: 211it [00:00, 1364.98it/s]
D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   1   0   0  17   0   0   0   1   0   0   0   0]
 [  0   0 446   6   8   1   0   0   0   0  32  69   9   0   0   0   0]
 [  0   0  81 148   7   0   1   0   0   0   9  71  15   0   0   0   0]
 [  0   0  33   0  42   1   8   0   0   0   4   5   2   0   0   0   0]
 [  0   2   0   0   3 165   0   0   2   0   0   1   5   0   6   9   0]
 [  0   0   0   0   4   2 282   0   0   0   0   0   0   0   0   4   0]
 [  0   0   0   0   0   2   0   1   8   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0 191   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   2   6   0   0   0   0   0   0   0   0   0   0]
 [  0   0  20   4   0   2   1   0   0   0 241 107  14   0   0   0   0]
 [  0   0  94  16   2   3   5   0   3   0  49 802   7   0   0   1   0]
 [  0   0  28   0  12   0   0   0   0   0  12  39 144   0   0   1   1]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0  77   0   5   0]
 [  0   0   0   0   0   5   1   0   0   0   0   0   0   1 494   5   0]
 [  0   0   0   0   0   4  35   0   0   0   0   0   0   7  54  55   0]
 [  0   0   1   0   0   0   0   0   0   0   0   3   0   0   0   0  33]]---
Accuracy : 76.122%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.000
        Corn-notill: 0.700
        Corn-mintill: 0.585
        Corn: 0.486
        Grass-pasture: 0.866
        Grass-trees: 0.894
        Grass-pasture-mowed: 0.167
        Hay-windrowed: 0.927
        Oats: 0.000
        Soybean-notill: 0.655
        Soybean-mintill: 0.772
        Soybean-clean: 0.664
        Wheat: 0.922
        Woods: 0.932
        Buildings-Grass-Trees-Drives: 0.468
        Stone-Steel-Towers: 0.930
---
Kappa: 0.725
```



### hamida

#### 一：

震荡的比较厉害。

![image-20220316132053359](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316132055.png)

```
python main.py --model hamida --dataset IndianPines --training_sample 0.09 --epoch 300 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model hamida --dataset IndianPines --training_sample 0.09 --epoch 300 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
922 samples selected (over 10249)
Running an experiment with the hamida model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'hamida', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.09, 'sampling_mode': 'random', 'epoch': 300, 'cl
ass_balancing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n
_classes': 17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1
., 1., 1., 1., 1.],
       device='cuda:0'), 'patch_size': 5, 'learning_rate': 0.01, 'batch_size': 100, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x000001A30
48762C8>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1        [-1, 20, 198, 3, 3]             560
            Conv3d-2         [-1, 20, 99, 3, 3]           1,220
            Conv3d-3         [-1, 35, 99, 1, 1]          18,935
            Conv3d-4         [-1, 35, 50, 1, 1]           3,710
            Conv3d-5         [-1, 35, 50, 1, 1]           3,710
            Conv3d-6         [-1, 35, 26, 1, 1]           2,485
            Linear-7                   [-1, 17]          15,487
================================================================
Total params: 46,107
Trainable params: 46,107
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 0.47
Params size (MB): 0.18
Estimated Total Size (MB): 0.66
----------------------------------------------------------------

Inference on the image: 199it [00:00, 453.28it/s]
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0    1    1    0    0    0    0    0   40    0    0    0    0    0
     0    0    0]
 [   0    0  611   17    9    0    0    0    0    0  295  351    8    0
     0    8    0]
 [  47    0   84  225    5    0    7    0    0    0  133  250    2    0
     0    2    0]
 [   0    0   69   10  114    0    4    0    0    1    0    6    0    0
     0   12    0]
 [  14    1    7    3    3  301   22    2    0    0    0    3    1    0
    81    2    0]
 [   0    0    0    0    0    0  643    0    0    6    0    0    0    0
     3   12    0]
 [   0    0    0    0    0    3    2    7   12    0    0    0    0    0
     0    1    0]
 [   0    0    0    0    0    0    3    0  431    0    0    0    0    0
     0    1    0]
 [   0    0    0    0    0    0   10    0    0    8    0    0    0    0
     0    0    0]
 [   5    0    7    7    1    0    7    0    0    0  664  179    7    0
     0    8    0]
 [  36    0   83   14   24    4    8    0    0    0  432 1592   31    0
     0   10    0]
 [   0    0  123    4    7    0    3    0    0    1  193   69  128    0
     0   11    1]
 [   0    0    0    1    1    0    5    0    0    0    0    0    0  174
     0    6    0]
 [   0    0    0    0    1    0    2    0    0    0    0    0    0    0
  1141    7    0]
 [  45    0    0    2    5    0   71    0    0    2    0    1    0    0
    94  131    0]
 [   0    0    0    0    0    0    0    0    0    0   19    0    0    0
     0    0   66]]---
Accuracy : 66.870%
---
F1 scores :
        Undefined: 0.000
        Alfalfa: 0.045
        Corn-notill: 0.535
        Corn-mintill: 0.434
        Corn: 0.591
        Grass-pasture: 0.805
        Grass-trees: 0.886
        Grass-pasture-mowed: 0.412
        Hay-windrowed: 0.939
        Oats: 0.444
        Soybean-notill: 0.507
        Soybean-mintill: 0.680
        Soybean-clean: 0.357
        Wheat: 0.964
        Woods: 0.924
        Buildings-Grass-Trees-Drives: 0.466
        Stone-Steel-Towers: 0.868
---
Kappa: 0.620
```

#### 二：

```
python main.py --model hamida --dataset IndianPines --training_sample 0.20 --epoch 500 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model hamida --dataset IndianPines --training_sample 0.20 --epoch 500 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
2049 samples selected (over 10249)
Running an experiment with the hamida model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'hamida', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.2, 'sampling_mode': 'random', 'epoch': 500, 'cla
ss_balancing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n_
classes': 17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
, 1., 1., 1., 1.],
       device='cuda:0'), 'patch_size': 5, 'learning_rate': 0.01, 'batch_size': 100, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x000001BFE
44A6088>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1        [-1, 20, 198, 3, 3]             560
            Conv3d-2         [-1, 20, 99, 3, 3]           1,220
            Conv3d-3         [-1, 35, 99, 1, 1]          18,935
            Conv3d-4         [-1, 35, 50, 1, 1]           3,710
            Conv3d-5         [-1, 35, 50, 1, 1]           3,710
            Conv3d-6         [-1, 35, 26, 1, 1]           2,485
            Linear-7                   [-1, 17]          15,487
================================================================
Total params: 46,107
Trainable params: 46,107
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 0.47
Params size (MB): 0.18
Estimated Total Size (MB): 0.66
----------------------------------------------------------------

Inference on the image: 199it [00:00, 455.98it/s]
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0   29    0    0    0    1    0    0    7    0    0    0    0    0
     0    0    0]
 [   0    0  990   29    4    0    0    0    0    0   60   22   37    0
     0    1    0]
 [  40    0   74  500    6    1    0    0    0    0   13   10   20    0
     0    0    0]
 [   0    0   25   16  140    0    1    0    0    0    0    1    7    0
     0    0    0]
 [  11    0    1    0    0  369    0    0    0    0    0    0    1    0
     2    2    0]
 [   0    0    0    0    0    0  582    0    0    0    0    0    0    0
     0    2    0]
 [   0    0    0    0    0    0    0   15    7    0    0    0    0    0
     0    0    0]
 [   0    5    0    0    1    1    0    0  372    0    0    0    0    0
     0    3    0]
 [   0    0    0    0    0    1    0    0    0   15    0    0    0    0
     0    0    0]
 [   3    0   32    4    3    0    1    0    0    0  601   89   44    0
     0    1    0]
 [  32    0  341   35    4    1    0    0    0    0   93 1387   66    0
     1    4    0]
 [   0    0   15   13    1    1    0    0    0    0    5   19  419    0
     0    2    0]
 [   0    0    0    1    0    0    0    0    0    0    0    0    0  163
     0    0    0]
 [   0    0    0    4    0    4    0    0    0    0    0    0    0    0
   995    9    0]
 [  42    0    0    0    0    5   11    0    0    0    1    0    0    2
     7  241    0]
 [   0    0    0    0    0    0    0    0    0    0    1    0    0    0
     0    0   73]]---
Accuracy : 84.037%
---
F1 scores :
        Undefined: 0.000
        Alfalfa: 0.817
        Corn-notill: 0.755
        Corn-mintill: 0.790
        Corn: 0.802
        Grass-pasture: 0.958
        Grass-trees: 0.987
        Grass-pasture-mowed: 0.811
        Hay-windrowed: 0.969
        Oats: 0.968
        Soybean-notill: 0.774
        Soybean-mintill: 0.794
        Soybean-clean: 0.784
        Wheat: 0.991
        Woods: 0.987
        Buildings-Grass-Trees-Drives: 0.840
        Stone-Steel-Towers: 0.993
---
Kappa: 0.819
```

#### 三：

Accuracy : 92.797% 准确率已经很高了。继续提升参数，然后看看有没有性能提升。

![image-20220316150207191](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316150209.png)

```
python main.py --model hamida --dataset IndianPines --training_sample 0.4 --epoch 1000 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model hamida --dataset IndianPines --training_sample 0.4 --epoch 1000 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
4099 samples selected (over 10249)
Running an experiment with the hamida model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'hamida', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.4, 'sampling_mode': 'random', 'epoch': 1000, 'cl
ass_balancing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n
_classes': 17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1
., 1., 1., 1., 1.],
       device='cuda:0'), 'patch_size': 5, 'learning_rate': 0.01, 'batch_size': 100, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x000001DFF
F853A88>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1        [-1, 20, 198, 3, 3]             560
            Conv3d-2         [-1, 20, 99, 3, 3]           1,220
            Conv3d-3         [-1, 35, 99, 1, 1]          18,935
            Conv3d-4         [-1, 35, 50, 1, 1]           3,710
            Conv3d-5         [-1, 35, 50, 1, 1]           3,710
            Conv3d-6         [-1, 35, 26, 1, 1]           2,485
            Linear-7                   [-1, 17]          15,487
================================================================
Total params: 46,107
Trainable params: 46,107
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 0.47
Params size (MB): 0.18
Estimated Total Size (MB): 0.66
----------------------------------------------------------------

Inference on the image: 199it [00:00, 448.24it/s]
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0   19    0    0    0    0    0    0    7    0    0    0    1    0
     0    0    0]
 [   0    0  785   21   12    0    0    0    0    0   12   22    4    0
     1    0    0]
 [  33    0    8  428    3    0    0    0    0    0    2   15    9    0
     0    0    0]
 [   0    0    5    3  131    0    0    0    0    0    0    3    0    0
     0    0    0]
 [   7    0    0    0    0  270    2    0    0    0    0    5    4    0
     1    1    0]
 [   0    0    0    0    0    0  437    0    0    0    0    0    0    0
     0    1    0]
 [   0    0    0    0    0    0    0   16    0    0    0    0    0    0
     0    1    0]
 [   0    0    0    0    0    0    0    0  287    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    1    0    0    0   11    0    0    0    0
     0    0    0]
 [   3    0    5    0    0    1    0    0    0    0  525   38   10    0
     0    1    0]
 [  23    0   31   12    2    0    0    0    0    0   27 1363   15    0
     0    0    0]
 [   0    0    9    2    0    0    0    0    0    0    1   11  333    0
     0    0    0]
 [   0    0    0    0    0    0    0    0    0    0    0    0    0  123
     0    0    0]
 [   0    0    0    0    0    0    1    0    0    0    0    0    0    0
   748   10    0]
 [  36    0    0    0    0    1    0    0    0    1    0    0    0    5
     9  180    0]
 [   0    0    1    0    0    0    0    0    0    0    1    0    3    0
     0    0   51]]---
Accuracy : 92.797%
---
F1 scores :
        Undefined: 0.000
        Alfalfa: 0.826
        Corn-notill: 0.923
        Corn-mintill: 0.888
        Corn: 0.903
        Grass-pasture: 0.959
        Grass-trees: 0.995
        Grass-pasture-mowed: 0.970
        Hay-windrowed: 0.988
        Oats: 0.917
        Soybean-notill: 0.912
        Soybean-mintill: 0.930
        Soybean-clean: 0.906
        Wheat: 0.980
        Woods: 0.986
        Buildings-Grass-Trees-Drives: 0.845
        Stone-Steel-Towers: 0.953
---
Kappa: 0.918

```



五：



准确率  95.259% visdom里面有很多

```
python main.py --model hamida --dataset IndianPines --training_sample 0.5 --epoch 1000 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model hamida --dataset IndianPines --training_sample 0.5 --epoch 1000 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
5124 samples selected (over 10249)
Running an experiment with the hamida model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'hamida', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.5, 'sampling_mode': 'random', 'epoch': 1000, 'cl
ass_balancing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n
_classes': 17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1
., 1., 1., 1., 1.],
       device='cuda:0'), 'patch_size': 5, 'learning_rate': 0.01, 'batch_size': 100, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x00000263C
54D2F48>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1        [-1, 20, 198, 3, 3]             560
            Conv3d-2         [-1, 20, 99, 3, 3]           1,220
            Conv3d-3         [-1, 35, 99, 1, 1]          18,935
            Conv3d-4         [-1, 35, 50, 1, 1]           3,710
            Conv3d-5         [-1, 35, 50, 1, 1]           3,710
            Conv3d-6         [-1, 35, 26, 1, 1]           2,485
            Linear-7                   [-1, 17]          15,487
================================================================
Total params: 46,107
Trainable params: 46,107
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 0.47
Params size (MB): 0.18
Estimated Total Size (MB): 0.66
----------------------------------------------------------------

Inference on the image: 199it [00:00, 421.12it/s]
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0   22    0    0    0    0    0    0    1    0    0    0    0    0
     0    0    0]
 [   0    0  682    7    2    0    0    0    0    0    7   10    6    0
     0    0    0]
 [  27    0    6  366    5    0    0    0    0    0    3    5    1    2
     0    0    0]
 [   0    0    1    4  112    0    0    0    0    0    0    0    0    0
     1    0    0]
 [   5    1    0    0    1  231    0    0    0    0    0    0    0    0
     4    0    0]
 [   0    0    0    0    0    0  361    0    0    0    0    0    0    0
     0    4    0]
 [   0    0    0    0    0    0    0   14    0    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    0    0  239    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    0    0    0   10    0    0    0    0
     0    0    0]
 [   1    0    3    3    0    1    0    0    0    0  466    3    9    0
     0    0    0]
 [  21    0   13    3    0    0    0    1    0    0   17 1164    4    0
     2    3    0]
 [   0    0    0    7    0    0    0    0    0    0    1    7  281    0
     0    1    0]
 [   0    0    0    1    0    0    0    0    0    0    0    0    0  101
     0    0    0]
 [   0    0    0    0    0    0    1    0    0    0    0    0    0    0
   630    2    0]
 [  30    0    0    0    0    0    0    0    0    0    0    4    0    0
     2  157    0]
 [   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0   46]]---
Accuracy : 95.259%
---
F1 scores :
        Undefined: 0.000
        Alfalfa: 0.957
        Corn-notill: 0.961
        Corn-mintill: 0.908
        Corn: 0.941
        Grass-pasture: 0.975
        Grass-trees: 0.993
        Grass-pasture-mowed: 0.966
        Hay-windrowed: 0.998
        Oats: 1.000
        Soybean-notill: 0.951
        Soybean-mintill: 0.962
        Soybean-clean: 0.940
        Wheat: 0.985
        Woods: 0.991
        Buildings-Grass-Trees-Drives: 0.872
        Stone-Steel-Towers: 1.000
---
Kappa: 0.946
```



### lee

#### 一：

![image-20220316152546346](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316152548.png)

```
python main.py --model lee --dataset IndianPines --training_sample 0.09 --epoch 300 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model lee --dataset IndianPines --training_sample 0.09 --epoch 300 --cuda 0  
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
922 samples selected (over 10249)
Running an experiment with the lee model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'lee', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.09, 'sampling_mode': 'random', 'epoch': 300, 'class
_balancing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n_cl
asses': 17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 
1., 1., 1., 1.],
       device='cuda:0'), 'patch_size': 5, 'learning_rate': 0.001, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x0000025166C74608>, 'batch_s
ize': 100, 'supervision': 'full', 'center_pixel': False}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1         [-1, 128, 1, 5, 5]         230,528
            Conv3d-2         [-1, 128, 1, 5, 5]          25,728
 LocalResponseNorm-3            [-1, 256, 5, 5]               0
            Conv2d-4            [-1, 128, 5, 5]          32,896
 LocalResponseNorm-5            [-1, 128, 5, 5]               0
            Conv2d-6            [-1, 128, 5, 5]          16,512
            Conv2d-7            [-1, 128, 5, 5]          16,512
            Conv2d-8            [-1, 128, 5, 5]          16,512
            Conv2d-9            [-1, 128, 5, 5]          16,512
           Conv2d-10            [-1, 128, 5, 5]          16,512
          Dropout-11            [-1, 128, 5, 5]               0
           Conv2d-12            [-1, 128, 5, 5]          16,512
          Dropout-13            [-1, 128, 5, 5]               0
           Conv2d-14             [-1, 17, 5, 5]           2,193
================================================================
Total params: 390,417
Trainable params: 390,417
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 0.35
Params size (MB): 1.49
Estimated Total Size (MB): 1.85
----------------------------------------------------------------


Inference on the image: 199it [00:01, 136.99it/s]
D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0    0    1    0    0    3    0    0   38    0    0    0    0    0
     0    0    0]
 [   0    0  355  338    7    2    2    0    3    0   63  469   59    0
     0    1    0]
 [   0    0   65  513   15    1    0    0    0    0    0  129   32    0
     0    0    0]
 [   0    0   77    5   68    7   17    0    6    0    0   34    2    0
     0    0    0]
 [   0    0    0    0   12   94   25    0    9    0    0    4    3    0
   288    5    0]
 [   0    0    0    0    2   32  558    0    0    0    0    0    0    0
     3   69    0]
 [   0    0    0    0   15    5    0    0    5    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    3    0    0  432    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0   11    0    0    0    0    0    0    0
     0    7    0]
 [   0    0    3   59   10    2    1    0    2    0  269  427  112    0
     0    0    0]
 [   0    0   55  465   14    9   10    0    4    0   44 1604   29    0
     0    0    0]
 [   0    0   42  151   34    1    1    0    0    0    5  169  133    0
     0    3    1]
 [   0    0    0    0    1    0    0    0    0    0    0    0    0   93
     0   93    0]
 [   0    0    0    0    0    0    0    0    0    0    0    0    0    1
  1136   14    0]
 [   0    0    0    0    1    0   41    0    0    0    0    0    0   28
   109  172    0]
 [   0    0    2    0    1    0    0    0    0    0    0    0    2    0
     0    1   79]]---
Accuracy : 59.033%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.000
        Corn-notill: 0.374
        Corn-mintill: 0.449
        Corn: 0.343
        Grass-pasture: 0.314
        Grass-trees: 0.839
        Grass-pasture-mowed: 0.000
        Hay-windrowed: 0.925
        Oats: 0.000
        Soybean-notill: 0.425
        Soybean-mintill: 0.633
        Soybean-clean: 0.292
        Wheat: 0.602
        Woods: 0.846
        Buildings-Grass-Trees-Drives: 0.480
        Stone-Steel-Towers: 0.958
---
Kappa: 0.528
```



#### 二：

Accuracy : 78.780%  相比上面的就增长很多了。

![image-20220316154127515](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316154130.png)

```
python main.py --model lee --dataset IndianPines --training_sample 0.2 --epoch 400 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model lee --dataset IndianPines --training_sample 0.2 --epoch 400 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
2049 samples selected (over 10249)
Running an experiment with the lee model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'lee', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.2, 'sampling_mode': 'random', 'epoch': 400, 'class_
balancing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n_cla
sses': 17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1
., 1., 1., 1.],
       device='cuda:0'), 'patch_size': 5, 'learning_rate': 0.001, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x000001C5046D6088>, 'batch_s
ize': 100, 'supervision': 'full', 'center_pixel': False}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1         [-1, 128, 1, 5, 5]         230,528
            Conv3d-2         [-1, 128, 1, 5, 5]          25,728
 LocalResponseNorm-3            [-1, 256, 5, 5]               0
            Conv2d-4            [-1, 128, 5, 5]          32,896
 LocalResponseNorm-5            [-1, 128, 5, 5]               0
            Conv2d-6            [-1, 128, 5, 5]          16,512
            Conv2d-7            [-1, 128, 5, 5]          16,512
            Conv2d-8            [-1, 128, 5, 5]          16,512
            Conv2d-9            [-1, 128, 5, 5]          16,512
           Conv2d-10            [-1, 128, 5, 5]          16,512
          Dropout-11            [-1, 128, 5, 5]               0
           Conv2d-12            [-1, 128, 5, 5]          16,512
          Dropout-13            [-1, 128, 5, 5]               0
           Conv2d-14             [-1, 17, 5, 5]           2,193
================================================================
Total params: 390,417
Trainable params: 390,417
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 0.35
Params size (MB): 1.49
Estimated Total Size (MB): 1.85
----------------------------------------------------------------


Inference on the image: 199it [00:01, 134.68it/s]
D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0   15    1    0    0    0    0    0   21    0    0    0    0    0
     0    0    0]
 [   0    0  997    1    2    0    0    0    0    0   45   62   32    0
     0    4    0]
 [   0    0  131  303    4    0    0    0    0    0    7  136   82    1
     0    0    0]
 [   0    0   68    7   95    0    0    1    0    0    2    3   14    0
     0    0    0]
 [   0    0    0    0    6  336    6    2    0    0    0    1    0    0
    18   17    0]
 [   0    0    0    0    8    0  567    0    0    0    0    0    0    0
     1    8    0]
 [   0    0    6    0    0    5    0    8    3    0    0    0    0    0
     0    0    0]
 [   0    3    0    0    0    2    0    0  377    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    5    0    0    5    0    0    0    0
     0    6    0]
 [   0    0   87    6    6    2    1    0    0    0  596   69   11    0
     0    0    0]
 [   0    0  253    7    2    5    0    0    0    0  151 1533   12    0
     0    1    0]
 [   0    0  151    4    4    1    1    0    0    0   28   25  257    1
     0    2    1]
 [   0    0    0    0    1    0    0    0    0    0    0    0    0  162
     0    1    0]
 [   0    0    0    0    0    7    0    0    0    0    0    0    0    3
   977   25    0]
 [   0    0    0    0    0    1   11    0    0   13    0    0    1   38
    87  158    0]
 [   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0   74]]---
Accuracy : 78.780%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.545
        Corn-notill: 0.703
        Corn-mintill: 0.611
        Corn: 0.597
        Grass-pasture: 0.902
        Grass-trees: 0.965
        Grass-pasture-mowed: 0.485
        Hay-windrowed: 0.963
        Oats: 0.294
        Soybean-notill: 0.742
        Soybean-mintill: 0.808
        Soybean-clean: 0.581
        Wheat: 0.878
        Woods: 0.933
        Buildings-Grass-Trees-Drives: 0.595
        Stone-Steel-Towers: 0.993
---
Kappa: 0.757
```



#### 三：

Accuracy : 93.675% 性能已经很好了。看曲线的走势，感觉再加epochs已经不能增加准确率了。

![image-20220316160232186](/Users/roczhang/Library/Application Support/typora-user-images/image-20220316160232186.png)

```
python main.py --model lee --dataset IndianPines --training_sample 0.4 --epoch 1000 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model lee --dataset IndianPines --training_sample 0.4 --epoch 1000 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
4099 samples selected (over 10249)
Running an experiment with the lee model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'lee', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.4, 'sampling_mode': 'random', 'epoch': 1000, 'class
_balancing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n_cl
asses': 17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 
1., 1., 1., 1.],
       device='cuda:0'), 'patch_size': 5, 'learning_rate': 0.001, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x00000239D0D09E48>, 'batch_s
ize': 100, 'supervision': 'full', 'center_pixel': False}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1         [-1, 128, 1, 5, 5]         230,528
            Conv3d-2         [-1, 128, 1, 5, 5]          25,728
 LocalResponseNorm-3            [-1, 256, 5, 5]               0
            Conv2d-4            [-1, 128, 5, 5]          32,896
 LocalResponseNorm-5            [-1, 128, 5, 5]               0
            Conv2d-6            [-1, 128, 5, 5]          16,512
            Conv2d-7            [-1, 128, 5, 5]          16,512
            Conv2d-8            [-1, 128, 5, 5]          16,512
            Conv2d-9            [-1, 128, 5, 5]          16,512
           Conv2d-10            [-1, 128, 5, 5]          16,512
          Dropout-11            [-1, 128, 5, 5]               0
           Conv2d-12            [-1, 128, 5, 5]          16,512
          Dropout-13            [-1, 128, 5, 5]               0
           Conv2d-14             [-1, 17, 5, 5]           2,193
================================================================
Total params: 390,417
Trainable params: 390,417
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 0.35
Params size (MB): 1.49
Estimated Total Size (MB): 1.85
----------------------------------------------------------------

Inference on the image: 199it [00:01, 135.13it/s]
D:\RocZhang\code\DeepHyperX\utils.py:371: RuntimeWarning: invalid value encountered in double_scalars
  F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0   23    0    0    0    2    0    0    2    0    0    0    0    0
     0    0    0]
 [   0    0  761    9    4    0    0    0    0    1   24   43   15    0
     0    0    0]
 [   0    0    6  456    9    2    0    0    0    0    4   13    7    1
     0    0    0]
 [   0    0    0    8  129    0    1    0    0    0    0    0    4    0
     0    0    0]
 [   0    0    0    0    0  280    0    0    1    0    0    0    6    0
     3    0    0]
 [   0    0    0    0    0    1  433    0    0    0    0    0    0    0
     1    3    0]
 [   0    0    0    0    0    1    0   15    1    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    0    0  287    0    0    0    0    0
     0    0    0]
 [   0    0    0    0    0    0    0    0    0   11    0    1    0    0
     0    0    0]
 [   0    0   10    4    1    2    0    0    0    0  521   44    1    0
     0    0    0]
 [   0    0   15   20    1    0    0    0    0    0   23 1398   15    0
     0    1    0]
 [   0    0    7    5    1    0    0    0    0    0    7    4  330    0
     0    1    1]
 [   0    0    0    0    0    0    0    0    0    0    0    0    0  123
     0    0    0]
 [   0    0    0    0    0    0    1    0    0    0    0    0    0    1
   749    8    0]
 [   0    0    0    0    0    2   16    0    0    1    0    0    0    0
    24  189    0]
 [   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0   56]]---
Accuracy : 93.675%
---
F1 scores :
        Undefined: nan
        Alfalfa: 0.920
        Corn-notill: 0.919
        Corn-mintill: 0.912
        Corn: 0.899
        Grass-pasture: 0.966
        Grass-trees: 0.974
        Grass-pasture-mowed: 0.938
        Hay-windrowed: 0.993
        Oats: 0.880
        Soybean-notill: 0.897
        Soybean-mintill: 0.940
        Soybean-clean: 0.899
        Wheat: 0.992
        Woods: 0.975
        Buildings-Grass-Trees-Drives: 0.871
        Stone-Steel-Towers: 0.991
---
Kappa: 0.928
```



### chen

#### 一：

感觉训练样本有点少，准确率震荡幅度大。

![image-20220316163958754](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316164000.png)

```
python main.py --model chen --dataset IndianPines --training_sample 0.09 --epoch 300 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model chen --dataset IndianPines --training_sample 0.09 --epoch 300 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
922 samples selected (over 10249)
Running an experiment with the chen model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'chen', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.09, 'sampling_mode': 'random', 'epoch': 300, 'clas
s_balancing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n_c
lasses': 17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1.],
       device='cuda:0'), 'patch_size': 27, 'learning_rate': 0.003, 'batch_size': 100, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x0000027
3FA9FEB48>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1      [-1, 32, 169, 24, 24]          16,416
         MaxPool3d-2      [-1, 32, 169, 12, 12]               0
           Dropout-3      [-1, 32, 169, 12, 12]               0
            Conv3d-4        [-1, 32, 138, 9, 9]         524,320
         MaxPool3d-5        [-1, 32, 138, 4, 4]               0
           Dropout-6        [-1, 32, 138, 4, 4]               0
            Conv3d-7        [-1, 32, 107, 1, 1]         524,320
           Dropout-8        [-1, 32, 107, 1, 1]               0
            Linear-9                   [-1, 17]          58,225
================================================================
Total params: 1,123,281
Trainable params: 1,123,281
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.56
Forward/backward pass size (MB): 39.51
Params size (MB): 4.28
Estimated Total Size (MB): 44.35
----------------------------------------------------------------

Inference on the image: 142it [00:32,  4.39it/s]
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0    2    2    0    0   38    0    0    0    0    0    0    0    0
     0    0    0]
 [  74    0  681  111    0   15    0    0    0    0  231  187    0    0
     0    0    0]
 [ 413    0   40  213    0    0    0    0    0    0   48   38    0    3
     0    0    0]
 [ 122    0   64    1    2    0    3    0    0    0   24    0    0    0
     0    0    0]
 [ 180    0    0    0    0  199   12    0    0    0    0    3    0    2
    44    0    0]
 [   0    0    0    0    0  176  453    0    0    0    0    4    0    0
    31    0    0]
 [   0    0    0    0    0   25    0    0    0    0    0    0    0    0
     0    0    0]
 [ 176    0    0    0    0  103    0    0  156    0    0    0    0    0
     0    0    0]
 [   0    0    0    4    0    1    7    0    0    0    5    1    0    0
     0    0    0]
 [ 138    0   34   16    0   24    8    0    0    0  620   45    0    0
     0    0    0]
 [ 255    0  444   58    0   65   31    1    1    0  279 1099    0    0
     0    0    1]
 [ 162    0  158   48    0    4    2    0    0    0   39   23   87   17
     0    0    0]
 [   0    0    0    0    4   23   17    0    0    0    9    0    0  126
     8    0    0]
 [ 293    0    0    0    0   82    0    0    0    0    0    0    0    0
   776    0    0]
 [ 258    0    0    0    0   54   34    0    0    0    3    0    0    2
     0    0    0]
 [   0    0   16    4    0    0    0    0    0    0   47   15    3    0
     0    0    0]]---
Accuracy : 47.325%
---
F1 scores :
        Undefined: 0.000
        Alfalfa: 0.091
        Corn-notill: 0.497
        Corn-mintill: 0.352
        Corn: 0.018
        Grass-pasture: 0.319
        Grass-trees: 0.736
        Grass-pasture-mowed: 0.000
        Hay-windrowed: 0.527
        Oats: 0.000
        Soybean-notill: 0.566
        Soybean-mintill: 0.602
        Soybean-clean: 0.276
        Wheat: 0.748
        Woods: 0.772
        Buildings-Grass-Trees-Drives: 0.000
        Stone-Steel-Towers: 0.000
---
Kappa: 0.417
```

#### 二：

Accuracy : 72.697% 感觉虽然训练的很慢，但是效果并不好。但是这个损失曲线和准确率提升的很稳定。

![image-20220316204338666](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316204341.png)

```
python main.py --model chen --dataset IndianPines --training_sample 0.3 --epoch 400 --cuda 0
```

```
(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\DeepHyperX>python main.py --model chen --dataset IndianPines --training_sample 0.3 --epoch 400 --cuda 0
Computation on CUDA GPU device 0
Setting up a new session...
Image has dimensions 145x145 and 200 channels
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
3074 samples selected (over 10249)
Running an experiment with the chen model run 1/1
D:\RocZhang\code\DeepHyperX\utils.py:465: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  train_gt[train_indices] = gt[train_indices]
D:\RocZhang\code\DeepHyperX\utils.py:466: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of 
`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  test_gt[test_indices] = gt[test_indices]
{'dataset': 'IndianPines', 'model': 'chen', 'folder': './Datasets/', 'cuda': 0, 'runs': 1, 'training_sample': 0.3, 'sampling_mode': 'random', 'epoch': 400, 'class
_balancing': False, 'test_stride': 1, 'flip_augmentation': False, 'radiation_augmentation': False, 'mixture_augmentation': False, 'with_exploration': False, 'n_cl
asses': 17, 'n_bands': 200, 'ignored_labels': [0], 'device': device(type='cuda', index=0), 'weights': tensor([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 
1., 1., 1., 1.],
       device='cuda:0'), 'patch_size': 27, 'learning_rate': 0.003, 'batch_size': 100, 'scheduler': <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x000001B
3A63C29C8>, 'supervision': 'full', 'center_pixel': True}
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1      [-1, 32, 169, 24, 24]          16,416
         MaxPool3d-2      [-1, 32, 169, 12, 12]               0
           Dropout-3      [-1, 32, 169, 12, 12]               0
            Conv3d-4        [-1, 32, 138, 9, 9]         524,320
         MaxPool3d-5        [-1, 32, 138, 4, 4]               0
           Dropout-6        [-1, 32, 138, 4, 4]               0
            Conv3d-7        [-1, 32, 107, 1, 1]         524,320
           Dropout-8        [-1, 32, 107, 1, 1]               0
            Linear-9                   [-1, 17]          58,225
================================================================
Total params: 1,123,281
Trainable params: 1,123,281
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.56
Forward/backward pass size (MB): 39.51
Params size (MB): 4.28
Estimated Total Size (MB): 44.35
----------------------------------------------------------------

Inference on the image: 142it [00:41,  3.43it/s]
Confusion matrix :
[[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [   0   27    0    0    0    0    0    0    0    0    0    5    0    0
     0    0    0]
 [  57    0  728    3    0    0    0    0    0    0   25  184    3    0
     0    0    0]
 [ 317    0    0  240    0    1    0    0    0    0   10   13    0    0
     0    0    0]
 [  92    0    0    1   58    0    7    0    0    0    5    3    0    0
     0    0    0]
 [ 142    0    0    0    0  177    2    6    0    0    2    0    0    0
     9    0    0]
 [   0    0    0    0    0    0  505    0    0    0    0    4    0    0
     2    0    0]
 [   0    0    0    0    0    0    0   20    0    0    0    0    0    0
     0    0    0]
 [ 138    0    0    0    0    0    7    0  186    0    0    0    0    0
     4    0    0]
 [   0    0    0    0    0    0    5    0    0    9    0    0    0    0
     0    0    0]
 [ 104    0    0    0    0    0    0    0    0    0  573    0    3    0
     0    0    0]
 [ 184    0    0    0    0    0    1    0    0    0    3 1531    0    0
     0    0    0]
 [ 120    0   10    0    1    1    0    0    0    0   16    5  259    0
     0    0    3]
 [   0    0    0    0    0    2    3    0    0    0    0    0    0  138
     0    0    0]
 [ 220    0    0    0    0    0    0    0    0    0    0    4    0    0
   662    0    0]
 [ 198    0    0    0    0    0    1    0    0    1    2    1    0    0
     6   61    0]
 [   0    0    0    0    0    0    0    0    0    0    9    7    7    0
     0    0   42]]---
Accuracy : 72.697%
---
F1 scores :
        Undefined: 0.000
        Alfalfa: 0.915
        Corn-notill: 0.838
        Corn-mintill: 0.582
        Corn: 0.516
        Grass-pasture: 0.682
        Grass-trees: 0.969
        Grass-pasture-mowed: 0.870
        Hay-windrowed: 0.714
        Oats: 0.750
        Soybean-notill: 0.865
        Soybean-mintill: 0.881
        Soybean-clean: 0.754
        Wheat: 0.982
        Woods: 0.844
        Buildings-Grass-Trees-Drives: 0.369
        Stone-Steel-Towers: 0.764
---
Kappa: 0.694

```



#### 三：



这个跑了几个小时，terminal连接断开了。

```
python main.py --model chen --dataset IndianPines --training_sample 0.6 --epoch 500 --cuda 0
```



由于前面的一个模型需要运行很多次来比较准确率，但是呢visdom画的图都在一个页面，导致页面及其的混乱。所以我在画图时加了一个时间戳，在保存文件时将每次运行的visdom画图都分离开。

### li

```
Network :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1        [-1, 16, 196, 3, 3]           1,024
            Conv3d-2        [-1, 32, 196, 1, 1]          13,856
            Linear-3                   [-1, 17]         106,641
================================================================
Total params: 121,521
Trainable params: 121,521
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 0.26
Params size (MB): 0.46
Estimated Total Size (MB): 0.75
```

#### 一：

```
python main.py --model li --dataset IndianPines --training_sample 0.4 --epoch 500 --cuda 0
```

结果：保存在：IndianPines_li_1647486666

#### 二：

```
python main.py --model li --dataset IndianPines --training_sample 0.5 --epoch 600 --cuda 0
```

结果保存在：IndianPines_li_1647487068



感觉准确率还蛮高的：Accuracy : 93.385%



### he

#### 一：

这个损失曲线、准确率曲线很理想啊。

![image-20220317121412240](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220317121414.png)

```
python main.py --model he --dataset IndianPines --training_sample 0.3 --epoch 600 --cuda 0
```

IndianPines_he_1647487861

### luo

#### 一：

```
python main.py --model luo --dataset IndianPines --training_sample 0.4 --epoch 1000 --cuda 0
```

IndianPines_luo_1647566577  

#### 二：

```
python main.py --model luo --dataset IndianPines --training_sample 0.6 --epoch 2000 --cuda 0
```

IndianPines_luo_1647567970



### sharma

#### 一：

```
python main.py --model sharma --dataset IndianPines --training_sample 0.2 --epoch 300 --cuda 0
```

这个命令搞错了，重新运行了。

IndianPines_sharma_1647569408    Accuracy : 42.390%

#### 二：

```
python main.py --model sharma --dataset IndianPines --training_sample 0.4 --epoch 500 --cuda 0
```

IndianPines_sharma_1647569204   Accuracy : 43.106%

#### 三

```
python main.py --model sharma --dataset IndianPines --training_sample 0.5 --epoch 1000 --cuda 0
```

IndianPines_sharma_1647569242  42.712%



### liu

一：

```
python main.py --model liu --dataset IndianPines --training_sample 0.2 --epoch 300 --cuda 0
```



二：

```
python main.py --model liu --dataset IndianPines --training_sample 0.4 --epoch 500 --cuda 0
```

 

三：

```
python main.py --model liu --dataset IndianPines --training_sample 0.6 --epoch 1000 --cuda 0
```





### boulch

一：

Accuracy : 71.110%  网络比较大，训练的比较慢

![image-20220318162005934](https://cdn.jsdelivr.net/gh/dlagez/img@master/20220318162008.png)

```
python main.py --model boulch --dataset IndianPines --training_sample 0.2 --epoch 300 --cuda 0
```

Accuracy : 71.110%
