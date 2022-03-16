### gan

训练记录：

由于默认的训练batch_size为64，所以我想测试一下batch_size为128的情况下的训练效果。

```bash
python gan.py --n_epochs 2000 --batch_size 128 --lr 0.00002

(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\PyTorch-GAN\implementations\gan>python gan.py --n_epochs 2000 --batch_size 128 --lr 0.00002
Namespace(b1=0.5, b2=0.999, batch_size=128, channels=1, img_size=28, latent_dim=100, lr=2e-05, n_cpu=8, n_epochs=2000, sample_interval=400)
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../../data/mnist\MNIST\raw\train-images-idx3-ubyte.gz
 60%|████████████████████████████████████████████████████████████████████████████████▏                                                    | 5971968/9912422 [00:03<00:01, 2299014.99it/s]E
xtracting ../../data/mnist\MNIST\raw\train-images-idx3-ubyte.gz to ../../data/mnist\MNIST\raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ../../data/mnist\MNIST\raw\train-labels-idx1-ubyte.gz
                                                                                                                                                                                         E
xtracting ../../data/mnist\MNIST\raw\train-labels-idx1-ubyte.gz to ../../data/mnist\MNIST\raw                                                                  | 0/28881 [00:00<?, ?it/s] 
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ../../data/mnist\MNIST\raw\t10k-images-idx3-ubyte.gz
                                                                                                                                                                                         E
xtracting ../../data/mnist\MNIST\raw\t10k-images-idx3-ubyte.gz to ../../data/mnist\MNIST\raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../../data/mnist\MNIST\raw\t10k-labels-idx1-ubyte.gz            | 1253376/1648877 [00:01<00:00, 1097264.43it/s]
                                                                                                                                                                                         E
xtracting ../../data/mnist\MNIST\raw\t10k-labels-idx1-ubyte.gz to ../../data/mnist\MNIST\raw
Processing...
D:\anaconda3\envs\detectron2\lib\site-packages\torchvision\datasets\mnist.py:480: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors.
 This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before conver
ting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ..\torch\csrc\utils\tensor_numpy.cpp:141.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
Done!
32768it [00:03, 10294.39it/s]
1654784it [00:02, 614514.60it/s]
8192it [00:00, 12810.17it/s]
[Epoch 0/2000] [Batch 0/469] [D loss: 0.672173] [G loss: 0.696674]
9920512it [00:12, 818786.25it/s]
[Epoch 0/2000] [Batch 1/469] [D loss: 0.659687] [G loss: 0.696357]
[Epoch 0/2000] [Batch 2/469] [D loss: 0.650375] [G loss: 0.696014]
[Epoch 0/2000] [Batch 3/469] [D loss: 0.641060] [G loss: 0.695671]
[Epoch 0/2000] [Batch 4/469] [D loss: 0.635681] [G loss: 0.695405]
[Epoch 0/2000] [Batch 5/469] [D loss: 0.626408] [G loss: 0.695082]
[Epoch 0/2000] [Batch 6/469] [D loss: 0.615033] [G loss: 0.694762]
[Epoch 0/2000] [Batch 7/469] [D loss: 0.609363] [G loss: 0.694478]
[Epoch 0/2000] [Batch 8/469] [D loss: 0.599206] [G loss: 0.694221]
[Epoch 0/2000] [Batch 9/469] [D loss: 0.591835] [G loss: 0.693918]
[Epoch 0/2000] [Batch 10/469] [D loss: 0.584488] [G loss: 0.693610]
[Epoch 0
```

这是训练了147个epoch之后的图像，感觉学习率低了。

<img src="https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220315183516909.png" alt="image-20220315183516909" style="zoom: 33%;" />

```bash
python gan.py --n_epochs 2000 --batch_size 128 --lr 0.0004

(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\PyTorch-GAN\implementations\gan>python gan.py --n_epochs 2000 --batch_size 128 -
-lr 0.0004
Namespace(b1=0.5, b2=0.999, batch_size=128, channels=1, img_size=28, latent_dim=100, lr=0.0004, n_cpu=8, n_epochs=2000, sample_interva
l=400)
[Epoch 0/2000] [Batch 0/469] [D loss: 0.711943] [G loss: 0.694535]
[Epoch 0/2000] [Batch 1/469] [D loss: 0.550376] [G loss: 0.687245]
[Epoch 0/2000] [Batch 2/469] [D loss: 0.457509] [G loss: 0.680358]
[Epoch 0/2000] [Batch 3/469] [D loss: 0.404701] [G loss: 0.670983]
[Epoch 0/2000] [Batch 4/469] [D loss: 0.384761] [G loss: 0.657429]
[Epoch 0/2000] [Batch 5/469] [D loss: 0.387132] [G loss: 0.637093]
[Epoch 0/2000] [Batch 6/469] [D loss: 0.396995] [G loss: 0.612564]
[Epoch 0/2000] [Batch 7/469] [D loss: 0.411593] [G loss: 0.586392]
[Epoch 0/2000] [Batch 8/469] [D loss: 0.425523] [G loss: 0.566853]
[Epoch 0/2000] [Batch 9/469] [D loss: 0.435751] [G loss: 0.554157]
[Epoch 0/2000] [Batch 10/469] [D loss: 0.434952] [G loss: 0.564760]
[Epoch 0/2000] [Batch 11/469] [D loss: 0.427122] [G loss: 0.596546]
[Epoch
```

训练了228个epoch，感觉效果还不错。在这228epoch之前的图像其实也还可以，就是有个别的优点模糊。

<img src="/Users/roczhang/Library/Application Support/typora-user-images/image-20220315195926751.png" alt="image-20220315195926751" style="zoom: 50%;" />



### dcgan

```bash
python dcgan.py

(detectron2) loongtr@DESKTOP-GCHQHDA D:\RocZhang\code\PyTorch-GAN\implementations\dcgan>python dcgan.py
Namespace(b1=0.5, b2=0.999, batch_size=64, channels=1, img_size=32, latent_dim=100, lr=0.0002, n_cpu=8, n_epochs=200, sample_interval=400)    
[Epoch 0/200] [Batch 0/938] [D loss: 0.693144] [G loss: 0.707567]
[Epoch 0/200] [Batch 1/938] [D loss: 0.693303] [G loss: 0.706935]
[Epoch 0/200] [Batch 2/938] [D loss: 0.693199] [G loss: 0.706265]
[Epoch 0/200] [Batch 3/938] [D loss: 0.693204] [G loss: 0.705718]
[Epoch 0/200] [Batch 4/938] [D loss: 0.693156] [G loss: 0.705061]
[Epoch 0/200] [Batch 5/938] [D loss: 0.693084] [G loss: 0.704434]
[Epoch 0/200] [Batch 6/938] [D loss: 0.692994] [G loss: 0.704146]
[Epoch 0/200] [Batch 7/938] [D loss: 0.692968] [G loss: 0.703520]
[Epoch 0/200] [Batch 8/938] [D loss: 0.692864] [G loss: 0.702865]
[Epoch 0/200] [Batch 9/938] [D loss: 0.692878] [G loss: 0.702395]
[Epoch 0/200] [Batch 10/938] [D loss: 0.692710] [G loss: 0.701864]
[Epoch 0/200] [Batch 11/938] [D loss: 0.692686] [G loss: 0.701409]
```

先使用默认的参数跑看看，跑了200epochs，相比原始gan，在两个网络运行差不多epochs的情况下，整体图片上的噪声点少了。生成的手写字也相差不大，甚至原始的gan生成的质量还要稍微好一点。可能是个别的情况

<img src="https://cdn.jsdelivr.net/gh/dlagez/img@master/20220316085610.png" alt="image-20220316085537708" style="zoom:50%;" />