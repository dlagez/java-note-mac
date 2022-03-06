看了这个代码实现，感觉实现的很好，但是代码优点杂乱，不方便去解读，但是按照readme的介绍，把它跑了一遍。

它的数据下载是没有问题的，执行sh文件即可下载数据集。

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix



后来看了一个仓库：https://github.com/eriklindernoren/PyTorch-GAN

这个仓库里面实现了很多的gan论文。其中就包括pix2pix，我看了一下实现代码，非常简洁。一共三个文件，所以我决定使用这个代码来查看pix2pix的网络结构。

![image-20220227204028494](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220227204028494.png)

这个readme中的示例代码，虽然有些简洁，但是可以运行。其中数据集的下载有问题，我是使用了其他代码下载了数据集然后将数据集拷贝过来的。下载这个数据集`facades`的方式有很多，直接百度即可。

```
$ cd data/
$ bash download_pix2pix_dataset.sh facades
$ cd ../implementations/pix2pix/
$ python3 pix2pix.py --dataset_name facades
```

执行之后会是默认的配置来进行网络的训练。训练200个epoch。

```
(yolact_env) shujixueyuan@shujixueyuan:~/8206/roc/code/PyTorch-GAN/implementations/pix2pix$ python3 pix2pix.py --dataset_name facades
Namespace(b1=0.5, b2=0.999, batch_size=1, channels=3, checkpoint_interval=-1, dataset_name='facades', decay_epoch=100, epoch=0, img_height=256, img_width=256, lr=0.0002, n_cpu=8, n_epochs=200, sample_interval=500)
[Epoch 39/200] [Batch 136/506] [D loss: 0.001436] [G loss: 29.534107, pixel: 0.285508, adv: 0.983272] ETA: 0:55:16.0611686.103595
```

