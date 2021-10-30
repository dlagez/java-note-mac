GAN是一个生成模型，它的目的是将一个随机高斯分布或者其他分布的噪声向量通过一个生成网络得到一个和真的数据分布差不多的生成分布。训练的时候，怎么衡量这个生成分布和目标分布差不多呢？它是通过构造一个叫判别器的分类神经网络来衡量。

数学原理来说，他的目标函数是一个极大极小问题。

- 对于生成数据，需要它越逼真越好，所以训练生成器时，判别器接收后极大它的概率输出；但是在训练判别器时，判别器接收这个假样本后需要极小它的概率输出。
- 对于真实数据，训练判别器时，为了保持它有意义，极大化它的概率输出。


