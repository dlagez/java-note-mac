我们知道深度学习中对图像处理应用最好的模型是CNN，那么如何把CNN与GAN结合？DCGAN是这方面最好的尝试之一

它只是把上述的G和D换成了两个卷积神经网络（CNN）。但不是直接换就可以了，DCGAN对卷积神经网络的结构做了一些改变，以提高样本的质量和收敛的速度，这些改变有：

- 取消所有pooling层。G网络中使用转置卷积（transposed convolutional layer）进行上采样，D网络中用加入stride的卷积代替pooling。
- 在D和G中均使用batch normalization
- 去掉FC层，使网络变为全卷积网络
- G网络中使用ReLU作为激活函数，最后一层使用tanh
- D网络中使用LeakyReLU作为激活函数