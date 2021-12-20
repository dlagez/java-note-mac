



# 方法：

首先需要将数据进行归一化。

**L2范数归一化方法**



使用多种方法进行对比：朴素贝叶斯、最近邻、决策树、随机森林、支持向量机、神经网络等，最后选择神经网络（SVM）

scikit的多层感知机的使用。

https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor

多层感知机的wiki

https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E5%99%A8

使用多层感知机和支持向量机对比。

MLP在80年代的时候曾是相当流行的机器学习方法，拥有广泛的应用场景，譬如语音识别、图像识别、机器翻译等等，但自90年代以来，MLP遇到来自更为简单的支持向量机的强劲竞争。近来，由于[深度学习](https://zh.wikipedia.org/wiki/深度学习)的成功，MLP又重新得到了关注。



人工神经网络领域通常被称为神经网络或多层感知器，这可能是最有用的神经网络类型。感知器是一种单神经元模型，是大型神经网络的前身。这是一个研究如何利用简单的生物大脑模型来解决困难的计算任务的领域，比如我们在机器学习中看到的预测建模任务。我们的目标不是创建真实的大脑模型，而是开发鲁棒的算法和数据结构，我们可以使用这些算法和数据结构来模拟困难的问题。

神经网络的力量来自于它们学习训练数据表示的能力，以及如何最好地将其与您想要预测的输出变量联系起来。从这个意义上讲，神经网络学习映射。在数学上，它们能够学习任何映射函数，并且已被证明是一种通用的近似算法。神经网络的预测能力来自网络的分层或多层结构。数据结构可以在不同的尺度或分辨率下挑出（学会表示）功能，并将它们组合成高阶功能。例如从行，到形状的线条集合。



## 评估方法一

http://www.gabormelli.com/RKB/sklearn.neural_network.MLPRegressor

https://github.com/omoreira/GM-Python-Workbook/blob/master/NN_examples/MLP_regression_10foldcv_boston.py

使用这个多层感知机算法，画出来的图是这样的。横轴是真实值，纵轴是预测值，如果拟合好的话应该是一个45度角的倾斜直线。

画出来三个激活函数的对比图



## 评估方法二

使用 mlp decisionTree 还要选出几个方法来做对比

https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_partial_dependence_visualization_api.html#sphx-glr-auto-examples-miscellaneous-plot-partial-dependence-visualization-api-py

画局部依赖图，使用mlpreg和决策树对比。

部分依赖的解释：https://www.jianshu.com/p/d04f41c79c17



## 评估方法三

https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html#sphx-glr-auto-examples-inspection-plot-partial-dependence-py

看看这个



论文：基于核偏最小二乘特征提取的垃圾邮件过滤方法的研究 https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201301&filename=1012039968.nh&uniplatform=NZKPT&v=AJjnqXKkApFr6vL86MWPCjHP9csBNiamoqToCP6GEXX3SJ6c4K5jqkqOgYai5ha4







参考：

波士顿房价预测：

- https://www.zhihu.com/question/39792141
- https://www.cnblogs.com/wish-together/p/14764148.html

癌症预测：

Prediction of breast cancer malignancy using an artificial neural network

- https://acsjournals.onlinelibrary.wiley.com/doi/abs/10.1002/1097-0142(19941201)74:11%3C2944::AID-CNCR2820741109%3E3.0.CO;2-F

我觉得查论文的方法是有问题的，应该查分子描述对蛋白质结构的预测。而不是癌症的预测，癌症的预测太大了，有用x光预测等等，不是用的分子描述符号。

搜索关键词：Molecular descriptor nureal network prediction

物理性质预测：

- https://www.sciencedirect.com/science/article/abs/pii/S0169409X03001170



预测ic50

https://pubs.acs.org/doi/abs/10.1021/ci990125r
