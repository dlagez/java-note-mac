# 第一问

核偏最小二乘方法：

A Kernel-Based Multivariate Feature Selection Method for Microarray Data Classification 这个摘要写的值的借鉴。

高维和小样本量及其固有的过度拟合风险，对在微阵列数据分类中构建高效分类器提出了巨大挑战。因此，应在数据分类之前进行特征选择技术以提高预测性能。一般来说，过滤方法可以被认为是主要或辅助选择机制，因为它们的简单性、可扩展性和低计算复杂度。然而，一系列简单的例子表明，过滤方法会导致不太准确的性能，因为它们忽略了特征的依赖关系。尽管很少有出版物专注于通过基于多元的方法揭示特征之间的关系，但这些方法仅通过线性方法描述特征之间的关系。而简单的线性组合关系限制了性能的提升。在本文中，我们使用核方法来发现特征之间以及特征与目标之间的内在非线性相关性。此外，正交分量的数量由内核 Fishers 线性判别分析 (FLDA) 以自适应方式确定，而不是通过手动参数设置。为了揭示我们方法的有效性，我们进行了多次实验，并比较了我们的方法与其他竞争性多变量特征选择器之间的结果。在我们的比较中，我们在两个组数据集上使用了两个分类器（支持向量机、k-最近邻），即两类和多类数据集。实验结果表明，我们的方法的性能优于其他方法，尤其是在三个硬分类数据集上，即王氏乳腺癌、戈登肺腺癌和 Pomeroy 髓母细胞瘤。

Extraction Using T-Test Statistics and Kernel Partial Least Squares

在本文中，我们提出了一种基因提取方法，它结合使用两种标准特征提取方法，即 T 检验方法和核偏最小二乘法 (KPLS)。 首先，基于 T 检验方法的预处理步骤用于过滤不相关和嘈杂的基因。 然后使用 KPLS 提取具有高信息量的特征。 最后，提取的特征被送入分类器。 实验在三个基准数据集上进行：乳腺癌、ALL/AML 白血病和结肠癌。 虽然使用 T-test 方法或 KPLS 都不能产生令人满意的结果，但实验结果表明，将这两者结合使用可以显着提高分类精度，并且这种简单的组合可以在所有方面获得最先进的性能。 三个数据集。

**Simultaneous Dimensionality Reduction and Human Age Estimation via Kernel Partial Least Squares Regression**

由于现实中的许多潜在应用，人类年龄估计最近已成为计算机视觉和模式识别领域的一个活跃研究课题。在本文中，我们建议使用核偏最小二乘 (KPLS) 回归进行年龄估计。 KPLS（或线性 PLS）方法与以前的方法相比有几个优点：（1）KPLS 可以在单个学习框架中降低特征维数并同时学习老化函数，而不是单独执行每个任务使用不同的技术； (2) KPLS可以找到少量的潜在变量，例如20个，将数千个特征投影到一个非常低维的子空间中，这可能对实时应用有很大影响； (3) KPLS 回归有一个输出向量，可以包含多个标签，因此可以一起解决几个相关问题，例如年龄估计、性别分类和种族估计。这是首次引入并应用核 PLS 方法以高精度解决计算机视觉中的回归问题。在一个非常大的数据库上的实验结果表明，KPLS 明显优于流行的 SVM 方法，并且在人类年龄估计方面优于最先进的方法。

A Kernel Partial Least Square Based Feature Selection Method

最大相关性和最小冗余（mRMR）已被公认为最好的特征选择方法之一。 本文提出了一种基于核偏最小二乘 (KPLS) 的 mRMR 方法，旨在简化计算并提高高维数据的分类精度。 已经在七个不同维度和实例数量的真实数据集上进行了这种方法的实验，并在四个不同的分类器上测量了性能：朴素贝叶斯、线性判别分析、随机森林和支持向量机。 实验结果显示了所提出的方法优于几种竞争特征选择技术的优势。





# 第二问：

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