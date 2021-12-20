## Introduction

GANS consist of two models：a generator and a discriminator. These two models are typically implemented by neural network.  but they can be implemented with any form of differentiable system that maps data from one space to the other. 

The generator tries to capture the distribution of true examples for new data example generation

The discriminator is usually a binary classifier, discriminating generated examples from the true examples as accurately as possible

 The optimization terminates at a saddle point that is a minimum with respect to the generator and a maximum with respect to the discriminator

That is, the optimization goal is to reach Nash equilibrium

Then, the generator can be thought to have captured the real distribution of true examples.

gan包含两个模型，一个生成器一个判别器。他们都是由神经网络实现的，生成器为了生成新的数据，他会去捕捉训练数据的分布。优化的目标是达到纳什平衡

### Generative algorithms

Generative algorithms can be classified into two classes: explicit density model and implicit density model.

#### *Explicit density model*

The explicit density models include maximum likelihood estimation (MLE), approximate inference [95], [96], and Markov chain method [97]–[99]. These explicit density models have an explicit distribution, but have lim- itations. For instance, MLE is conducted on true data and the parameters are updated directly based on the true data, which leads to an **overly smooth** generative model.

极大释然估计：直接从数据中学习，并更新参数。导致训练出来的模型过度的平滑（smooth）。

The generative model learned by approximate inference can only approach the lower bound of the objective function rather than directly approach the objective function. because of the difficulty in solving the objective function

通过近似推理学习的生成网络模型只能接近目标函数的下届，而不能直接接近目标函数。

It may fail to represent the complexity of true data distribution and learn the high-dimensional data distributions

#### *Implicit density model*

An implicit density model does not directly estimate or fit the data distribution. It produces data instances from the distribution without an explicit hypothesis 

and utilizes the produced examples to modify the model.

Prior to GANs, the implicit density model generally needs to be trained utilizing either ancestral sampling [102] or Markov chain-based sampling, which is inefficient and limits their practical applications.

GANs belong to the **directed implicit density model** category.

### **Adversarial idea**

Adversarial examples [108]–[117] have the adversarial idea, too. Adversarial examples are those examples which are very different from the real examples, but are classified into a real category very confidently, or those that are slightly different than the real examples, but are classified into a wrong category. This is a very hot research topic recently [112], [113]. To be against adversarial attacks [118], [119], references [120], [121] utilize GANs to conduct the right defense.

### **ALGORITHMS**

#### **Generative Adversarial Nets (GANs)**

The GANs framework is straightforward to implement when the models are both neural networks.

 In order to learn the generator’s distribution `pg` over data `x`, a prior on input noise variables is defined as `pz (z)`  and `z` is the noise variable.

Then, GANs represent a mapping from noise space to data space as `G (z, ✓g )`, where `G` is a differentiable function represented by a neural network with parameters `✓g`

gans直接使用神经网络实现。为了能够学习到生成器在数据x上的分布。

#### *Objective function*

The penalizations for two errors made by G are completely different. The first error is that G produces implausible samples and the penalization is rather large. The second error is that G does not produce real samples and the penalization is quite small. The first error is that the generated samples are inaccurate while the second error is that generated samples are not diverse enough.

Based on this, G prefers producing repeated but safe samples rather than taking risk to produce different but unsafe samples

G网络更愿意产生重复但安全的样本，而不是去冒险产生不同但不安全的样本。









