---
title: Bayesian Optimization|贝叶斯优化
date: 2021-02-28
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- Bayesian Optimization
- Deep Learning
mathjax: true
toc: true
comments: 

---

# 贝叶斯优化

> Hyperparameters 超参的调整是 Deep Learning 较为耗时的一部分。比起 grid search 耗时耗力的暴力破解与 random search 的若有缘则相见。贝叶斯优化提供了一种相对有方向的搜索方案。

## 算法大致思路

![相关图片](/assets/img/BO/image-20210228155822610.png =x300)

<!--more-->

SMBO 算法的本质，是建立并不断完善 hyper parameters ($x$)与 模型最优解 ($f(x)$) 间关系的过程。这个算法的终止由人为决定，比如限制算法运行时间，设置目标值等。

### SMBO 算法解释

在上述表格中 $f$ 为我们想要求出其超参的模型

首先，选择几个超参方案 $x_1, x_2 ,...$ 组成我们的超参集 $X$，并计算在这些超参组合下模型 $f$ 可以达到的最优分数 $f(x)$。因此对于每一个超参方案都会映射到一个最优分数。这些映射的集合便是 $D = \{(x_1,f(x_1)),(x_2,f(x_2)),...\}$

而后我们对下面步邹迭代，直到达到目标条件：

+ 给定 prior $D$， 更新模型的 posterior $P(f(x_*) | x_*,D)$ 
+ 通过 posterior, 我们使用 acquisition function 求得一个新的超参方案 $x_t \in x_*$。不同的 axquisition funciton 能使我们选出的 $x_t$ 满足各种需求与假设。
+ 将 $x_t$ 带入到模型中，计算对应的 $f(x_t)$， 并将 $(x_t,f(x_t)$ 加入 $D$。这一步也是最耗费时间的一部。

最后，当算法结束时。我们便可以从 $D$ 中找到我们想要的最优超参



### 迭代中的相关算法

#### GP 高斯过程

对于迭代过程中的第一步， 求解 $P(f(x_*)|x_*,D)$  。高斯过程是最常用的，其中 $x_*$ 为我们还未计算的超参， $D$ 为以观测的超参组及其对应模型分数。



高斯过程由均值函数 $m$ 与核函数 (kernel function) $k$ 决定，例如我们可以选择如下和函数：

$$
m(x)={ }_{4}^{1} x^{2}, \quad k\left(x, x^{*}\right)=\exp \left(-{ }_{2}^{1}\left(x-x^{*}\right)^{2}\right)
$$


我们假设对于任意一个点（一个超参组） $x_i$， 他的 $P(f(x)|x)$ 服从高斯分布 $N(\mu,K)$ 

其中

$$
\begin{array}{c}
\mu_{i}=m\left(x_{i}\right) \\
\mathbf K_{i j}=k\left(x_{i}, x_{j}\right)
\end{array}
$$


给定 $x_*,D$， 当我们预测新的 $f(x_*)$ 时，根据 GP 的定义，已知与未知模型分数的联合分布 $p(f(x),f(x_*)|x,x_*)$ 服从

$$
\left(\begin{array}{c}
\mathbf{f(x)} \\
\mathbf{f(x_*)}
\end{array}\right) \sim \mathcal{N}\left(\left(\begin{array}{c}
\mathbf{0} \\
\mathbf{0}_{*}
\end{array}\right),\left(\begin{array}{cc}
\mathbf{K}_y & \mathbf{K}_{*} \\
\mathbf{K}_{*}^{T} & \mathbf{K}_{* *}
\end{array}\right)\right)
$$


其中：

$$
\begin{array}{c}
\mathbf{K}_{y}=\kappa(\mathbf{X}, \mathbf{X})+\sigma_{y}^{2} \mathbf{I}=\mathbf{K}+\sigma_{y}^{2} \mathbf{I} \\
\mathbf{K}_{*}=\kappa\left(\mathbf{X}, \mathbf{X}_{*}\right) \\
\mathbf{K}_{* *}=\kappa\left(\mathbf{X}_{*}, \mathbf{X}_{*}\right)
\end{array}
$$

因此我们可以得到后验 posterior:  $p(f(x_*)|x_*,D)$ 服从分布：

$$
\begin{aligned}
p\left(\mathbf{f(x_*)} \mid \mathbf{x}_{*}, \mathbf{D}\right) &=\mathcal{N}\left(\mathbf{f(x^*)} \mid \mu_{*}, \Sigma_{*}\right) \\
\mu_{*} &=\mathbf{K}_{*}^{T} \mathbf{K_y}^{-1} \mathbf{f(x^*)} \\
\Sigma_{*} &=\mathbf{K}_{* *}-\mathbf{K}_{*}^{T} \mathbf{K_y}^{-1} \mathbf{K}_{*}
\end{aligned}
$$


### Acquisition Function

如何找到下一个我们将要探索的点，是这整个算法的关键。与强化学习相似，我们希望我们的模型能够权衡 exploration 和 exploitation。



#### Expected Improvement

EI 的假设是：我们观测到的 $f(x)$ 并不存在误差项。

EI 的 utility function 取决于我们的 objective function 的优化方向，一般来说，当我们使用 loss 等计算模型评分，此时我们希望 loss 越小越好，则：

$$
u(x) = max(0, f' - f(x))
$$

其中， $f' = min(f)$，即我们观测到的 $D$ 中的模型最低分。

因此 

$$
x_t = \arg \max _{x \in \mathcal{X}}{E[u(x)|x,D]} \\
\begin{array}{l}
=\int_{-\infty}^{f^{\prime}}\left(f^{\prime}-f\right) \mathcal{N}(f ; \mu(x), \mathcal{K}(x, x)) d f \\
=\left(f^{\prime}-\mu(x)\right) \Phi\left(f^{\prime} ; \mu(x), \mathcal{K}(x, x)\right)+\mathcal{K}(x, x) \mathcal{N}\left(f^{\prime} ; \mu(x), \mathcal{K}(x, x)\right)
\end{array}
$$


若我们的优化对象为类似 $R^2$ 的大而好指标。则：

$$
u(x) = max(0,f(x) - f')
$$

其中， $f' = max(f)$，即我们观测到的 $D$ 中的模型最高分。

因此：

$$
x_t = \arg \max _{x \in \mathcal{X}} E[u(x)|x,D]\\
=\arg \max _{x \in \mathcal{X}}\left(\mu_{t-1}(x)-f'\right) \Phi\left(\frac{\mu_{t-1}(x)-f'}{\sigma_{t-1}(x)}\right)+\sigma_{t-1}(x) \phi\left(\frac{\mu_{t-1}(x)-f'}{\sigma_{t-1}(x)}\right)
$$


其中 $\Phi$ 和 $\phi$ 分别是 standard Gaussian distribution 的 cumulative distribution function（CDF）和 probability density function（PDF）。



#### probability of improvement

很经典的算法，但是似乎现在很少用了？

与 EI 相似，Probalitity of Improvement 的 utility function 也是由优化的方向决定，当我们希望模型指标越小越好时：


$$
u(x)=\left\{\begin{array}{ll}
o, & \text { if } f(x)>f^{\prime} \\
1, & \text { if } f(x) \leq f^{\prime}
\end{array}\right.
$$

其中， $f' = min(f)$，即我们观测到的 $D$ 中的模型最低分。

因此

$$
\begin{aligned}x_t =\arg \max _{x \in \mathcal{X}}
E[u(x) \mid x, D]

&=\int_{-\infty}^{f^{\prime}} \mathcal{N}(f ; \mu(x), \mathcal{K}(x, x)) d f \\
&=\Phi\left(f^{\prime} ; \mu(x), \mathcal{K}(x, x)\right)
\end{aligned}
$$


#### Entropy Search

类似于 decision tree 的 partition criteria，我们选择能提供做多信息的那个变量。

其中 utility function 为：

$$
u(x) = H[ x^* |D] - H[x^* | D, x, f(x)]
$$


#### Gaussian Process-Upper Confidence Bound :

$$
x_{t}=\arg \max _{x \in \mathcal{X}} \alpha_{t}(x)=\arg \max _{x \in \mathcal{X}} \mu_{t-1}(x)+\beta_{t}^{1 / 2} \sigma_{t-1}(x)
$$

其中 $\beta_t$ 用于平衡 exploration 和 exploitation。可以理解为 $\mu$ 的权重越大，UCB 则越倾向于 exploration。

"公式里面 $\beta_t$ 的值是根据理论分析推出来的，随时间递增；可是在实际应用里面，好多人为了简便直接把 $\beta_t$ 设成一个常数，也是可以的。" [引用观点](https://zhuanlan.zhihu.com/p/76269142)

## 参考

https://zhuanlan.zhihu.com/p/76269142

https://www.cnblogs.com/marsggbo/p/9866764.html

https://zhuanlan.zhihu.com/p/72403538

https://zhuanlan.zhihu.com/p/73832253

https://blog.csdn.net/weixin_41503009/article/details/107679561

https://blog.csdn.net/lj6052317/article/details/78772494/