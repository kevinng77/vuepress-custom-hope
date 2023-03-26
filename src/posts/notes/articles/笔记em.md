---
title: 机械学习笔书|EM 算法
date: 2021-03-15
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- Machine Learning
mathjax: true
toc: true
comments: 笔记
---

# Expectation Maximization

> EM 算法笔记，源于 [机器学习-白板推导系列(十)-EM 算法（Expectation Maximization）](https://www.bilibili.com/video/BV1qW411k7ao?spm_id_from=333.999.0.0)

对于混合模型，如 GMM。使用 MLE 直接求极大似然的解析解是十分困难的。EM 解决的就是具有隐变量的混合模型的参数估计。

## 算法

EM 算法公式：

E-step：根据后验 $P(Z|X,\theta^{(t)})$ 写出 $E_{z \mid x, \theta^{(t)}}[\log P(x, z \mid \theta)]$

M-step：令期望最大化 $\theta^{(t+1)}=\arg \max _{\theta} \int_{z} \log P(x, z \mid \theta) \cdot P\left(z \mid x, \theta^{(t)}\right) d z$

其中 $\int_{z} \log P(x, z \mid \theta) \cdot P\left(z \mid x, \theta^{(t)}\right) d z$ 可以表示为 $E_{z \mid x, \theta^{(t)}}[\log P(x, z \mid \theta)]$ 。

从公式看来，EM 可以分为  **求出期望** ， **期望最大化两步** 

EM 是一个迭代算法， $\theta ^t$ 为 $t$ 时刻的参数，若做到 $\log P\left(x \mid \theta^{(t)}\right) \leq \log P\left(x \mid \theta^{(t+1)}\right)$ ，才有可能求出最大期望。

####  **公式合法性证明** 

$$
\log P(x \mid \theta)=\log \frac{P(x, z \mid \theta)}{P(z \mid x, \theta)}=\log P(x, z \mid \theta)-\log P(z \mid x, \theta)
$$

对上式两边关于 $P\left(z \mid x, \theta^{(t)}\right)$ 求期望：

$$
\begin{aligned}
\text { 左边 } &=\int_{z} P\left(z \mid x, \theta^{(t)}\right) \cdot \log P(x \mid \theta) d z \\
&=\log P(x \mid \theta) \int_{z} P\left(z \mid x, \theta^{(t)}\right) d z \\
&=\log P(x \mid \theta)
\end{aligned}
$$

$$
\text { 右边 }=\underbrace{\int_{z} P\left(z \mid x, \theta^{(t)}\right) \cdot \log P(x, z \mid \theta) d z}_{Q\left(\theta, \theta^{(t)}\right)}-\underbrace{\int_{z} P\left(z \mid x, \theta^{(t)}\right) \cdot \log P(z \mid x, \theta) d z}_{H\left(\theta, \theta^{(t)}\right)}
$$


 **对于**  $Q(\theta,\theta^{(t)})$：

$$
\begin{array}{l}
Q\left(\theta^{(t)}, \theta^{(t)}\right)=\int_{z} P\left(z \mid x, \theta^{(t)}\right) \cdot \log P\left(x, z \mid \theta^{(t)}\right) d z \\
Q\left(\theta^{(t+1)}, \theta^{(t)}\right)=\int_{z} P\left(z \mid x, \theta^{(t)}\right) \cdot \log P\left(x, z \mid \theta^{(t+1)}\right) d z
\end{array}
$$

根据定义 $\theta^{(t+1)}=\arg \max _{\theta} \int_{z} P\left(z \mid x, \theta^{(t)}\right) \cdot \log P(x, z \mid \theta) d z$

所以 $Q\left(\theta^{(t+1)}, \theta^{(t)}\right) \geq Q\left(\theta^{(t)}, \theta^{(t)}\right)$

 **对于**  $H(\theta,\theta^{(t)})$：

$$
\begin{aligned}
& H\left(\theta^{(t+1)}, \theta^{(t)}\right)-H\left(\theta^{(t)}, \theta^{(t)}\right) \\
=& \int_{z} P\left(z \mid x, \theta^{(t)}\right) \cdot \log P\left(z \mid x, \theta^{(t+1)}\right) d z-\int_{z} P\left(z \mid x, \theta^{(t)}\right) \cdot \log P\left(z \mid x, \theta^{(t)}\right) d z \\
=& \int_{z} P\left(z \mid x, \theta^{(t)}\right) \cdot\left(\log P\left(z \mid x, \theta^{(t+1)}\right)-\log P\left(z \mid x, \theta^{(t)}\right)\right) d z \\
=& \int_{z} P\left(z \mid x, \theta^{(t)}\right) \cdot \log \frac{P\left(z \mid x, \theta^{(t+1)}\right)}{P\left(z \mid x, \theta^{(t)}\right)} d z\\
=& -K L\left(P\left(z \mid x, \theta^{(t)}\right) \| P\left(z \mid x, \theta^{(t+1)}\right)\right)\\
\le&0
\end{aligned}
$$

因此 

$$
Q\left(\theta^{(t)}, \theta^{(t)}\right)-H\left(\theta^{(t)}, \theta^{(t)}\right) \leq Q\left(\theta^{(t+1)}, \theta^{(t)}\right)-H\left(\theta^{(t+1)}, \theta^{(t)}\right)
$$

所以

$$
\log P\left(x \mid \theta^{(t)}\right) \leq \log P\left(x \mid \theta^{(t+1)}\right)
$$



