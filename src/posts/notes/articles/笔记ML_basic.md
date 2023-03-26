---
title: 白板机械学习笔书|基础一
date: 2021-05-07
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- Machine Learning
mathjax: true
toc: true
comments: 笔记
---

> 机械学习基础。笔记思路参考了 [shuhuai-白板机械学习](https://space.bilibili.com/97068901/video?tid=0&page=2&keyword=&order=pubdate) 系列教程。

<!--more-->

## 基础

#### 极大似然估计

极大似然估计可以理解为：已知观测变量来源于分布 $f(\theta)$，求 $\theta$ 使得出现当前观测变量的概率最大。
解极大似然，可以先定义出似然函数，而后取对数再令导数为 0，得到似然方程并求解。

如假设 $X$ 中有 $N$ 个观测样本，样本来源分布符合高斯分布 $f(\mu,\sigma)$。首先写出根据假设的高斯分布似然函数：

$$
\begin{aligned}
\log P(X \mid \theta) &=\log \prod_{i=1}^{N} P\left(x_{i} \mid \theta\right) \\
&=\sum_{i=1}^{N} \log \left(\frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{(x-\mu)^{2}}{2 \sigma^{2}}\right)\right) \\
&=\sum_{i=1}^{N}\left[\log \frac{1}{\sqrt{2 \pi}}+\log \frac{1}{\sigma}-\frac{(x-\mu)^{2}}{2 \sigma^{2}}\right]
\end{aligned}
$$

求解最优的 $\mu$ 参数：

$$
\begin{aligned}
\mu_{M L E} &=\arg \max _{\mu} \log P(X \mid \theta) \\
&=\arg \max _{\mu} \sum_{i=1}^{N}-\frac{\left(x_{i}-\mu\right)^{2}}{2 \sigma^{2}} \\
&=\arg \min _{\mu} \sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}
\end{aligned}
$$

而后求导：

$$
\begin{aligned}
\frac{\partial}{\partial \mu} \sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2} &=\sum_{i=1}^{N} 2\left(x_{i}-\mu\right)(-1)=0 \\
\sum_{i=1}^{N}\left(x_{i}-\mu\right) &=0 \\
\sum_{i=1}^{N} x_{i}-N \mu &=0 \\
\mu_{M L E} &=\frac{1}{N} \sum_{i=1}^{N} x_{i}
\end{aligned}
$$

同样的可以求得 $\sigma$ 的最优参数值：

$$
\begin{aligned}
\sigma_{M L E}^{2} &=\arg \max _{\sigma} \sum_{i=1}^{N}\left[\log \frac{1}{\sigma}-\frac{\left(x_{i}-\mu\right)^{2}}{2 \sigma^{2}}\right] \\
&=\arg \max _{\sigma} \sum_{i=1}^{N}\left[-\log \sigma-\frac{1}{2 \sigma^{2}}\left(x_{i}-\mu\right)^{2}\right]
\end{aligned}
$$

令导数为 0：

$$
\begin{aligned}
\frac{\partial}{\partial \sigma} \sum_{i=1}^{N}\left[-\log \sigma-\frac{1}{2 \sigma^{2}}\left(x_{i}-\mu\right)^{2}\right] &=\sum_{i=1}^{N}\left[-\frac{1}{\sigma}-\frac{1}{2}\left(x_{i}-\mu\right)^{2}(-2) \frac{1}{\sigma^{3}}\right]=0 \\
\sum_{i=1}^{N}\left[-\frac{1}{\sigma}+\left(x_{i}-\mu\right)^{2} \sigma^{-3}\right] &=0 \\
\sum_{i=1}^{N}\left[-\sigma^{2}+\left(x_{i}-\mu\right)^{2}\right] &=0 \\
-N \sigma^{2}+\sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2} &=0 \\
\sigma_{M L E}^{2} &=\frac{1}{N} \sum_{i=1}^{N}\left(x_{i}-\mu_{M L E}\right)^{2}
\end{aligned}
$$


#### 有偏估计与无偏估计

若参数数学期望等于它本身： $E[\hat{\mu}]=\mu \quad E[\hat{\sigma}]=\sigma$ ，则为无偏估计。

对于以上结果 $\mu$ 为无偏估计：

$$
\begin{aligned}
E\left[\mu_{M L E}\right] &=E\left[\frac{1}{N} \sum_{i=1}^{N} x_{i}\right] \\
&=\frac{1}{N} \sum_{i=1}^{N} E\left[x_{i}\right] \\
&=\frac{1}{N} \sum_{i=1}^{N} \mu \\
&=\mu
\end{aligned}
$$

 $\sigma_{M L E}^{2} =\frac{1}{N} \sum_{i=1}^{N}\left(x_{i}-\mu_{M L E}\right)^{2}$ 的最优解为有偏估计：

$$
\begin{aligned}
E\left[\sigma_{M L E}^{2}\right] &=E\left[\frac{1}{N} \sum_{i=1}^{N}\left(x_{i}-\mu_{M L E}\right)^{2}\right] \\
&=E\left[\frac{1}{N} \sum_{i=1}^{N}\left(x_{i}^{2}-2 x_{i} \mu_{M L E}+\mu_{M L E}^{2}\right)\right] \\
&=E\left[\frac{1}{N} \sum_{i=1}^{N} x_{i}^{2}-2 \mu_{M L E} \frac{1}{N} \sum_{i=1}^{N} x_{i}+\frac{1}{N} \sum_{i=1}^{N} \mu_{M L E}^{2}\right] \\
&=E\left[\frac{1}{N} \sum_{i=1}^{N} x_{i}^{2}-2 \mu_{M L E}^{2}+\mu_{M L E}^{2}\right] \\
&=E\left[\frac{1}{N} \sum_{i=1}^{N}\left(x_{i}^{2}-\mu_{M L E}^{2}\right)\right] \\
&=\frac{1}{N} \sum_{i=1}^{N}\left(E\left[x_{i}^{2}\right]-E\left[\mu_{M L E}^{2}\right]\right) \\
&=\frac{1}{N} \sum_{i=1}^{N}\left(D\left[x_{i}\right]+E\left[x_{i}\right]^{2}-D\left[\mu_{M L E}\right]-E\left[\mu_{M L E}\right]^{2}\right) \\
&=\frac{1}{N} \sum_{i=1}^{N}\left(\sigma_{M L E}^{2}+\mu^{2}-D\left[\frac{1}{N} \sum_{i=1}^{N} x_{i}\right]-\mu^{2}\right) \\
&=\frac{1}{N} \sum_{i=1}^{N}\left(\sigma_{M L E}^{2}-\frac{1}{N^{2}} N \sigma_{M L E}^{2}\right) \\
&=\frac{N-1}{N} \sigma_{M L E}^{2}
\end{aligned}
$$

无偏估计为：$\sigma_{M L E}^{2} =\frac{1}{N-1} \sum_{i=1}^{N}\left(x_{i}-\mu_{M L E}\right)^{2}$。  **因此从上面结果可以看出，使用极大似然估计，会带来一定的偏差。** 

#### 高维高斯分布

$x \sim N(\mu, \Sigma)=\frac{1}{(2 \pi)^{\frac{p}{2}}|\Sigma|^{\frac{1}{2}}} \exp (-\frac{1}{2} {(x-\mu)^{T} \Sigma^{-1}(x-\mu)})$

 **指数部分可以理解为 $x$ 与 $\mu$ 的**  [马氏距离](https://zhuanlan.zhihu.com/p/46626607)。

对 $\Sigma$ 进行特征值分解： $\Sigma=U \Lambda U^{T}$ ；其中 $U U^{T}=U^{T} U=I$ （两者正交） $U=\left(u_{1}, u_{2}, \cdots, u_{p}\right)_{p \times p} \Lambda=\operatorname{diag}_{i=1, \cdots, p}\left(\lambda_{i}\right)$ （对角阵）

 **特征值分解可以将矩阵分解为连加形式：** 

$$
\begin{aligned}
\Delta &=(x-\mu)^{T} \Sigma^{-1}(x-\mu) \\
&=(x-\mu)^{T} \sum_{i=1}^{p}\left(u_{i} \frac{1}{\lambda_{i}} u_{i}^{T}\right)(x-\mu) \\
&=\sum_{i=1}^{p}\left((x-\mu)^{T} u_{i} \frac{1}{\lambda_{i}} u_{i}^{T}(x-\mu)\right) \\
&=\sum_{i=1}^{p}\left(y_{i} \frac{1}{\lambda_{i}} y_{i}^{T}\right) \\
&=\sum_{i=1}^{p}\left(\frac{y_{i}^{2}}{\lambda_{i}}\right)
\end{aligned}
$$

其中 $y_i = (x-\mu)^T u_i$ ，$y_i$ 是 $(x-\mu)$ 在 $\mu_i$ 上的投影。通过上述公式， **马氏距离可以理解为各个特征值 $\lambda_i$ 的加权。加权系数为不同维度特征在各个特征向量上的投影。**  

从几何上理解，当 $p=2$ 时，$\Delta=\frac{y_{1}^{2}}{\lambda_{1}}+\frac{y_{2}^{2}}{\lambda_{2}}=r$ 即为椭圆。$\Delta$ 的值决定了椭圆大小。

$p(x)=\frac{1}{(2 \pi)^{\frac{p}{2}}|\Sigma|^{\frac{1}{2}}} \exp \left(-\frac{1}{2} \Delta\right) \quad 0 \leq p(x) \leq 1$ ，若 $p=2$ ，$p(x)$ 给定后，$(x_1,x_2)$ 就是在三维空间 $(x_1,x_2,p(x))$ 上的椭圆曲线的切面。

由于 $\Sigma$ 参数复杂度为 $O(p^2)$，通常假设其为对角矩阵以简化计算（如 PCA），此时椭圆曲线的切面为正的，椭圆的对称轴平行于坐标轴。

若对角矩阵 $\Sigma$ 中的 $\lambda_i$ 都相等，则切面为正圆，这种情况称为 **各向同性** 。

#### 概率分布

给定联合概率分布

$$
x \sim N(\mu, \Sigma)=\frac{1}{(2 \pi)^{\frac{p}{2}}|\Sigma|^{\frac{1}{2}}} \exp (-\frac{1}{2} {(x-\mu)^{T} \Sigma^{-1}(x-\mu)})
$$

求边缘概率分布 $p(x_a),p(x_b)$ 与 条件概率分布 $p(x_b|x_a),p(x_a|x_b)$ 。（通常使用配方法）[视频](https://www.bilibili.com/video/BV1RW411m7WE?p=5&spm_id_from=pageDriver) 中使用了另一种方法：

 **定理：**  已知 $x \sim N(\mu, \Sigma)$ ，$y=Ax+B$ 则 $y \sim N\left(A \mu+B, A \Sigma A^{T}\right)$

求 $p(x_a)$：

令  $x_{a}=\underbrace{\left(\begin{array}{ll} I_{m} & 0 \end{array}\right)}_{A} \underbrace{\left(\begin{array}{c} x_{a} \\ x_{b} \end{array}\right)}_{x}+\underbrace{0}_{B}$

$$
\begin{array}{l}
E\left[x_{a}\right]=A \mu+B=\left(\begin{array}{ll}
I_{m} & 0
\end{array}\right)\left(\begin{array}{c}
\mu_{a} \\
\mu_{b}
\end{array}\right)+0=\mu_{a} \\
Var\left[x_{a}\right]=A \Sigma A^{T}=\left(\begin{array}{ll}
I_{m} & 0
\end{array}\right)\left(\begin{array}{cc}
\Sigma_{a a} & \Sigma_{a b} \\
\Sigma_{b a} & \Sigma_{b b}
\end{array}\right)\left(\begin{array}{c}
I_{m} \\
0
\end{array}\right)=\left(\begin{array}{cc}
\Sigma_{a a} & \Sigma_{a b}
\end{array}\right)\left(\begin{array}{c}
I_{m} \\
0
\end{array}\right)=\Sigma_{a a}
\end{array}
$$

则 $x_a \sim N(\mu_a, \Sigma_{aa})$

求 $p(x_b|x_a)$，先定义：

$$
\begin{aligned}
x_{b \cdot a} &=x_{b}-\Sigma_{b a} \Sigma_{a a}^{-1} x_{a} \\
\mu_{b \cdot a} &=\mu_{b}-\Sigma_{b a} \Sigma_{a a}^{-1} \mu_{a} \\
\Sigma_{b b \cdot a} &=\Sigma_{b b}-\Sigma_{b a} \Sigma_{a a}^{-1} \Sigma_{a b}
\end{aligned}
$$

其中第二、三式可通过一式推导得来。提示：将一式写成： 

$$
x_{b \cdot a}=\left(\begin{array}{cc}
-\Sigma_{b a} \Sigma_{a a}^{-1} & I
\end{array}\right)\left(\begin{array}{l}
x_{a} \\
x_{b}
\end{array}\right)
$$

相关知识：Schur Complement

由一式可得： $x_{b·a} \sim N(\mu_{b·a}, \Sigma_{bb·a})$，$x_{b}=x_{b \cdot a}+\Sigma_{b a} \Sigma_{a a}^{-1} x_{a}$ ，当 $x_a$ 给定时可以看做常量。因此：

$$
\begin{array}{l}
E\left[x_{b} \mid x_{a}\right]=\mu_{b \cdot a}+\Sigma_{b a} \Sigma_{a a}^{-1} x_{a} \\
Var\left[x_{b} \mid x_{a}\right]=Var\left[x_{b·a} \right]=\Sigma_{b b \cdot a}
\end{array}
$$

求得：

$$
x_{b} \mid x_{a} \sim N\left(\mu_{b \cdot a}+\Sigma_{b a} \Sigma_{a a}^{-1} x_{a}, \Sigma_{b b \cdot a}\right) \tag 1
$$

#### 线性高斯模型基础

已知：$p(x)=N\left(\mu, \Lambda^{-1}\right)$ ，$p(y \mid x)=N\left(A x+b, L^{-1}\right)$ 
求：$p(y),p(x|y)$ 

此处 $\Lambda^{-1}$ 为 $(\text { convariance matrix })^{-1}$

 **求**  $p(y)$ ：

$$
E[y]=E[A x+b+\epsilon]=A E[x]+b+E[\epsilon]=A \mu+b\\
Var[y]=Var[A x+b+\epsilon]=Var[A x+b]+Var[\epsilon]\\
= A\Lambda^{-1A^T} + L^{-1}
$$

因此：

$$
y \sim N\left(A \mu+b, A \Lambda^{-1} A^{T}+L^{-1}\right)
$$

 **求**  $p(x|y)$:

思路：先求出联合概率分布，在通过上文定理求条件概率。

定义 $z=\left(\begin{array}{l} x \\ y \end{array}\right)$ ，通过上文结论可知：

$$
Var[z]=\left(\begin{array}{cc}
\operatorname{cov}(x, x) & \operatorname{cov}(x, y) \\
\operatorname{cov}(y, x) & \operatorname{cov}(y, y)
\end{array}\right)=\left(\begin{array}{cc}
\Lambda^{-1} & \operatorname{cov}(x, y) \\
\operatorname{cov}(y, x) & L^{-1}+A \Lambda^{-1} A^{T}
\end{array}\right)
$$

只要求出 $cov(x,y)$ ，就可以通过上节定理求出 $p(x|y)$！

$$
\begin{aligned}
\operatorname{cov}(x, y) &=E\left[(x-E[x]) \cdot(y-E[y])^{T}\right] \\
&=E\left[(x-\mu) \cdot(A x+b+\epsilon-A \mu-b)^{T}\right] \\
&=E\left[(x-\mu) \cdot(A x-A \mu+\epsilon)^{T}\right] \\
&=E\left[(x-\mu) \cdot(A x-A \mu)^{T}+(x-\mu) \cdot \epsilon^{T}\right] \\
&=E\left[(x-\mu)(x-\mu)^{T} A^{T}\right]+E\left[(x-\mu) \epsilon^{T}\right]
\end{aligned}
$$

因为 $x-\mu$ 与 $\epsilon$ 独立，所以 $E\left[(x-\mu) \epsilon^{T}\right]=E[x-\mu] \cdot E\left[\epsilon^{T}\right]=(E[x]-\mu) E\left[\epsilon^{T}\right]=0$

继续，求得：

$$
\begin{aligned}
\operatorname{cov}(x, y) &=E\left[(x-\mu)(x-\mu)^{T} A^{T}\right] \\
&=E\left[(x-\mu)(x-\mu)^{T}\right] A^{T} \\
&=D[x] A^{T} \\
&=\Lambda^{-1} A^{T}
\end{aligned}
$$

所以，结合上节 $(1)$ 式可得：

$$
z \sim N\left(\left(\begin{array}{c}
\mu \\
A \mu+b
\end{array}\right),\left(\begin{array}{cc}
\Lambda^{-1} & \Lambda^{-1} A^{T} \\
A \Lambda^{-1} & L^{-1}+A \Lambda^{-1} A^{T}
\end{array}\right)\right)\\
x \mid y \sim N\left(\mu+\Lambda^{-1} A^{T}\left(L^{-1}+A \Lambda^{-1} A^{T}\right)^{-1}(y-A \mu-b), \Lambda^{-1}-\Lambda^{-1} A^{T}\left(L^{-1}+A \Lambda^{-1} A^{T}\right)^{-1} A \Lambda^{-1}\right)
$$

#### Jensen's 不等式

对于凸函数 $f(x)$ 有：$E[f(x)] \ge f(E[x])$ 。

 **证明：** 

过点 $(E[x],f(E[x]))$ 做切线 $l(x)=ax+b$。因此：$f(E[x]) = l(E[x]),\forall x ,f(x)\ge l(x)$

$$
\begin{aligned}
E[f(x)] & \geq E[l(x)] \\
&=E[a x+b] \\
&=a E[x]+b \\
&=f(E[x])
\end{aligned}
$$

 **Jensen's 不等式变型：** 

$$
t f(a)+(1-t) f(b) \geq f(t a+(1-t) b)
$$




