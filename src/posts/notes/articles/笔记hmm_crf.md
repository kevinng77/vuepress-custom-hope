---
title: 机械学习| HMM CRF 简笔
date: 2021-08-06
category:
- 知识笔记
tag: 
- NLP
- Machine Learning
---

> HMM/CRF 笔记 
> [机器学习-白板推导系列(十四)-隐马尔可夫模型 HMM](https://www.bilibili.com/video/BV1MW41167Rf?spm_id_from=333.999.0.0) 
> [机器学习-白板推导系列(十七)-条件随机场 CRF](https://www.bilibili.com/video/BV19t411R7QU?spm_id_from=333.999.0.0)

<!--more-->

## HMM

![相关图片](/assets/img/hmm_crf/image-20220316065626099.png =x300)

对于静态图，样本之间是独立同分布的：$x_{i} \stackrel{i i d}{\sim} p(x \mid \theta)$。 HMM 属于概率图中的动态模型，x 之间并不是独立的。

如对于一个 GMM 静态图 ，有 $P(x|z) \sim N(\mu,\Sigma)$ 。若加上时间则生成序列（称为动态模型）：

![相关图片](/assets/img/hmm_crf/image-20220316070532082.png =x300)

图例中，上方为 GMM 静态图，下方为动态图， **横向代表时间** ， **纵向代表混合** 。图中阴影 $X$ 为观测变量， **观测变量** 有对应的 **隐状态** ，这些隐状态通称为 **系统状态** 。

动态度可继续划分，系统状态是离散的话，则模型为 HMM；若状态是连续的，线性的为 Kalman 滤波，非线性的为 Particle 滤波。

####  **HMM 模型：** 

![相关图片](/assets/img/hmm_crf/image-20220316071056875.png )

图中阴影变量为观测变量 $o_t$，状态空间 $V=\left\{v_{1}, v_{2}, \cdots, v_{M}\right\}$ 
白圈为状态变量 $i_t$，状态空间 $Q=\left\{q_{1}, q_{2}, \cdots, q_{N}\right\}$ 。

HMM 模型可表示为： $\lambda = (\pi, A, B)$ 其中转移矩阵为 $A=[a_{ij}],a_{ij}=P(i_{t+1}=q_j|i_t=q_i)$ 

发射矩阵为 $B=\left[b_{j}(k)\right], \quad b_{j}(k)=p\left(o_{t}=v_{k} \mid i_{t}=q_{j}\right)$ 。

#### HMM 假设

 **齐次 Markov 假设** ：当前状态 **仅** 与前一状态有关

 **观测独立假设** ：当前观测变量 **仅** 与当前对应的状态变量有关。

### HMM 的三个环节

Learning：如何学习模型的参数？
Decoding：给定模型参数与观测序列，如何找到一个状态序列 $I$ 使得 $P(I|O)$ 最大？
Evaluation：给定模型所有参数，如何衡量观测序列的概率？

#### Evaluation

给定 $\lambda$ 求 $P(O \mid \lambda)=\sum_{I} P(I, O \mid \lambda)=\sum_{I} P(O \mid I, \lambda) \cdot P(I \mid \lambda)$

基于齐次 Markov 假设：

$$
P(I \mid \lambda)=P\left(i_{T} \mid i_{T-1}, \lambda\right) \cdot P\left(i_{T-1} \mid i_{T-2}, \lambda\right) \cdots P\left(i_{2} \mid i_{1}, \lambda\right) \cdot P\left(i_{1}, \lambda\right)\\
P(I \mid \lambda)=\pi\left(a_{i_{1}}\right) \prod_{t=2}^{T} a_{i_{t-1}, i_{t}}
$$

基于观察独立假设：

$$
P(O \mid I, \lambda)=P\left(o_{T} \mid i_{T}, \lambda\right) \cdot P\left(o_{T-1} \mid i_{T-1}, \lambda\right) \cdots P\left(o_{1} \mid i_{1}, \lambda\right)\\
P(O \mid I, \lambda)=\prod_{t=1}^{T} b_{i_{t}}\left(o_{t}\right)
$$

暴力算法时间复杂度 $O(N^T)$

 **前向算法**  （$O(TN^2)$）：

$$
\alpha_{t+1}(j)=\sum_{i=1}^{N} b_{j}\left(O_{t+1}\right) \cdot a_{i j} \cdot \alpha_{t}(i)
$$

![相关图片](/assets/img/hmm_crf/image-20220316080244028.png =x300)

 **后向算法：** 

记：$\beta_{1}(i)=P\left(o_{2}, \cdots, o_{T} \mid i_{1}=q_{i}, \lambda\right)$ 

那么：

$$
P(O|\lambda)=\sum_{i=1}^{N} b_{i}\left(o_{1}\right) \cdot \beta_{1}(i) \cdot \pi_{i}\\
\beta_t(i)=\sum_{j=1}^{N} b_{j}\left(o_{t+1}\right) \cdot a_{i j} \cdot \beta_{t+1}(j)
$$

#### Learning

问题描述：求解 $\lambda_{M L E}=\underset{\lambda}{\operatorname{argmax}} P(O \mid \lambda)$

将 EM 算法应用与 HMM 后得出：

$$
\lambda^{(t+1)}=\arg \operatorname{aax}_{\lambda} \sum_{I} \log P(O, I \mid \lambda) \cdot P\left(I \mid O, \lambda^{(t)}\right)
$$

其中 $P\left(I \mid O, \lambda^{(t)}\right) = \frac{P\left(I, O \mid \lambda^{(t)}\right)}{P\left(O \mid \lambda^{(t)}\right)}$ ，$\lambda^{t}$ 与 $O$ 已知，因此：

$$
\lambda^{(t+1)}=\operatorname{argmax} \sum_{\lambda} \log P(O, I \mid \lambda) \cdot P\left(O, I \mid \lambda^{(t)}\right)
$$

在上一节已经求出：

$$
P(O \mid \lambda)=\sum_{I} P(O, I \mid \lambda)=\sum_{i_{1}} \sum_{i_{2}} \cdots \sum_{i_{T}}\left[\pi\left(a_{i_{1}}\right) \prod_{t=2}^{T} a_{i_{t-1}, i_{t}} \cdot \prod_{t=1}^{T} b_{i_{t}}\left(o_{t}\right)\right]
$$

所以：

$$
P(O, I \mid \lambda)=\pi\left(a_{i_{1}}\right) \prod_{t=2}^{T} a_{i_{t-1}, i_{t}} \cdot \prod_{t=1}^{T} b_{i_{t}}\left(o_{t}\right)
$$

$$
\sum_{\lambda} \log P(O, I \mid \lambda) \cdot P\left(O, I \mid \lambda^{(t)}\right)\\=\sum_{I}\left[\left(\log \pi_{i_{1}}+\sum_{t=2}^{T} \log a_{i_{t-1}, i_{t}}+\sum_{t=1}^{T} \log b_{i_{t}}\left(o_{t}\right)\right) \cdot P\left(O, I \mid \lambda^{(t)}\right)\right]
$$

其中 $\lambda$ 有 $\pi, A, B$ 三个参数，以下挑选 $\pi$ 进行推导：

$$
\begin{aligned}
\pi^{(t+1)} &=\underset{\pi}{\operatorname{argmax}} Q\left(\lambda, \lambda^{(t)}\right) \\
&=\operatorname{argmax} \sum_{\pi}\left[\log \pi_{i_{1}} \cdot P\left(O, I \mid \lambda^{(t)}\right)\right] \\
&=\underset{\pi}{\operatorname{argmax}} \sum_{i_{1}} \cdots \sum_{i_{T}}\left[\log \pi_{i_{1}} \cdot P\left(O, i_{1}, \cdots, i_{T} \mid \lambda^{(t)}\right)\right] \\
&=\underset{\pi}{\operatorname{argmax}} \sum_{i_{1}}\left[\log \pi_{i_{1}} \cdot P\left(O, i_{1} \mid \lambda^{(t)}\right)\right] \\
&=\operatorname{argmax} \sum_{\pi}^{N}\left[\log \pi_{i} \cdot P\left(O, i_{1}=q_{i} \mid \lambda^{(t)}\right)\right]
\end{aligned}
$$

HMM 的基础概念中有约束条件：$\sum_{i=1}^{N} \pi_{i}=1$

因此使用拉格朗日乘子法求解：

$$
L(\pi, \eta)=\sum_{i=1}^{N}\left[\log \pi_{i} \cdot P\left(O, i_{1}=q_{i} \mid \lambda^{(t)}\right)\right]+\eta\left(\sum_{i=1}^{N} \pi_{i}-1\right)
$$

$$
\frac{\partial L}{\partial \pi_{i}}=\frac{1}{\pi_{i}} P\left(O, i_{1}=q_{i} \mid \lambda^{(t)}\right)+\eta=0\\
P\left(O, i_{1}=q_{i} \mid \lambda^{(t)}\right)+\pi_{i} \eta=0\\
\sum_{i=1}^{N}\left[P\left(O, i_{1}=q_{i} \mid \lambda^{(t)}\right)+\pi_{i} \eta\right]=0
$$


$$
P\left(O \mid \lambda^{(t)}\right)+\eta=0\\
\eta=-P\left(O \mid \lambda^{(t)}\right)
$$

将 $\eta$ 代入 $P\left(O, i_{1}=q_{i} \mid \lambda^{(t)}\right)+\pi_{i} \eta=0$ 得:

$$
\pi_{i}^{(t+1)}=\frac{P\left(O, i_{1}=q_{i} \mid \lambda^{(t)}\right)}{P\left(O \mid \lambda^{(t)}\right)}
$$


$$
\pi^{(t+1)}=\left(\pi_{1}^{(t+1)}, \pi_{2}^{(t+1)}, \cdots, \pi_{N}^{(t+1)}\right)
$$

#### Decoding

解码采用维特比算法：

以下使用概率矩阵 $C$ 与 动作矩阵 $D$ 来进行辅助解释。其中 $X={w_1,w_2..w_K}$ 为观测变量，$Z=t_1,t_2,..t_N$ 为隐状态。$a$ 为状态转移矩阵，$b$ 为发射矩阵。

概率矩阵中 $c_{i,j}$ 记录 $max\ P(w_1,w_2...w_i,z_1,z_2..,z_i=t_j)$ 。即，对于序列 $w_1,w_2...w_i$ 在 $i$ 时刻对应的隐状态为 $t_j$ 时候的最大概率。$i$ 时刻之前的隐状态没有条件限制。

动作矩阵中 $d_{ij}$ 记录：对于序列 $w_1,w_2...w_i$ ， $i$ 时刻对应的隐状态为 $t_j$ 概率最大时， $i-1$ 时刻隐状态为 $d_{ij}$ 。

1. 初始化概率矩阵 $C$ 与 动作矩阵 $D$。

<img src="/assets/img/NLP_basic/image-20210428172547464.png">

<img src="/assets/img/NLP_basic/image-20210428172714202.png">

2. 前向推导，填充满 $C,D$ 两个矩阵。

<img src="/assets/img/NLP_basic/image-20210428172957030.png">

<img src="/assets/img/NLP_basic/image-20210428173049634.png">

3. 反向推导。

<img src="/assets/img/NLP_basic/image-20210428174009944.png">

## CRF

对于解决分类问题，模型可以分为硬模型（输出为类似 01 的确认值）与软模型（输出为概率）。

软模型可以再分为:
概率生成模型（对 $P(X,Y)$ 建模）。如朴素贝叶斯、HMM。
概率判别模型（对 $P(X|Y)$）进行建模。如 MEMM、CRF

CRF 概率图如下，打破了 HMM 的观测独立假设，即 $y_2$ 会受到 $x_1,x_2..x_t$ 的影响：

![相关图片](/assets/img/hmm_crf/image-20220316095103352.png =x300)

 **随机场：** 当给每一个位置中按照某种分布随机赋予相空间的一个值之后，其全体就叫做随机场。

 **马尔科夫随机场：** 

马尔科夫随机场及由马尔科夫性质的随机场，它指的是一个随机变量序列按时间先后关系依次排开的时候，第 N+1 时刻的分布特性，与 N 时刻以前的随机变量的取值无关。（比如今天天气仅与昨天有关，与前天...无关）

 **线性条件随机场 CRF** 

通常 NLP 中的 CRF 默认为线性链条件随机场。如同马尔科夫随机场，条件随机场为无向图模型。不同的是，CRF 具有给定的观测变量 $X$。

线性链条件随机场中 $X$ 与 $Y$ 结构相同。

####  **线性链条件随机场分解：** 

根据 Hammersley-Clifford 定理，马尔科夫随机场可以分解为：

$$
P(x)=\frac{1}{Z} \prod_{i=1}^{K} \psi_{i}\left(x_{c i}\right), x \in \mathbb{R}^{p}
$$

$C$ 是无向图的最大团，$x$ 联合概率分布，$\psi_{i}$为势函数。通常指定势函数为指数函数，因此：

$$
\begin{aligned}
P(x) &=\frac{1}{Z} \prod_{i=1}^{K} \exp \left[-E_{i}\left(x_{c i}\right)\right], x \in \mathbb{R}^{p} \\
&=\frac{1}{Z} \exp \sum_{i=1}^{K} F_{i}\left(x_{c i}\right), x \in \mathbb{R}^{p}
\end{aligned}
$$

根据 CRF 的概率图表示，有：

$$
P(Y \mid X)=\frac{1}{Z} \exp \sum_{t=1}^{T} F\left(y_{t-1}, y_{t}, x_{1: T}\right)
$$

其中 $F\left(y_{t-1}, y_{t}, x_{1: T}\right)=\triangle y_{t}, x_{1: T}+\triangle y_{t-1}, y_{t}, x_{1: T}$

$\triangle y_{t-1}, y_{t}, x_{1: T}=\sum_k \lambda_kf_k(y_{t-1}, y_{t}, x_{1: T}  )$ 为转移函数（类似 HMM 转移矩阵）

$\triangle y_{t}, x_{1: T}=\sum_{l=1}^{L} \eta_{l} g_{l}\left(y_{t}, x_{1: 1}\right)$ 为状态函数（类似 HMM 发射矩阵）。

其中  ，$f_k,g_l \in {0,1}$ 为给定的特征函数。对状态及转移的可能性进行限制。而 $\lambda, \eta$ 是需要学习的参数。

因此 CRF 的概率函数总结为：

$$
P(Y \mid X)=\frac{1}{Z(X)} \exp \sum_{i=1}^{T}\left(\sum_{k=1}^{K} \lambda_{k} t_{k}\left(y_{i-1}, y_{i}, X, i\right)+\sum_{l=1}^{L} \mu_{l} s_{l}\left(y_{i}, X, i\right)\right)
$$

#### CRF 简化表示

$$
P(Y=y \mid X=x)=\frac{1}{Z(x, \lambda, \eta)} \exp \sum_{t=1}^{T}\left[\lambda^{\top} \cdot f\left(y_{t-1}, y_{t}, x\right)+\eta^{\top} \cdot g\left(y_{t}, x\right)\right]
$$

定义 $\theta=\left(\begin{array}{l} \lambda \\ \eta \end{array}\right)_{k+L}$ ，$H=\left(\begin{array}{c} \sum_{t=1}^{T} f \\ \sum_{t=1}^{T} g \end{array}\right)_{k+L}$

因此： 

$$
P(Y=y \mid X=x)=\frac{1}{Z(x, \lambda, \eta)} \exp \sum_{t=1}^{T}\left[\theta^TH \right]
$$

其中 $Z$ 为归一化因子。

#### 参数学习 Learning

学习的目标是  $argmax_{\lambda, \eta}\prod P(y_i|x_i)$

即：

$$
\arg\max_{\lambda, \eta}  \sum_{i=1}^N\left(-\log z(x^i, \lambda, \eta)+\sum_{t=1}^{T}\left[\lambda^{\top} \cdot f\left(y_{t-1}, y_{t}, x\right)+\eta^{\top} \cdot g\left(y_{t}, x\right)\right]\right)\\
= \arg\max_{\lambda,\eta} L(\lambda, \eta, x^i)
$$

采用梯度下降/梯度上升法进行优化即可。

$$
\nabla_{\lambda} L=\sum_{i=1}^{N} \sum_{t=1}^{T}\left(f\left(y_{t-1}, y_{t} x^{(i)}\right)-\sum_{y_{t-1}} \sum_{y_{t}} P\left(y_{t-1}, y_{t}, x^{(i)}\right) \cdot f\left(y_{t-1}, y_{t}, x^{(i)}\right)\right)
$$


#### 解码 Decoding

解码采用维特比算法，与 HMM 部分的维特比算法相似。



