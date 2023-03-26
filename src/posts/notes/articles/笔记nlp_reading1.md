---
title: NLP|阅读随笔
date: 2022-02-24
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
mathjax: true
toc: true
comments: 笔记

---

> 内容覆盖深度学习初始化、标准化、正则等
> 关键词：BERT、Transformers、Xavier、LayerNorm、初始化

<!--more-->

## 深度学习的初始化

> 从任意的均值为 0、方差为 1/m 的分布 p(x) 中独立重复采样，使得参数矩阵正交、以保持输入输出模不变

深度学习常采用的 Xavier 初始化 $N(0, \frac 2{d_{in}+d_{out}})$ 。初始化方案为 bias 全 0，其他系数为 **正交矩阵** 。

从任意的均值为 0、方差为 1/n 的分布 $p(x)$ 中独立重复采样出来的 $n×n$ 矩阵，都接近正交矩阵。且 n 越大，近似程度越好。 **因此采样出来的正交矩阵的逆几乎等于转置。** 

正交矩阵的重要意义在于它在变换过程中保持了向量的模长不变。 **因此使用正交矩阵进行初始化，深度学习 $y=Wx+b$ 嵌套多，应保持输入输出模一致，避免开始就出现输出模为 0 或无穷大的情况。**  

当 m≥n 时，从任意的均值为 0、方差为 1/m 的分布 p(x) 中独立重复采样出来的 m×n 矩阵，近似满足$W^TW=I$。

考虑激活函数 x 小的时候 $tanh(x) \approx x$ ，因此 Xavier 直接适用于 tanh。使用 relu 会有一半元素被置零，因此需要对 $W$ 乘上 $\sqrt 2$ 保持模长不变（也就是凯明初始化）。

 **参考：** 

苏剑林. (Jan. 16, 2020). 《从几何视角来理解模型参数的初始化策略 》[Blog post]. Retrieved from https://kexue.fm/archives/7180

## Lipschitz 约束与 L2

> Lipschitz 约束可以解释 L2 正则加强模型泛化能力

 **扰动敏感** 会降低模型稳定性，对于参数与输入我们希望模型稳定：$f_{w+\Delta w}(x) \approx f_x(x)\approx f_w(x+\Delta x)$

Lipschitz 约束的思想是，当 $x1,x2$ 相近时，模型对应的输出也应该是相近的：$\left\|f_{w}\left(x_{1}\right)-f_{w}\left(x_{2}\right)\right\| \leq C(w) \cdot\left\|x_{1}-x_{2}\right\|$ 其中我们 **希望 $C(w)$ 越小越好** ，这样才是好模型。

 考虑 $f_w()$ 为全连接层，那么 L 约束表示为：$\left\|\frac{\partial f}{\partial y} W\left(x_{1}-x_{2}\right)\right\| \leq C(W, b) \cdot\left\|x_{1}-x_{2}\right\|$ 要对左边的导数项进行约束，必须使用导数有边界的激活函数，如 relu/tanh/sigmoid 等。对于 relu 导数为 1，公式可以简化为 $\left\|W \Delta x \right\| \leq C\cdot\left\|\Delta x\right\|$

通过 Frobenius 范数与柯西不等式，有：$\|W x\| \leq\|W\|_{F} \cdot\|x\|$ 其中 $\|W\|_{F}=\sqrt{\sum_{i, j} w_{i j}^{2}}$ ， **所以 $C$ 可以取 $\|W\|_{F}$。**  （此处 Frobenius 范数是一个粗糙的条件，准去额应该是谱范数）

要令 C 最小，直接加入 loss，得到：$\operatorname{loss}=\operatorname{loss}\left(y, f_{w}(x)\right)+\lambda\|W\|_{F}^{2}$   **因此 L2 正则化与 L 约束是等价的。L2 加强模型泛华能力也得到了证明。** 

> 其他参考原文：谱范数、主特征根、幂迭代 

 **谱范数‖W‖2 等于 W⊤W 的最大特征根（主特征根）的平方根，如果 W 是方阵，那么‖W‖2 等于 W 的最大的特征根绝对值。** 

参考：

苏剑林. (Oct. 07, 2018). 《深度学习中的 Lipschitz 约束：泛化与生成模型 》[Blog post]. Retrieved from https://spaces.ac.cn/archives/6051
[《Spectral Norm Regularization for Improving the Generalizability of Deep Learning》](https://arxiv.org/abs/1705.10941)一文已经做了多个实验，表明“谱正则化”在多个任务上都能提升模型性能。

## Transformer

> 截尾正态分布初始化时考虑调整方差

初始化采样分布考虑 截尾正态分布，限制采样区间在 $[a,b]$ 中 a=μ−2σ,b=μ+2σ。根据采样，实际的方差为 $γ\sigma^2$。

$$
 \gamma=\frac{\int_{-2}^{2} e^{-x^{2} / 2} x^{2} d x}{\int_{-2}^{2} e^{-x^{2} / 2} d x}=0.7737413 \ldots
$$

> 以保持输入输出二阶原点矩出发，可以求得 Xavier 初始化、HE 初始化。

推导初始化方法的思想是尽量让输入输出具有同样的均值和方差。

事实上，只要每层的输入输出的二阶（原点）矩能稳定在适当的范围内，那么在反向传播的时候，模型每层的梯度也都保持在原点的一定范围中，不会爆炸也不会消失，所以这个模型基本上就可以稳定训练。
假设输入输出即参数期望都为 0；假设输入层二阶矩为 1，输出层的二阶矩为：

$$
\mathbb{E}\left[y_{j}^{2}\right]=\sum_{i} \mathbb{E}\left[x_{i}^{2}\right] \mathbb{E}\left[w_{i, j}^{2}\right]=m \mathbb{E}\left[w_{i, j}^{2}\right]
$$

令 $\mathbb{E}\left[y_{j}^{2}\right]$ 为 1 需要 $\mathbb{E}\left[w_{i, j}^{2}\right] = 1/m$ 。所以参数方差为 $1/m$ ，$m$ 为输入的维度/节点数（也就是 Xavier 初始化）。

> 可以通过微调激活函数，来实现输入输出二阶矩相同

对于 RELU，使得二阶矩不变的方差参数方差为 $2/m$ （HE 初始化）。

对于 sigmoid, tanh，如果要令初始化二阶矩不变，可以考虑微调激活函数。假设采用“均值为 0、方差为 1/m”的初始化参数，若输入方差为 1，则输出的二阶矩为：

$$
\int_{-\infty}^{\infty} \frac{e^{-x^{2} / 2}}{\sqrt{2 \pi}} \operatorname{sigmoid}(x)^{2} d x=0.2933790 \ldots
$$

因此为了保持输入输出方差都为 1，将激活函数输出除上 $\sqrt {0.29..}$ 就行。

SELU 

$$
f(x)=\lambda \left\{\begin{array}{ll}
a\left(e^{x}-1\right) & \text { if }(x<0) \\
x & \text { if }(0 \leq x)
\end{array}\right.
$$

可以看做 ELU 根据上面思路微调后的形式，即 $SELU=\lambda \times ELU$。

> 目前标准化似乎有直接除标准差，不中心化的趋势

transformer 中的 layerNorm 模型：

$$
y_{i, j, k}=\frac{x_{i, j, k}-\mu_{i, j}}{\sqrt{\sigma_{i, j}^{2}+\epsilon}} \times \gamma_{k}+\beta_{k}, \quad \mu_{i, j}=\frac{1}{d} \sum_{k=1}^{d} x_{i, j, k}, \quad \sigma_{i, j}^{2}=\frac{1}{d} \sum_{k=1}^{d}\left(x_{i, j, k}-\mu_{i, j}\right)^{2}
$$

被 T5 使用的 RMS Norm ：

$$
y_{i, j, k}=\frac{x_{i, j, k}}{\sqrt{\sigma_{i, j}^{2}+\epsilon}} \times \gamma_{k}, \quad \sigma_{i, j}^{2}=\frac{1}{d} \sum_{k=1}^{d} x_{i, j, k}^{2}
$$

引用苏神原文的一段话：一个直观的猜测是，center 操作，类似于全连接层的 bias 项，储存到的是关于数据的一种先验分布信息，而把这种先验分布信息直接储存在模型中，反而可能会导致模型的迁移能力下降。所以 T5 不仅去掉了 Layer Normalization 的 center 操作，它把每一层的 bias 项也都去掉了。

> NTK 参数化也能保持二阶矩不变

使用均值为 0，方差为 1 的分布初始化参数，将模型的输出直接除以 $\sqrt m$ 以保持二阶矩 $y_{j}=b_{j}+\frac{1}{\sqrt{m}} \sum_{i} x_{i} w_{i, j}$ 

两个维度为 d 的向量 q,k 如果都采样自均值为 0，方差为 1 的分布。他们内积的二阶矩和方差都为 d：

$$
\begin{aligned}
\mathbb{E}\left[(\boldsymbol{q} \cdot \boldsymbol{k})^{2}\right] &=\mathbb{E}\left[\left(\sum_{i=1}^{d} q_{i} k_{i}\right)^{2}\right]=\mathbb{E}\left[\left(\sum_{i} q_{i} k_{i}\right)\left(\sum_{j} q_{j} k_{j}\right)\right] \\
&=\mathbb{E}\left[\sum_{i, j}\left(q_{i} q_{j}\right)\left(k_{i} k_{j}\right)\right]=\sum_{i, j} \mathbb{E}\left[q_{i} q_{j}\right] \mathbb{E}\left[k_{i} k_{j}\right] \\
&=\sum_{i} \mathbb{E}\left[q_{i}^{2}\right] \mathbb{E}\left[k_{i}^{2}\right]=d
\end{aligned}
$$

所以之后没有除 $\sqrt d$ softmax 的范围大概在 $(e^{-3\sqrt d}, e^{3\sqrt d} )$，8 头 MHA 的话 d=64。明显数值太大/小，带来梯度消失。

此处似乎与 LayerNorm 相似，都对输出除以 $d_{model}$ 维度上的标准差。

> post norm 稳定了前向传播的方差，但是消弱了残差本身与易于训练的有点。，通常要使用 warmup 和小学习率才能使他收敛。

 **Post Norm**  即 transformer 和 bert 所使用的设计表示为：

$$
x_{t+1}=\operatorname{Norm}\left(x_{t}+F_{t}\left(x_{t}\right)\right)
$$

假设 $x_t$ 与 $F_t$ 方差均为 1，那么 LN 相当于除上 $\sqrt 2$。

要验证梯度是否会消失，可以通过展开第 $l$ 层的输出来判断：

$$
\begin{aligned}
x_{l} &=\frac{x_{l-1}}{\sqrt{2}}+\frac{F_{l-1}\left(x_{l-1}\right)}{\sqrt{2}} \\
&=\frac{x_{l-2}}{2}+\frac{F_{l-2}\left(x_{l-2}\right)}{2}+\frac{F_{l-1}\left(x_{l-1}\right)}{\sqrt{2}} \\
&=\cdots \\
&=\frac{x_{0}}{2^{l / 2}}+\frac{F_{0}\left(x_{0}\right)}{2^{l / 2}}+\frac{F_{1}\left(x_{1}\right)}{2^{(l-1) / 2}}+\frac{F_{2}\left(x_{2}\right)}{2^{(l-2) / 2}}+\cdots+\frac{F_{l-1}\left(x_{l-1}\right)}{2^{1 / 2}}
\end{aligned}
$$

明显前面的信息权重有点小。因此有人提出了  **Pre Norm**  如 GPT-2：

$$
x_{t+1}=x_{t}+F_{t}\left(\operatorname{Norm}\left(x_{t}\right)\right)
$$

因此 $l$ 层的方差近似于 $l$ ，对 $l$ 层输出展开为：

$$
x_{l}\approx x_{0}+F_{0}\left(x_{0}+d_0 \right)+F_{1}\left(x_{1} / \sqrt{2} + d_1\right)+\cdots+F_{l-1}\left(x_{l-1} / \sqrt{l} + d_l\right)
$$

 **参考：** 

苏剑林. (Aug. 17, 2021). 《浅谈 Transformer 的初始化、参数化与标准化 》[Blog post]. Retrieved from https://kexue.fm/archives/8620 苏剑林. (Aug. 17, 2021). 《浅谈 Transformer 的初始化、参数化与标准化 》[Blog post]. Retrieved from https://kexue.fm/archives/8620
[《Root Mean Square Layer Normalization》](https://arxiv.org/abs/1910.07467)
[《Do Transformer Modifications Transfer Across Implementations and Applications?》](https://arxiv.org/abs/2102.11972)
[《Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks》](https://arxiv.org/abs/2002.10444)
[《ReZero is All You Need: Fast Convergence at Large Depth》](https://arxiv.org/abs/2003.04887)
[《Fixup Initialization: Residual Learning Without Normalization》](https://arxiv.org/abs/1901.09321)

## BERT 的一些细节

> BERT 的 Post Norm 削弱了残差链接的效果，可能导致梯度消失；

梯度消失对于 finetune 可能是好事：假设底层的 bert 权重保留的是语言基础信息，高层保留更多任务相关信息，那么我们可能希望较少底层参数的变动。 **所以 post norm 模型的 finetune 效果会比 pre norm 好嘛？** 

Post Norm 的方式稳定了前向传播时的数值范围，还是有一定作用的。

目前主流 NLP 优化器为 Adam 及其变种，包含了动量和二阶矩矫正：

$$
\Delta \theta=-\eta \frac{\mathbb{E}_{t}\left[g_{t}\right]}{\sqrt{\mathbb{E}_{t}\left[g_{t}^{2}\right]}}
$$

猜测这种矫正梯度的方式 **可能** 了 Post Norm 的问题。 

> 梯度的大小放映了输出对于输入的依赖，不适用 warmup 可能会导致使用 post norm 的模型无法收敛。

控制梯度大小能够保持输出对于输入的依赖，如 SKIPNET 能够缓解 VAE 的后验塌陷问题。
对于 warmup 为何有效，有一种 **直观上的解释：（无证明）** 由于 postnorm 导致的梯度消失，靠近输出层的参数学习更快。一开始采用太大的学习率，高层参数将会快速收敛。由于 Adam 的矫正，此时的梯度中包含了很多的噪声，导致中低层参数学习方向错误。

> BERT 初始化采用小标准差分布，在保持 PostNorm 的情况下环节梯度消失

BERT 采用标准差为 0.2 的截断正太分布。实际标准差约为 0.176 （因为是截断正太分布）。该标准差对于 Xavier （$\frac 1{\sqrt 768}\approx 0.0361$）来说偏小。

根据 [上文 Post Norm](#transformer) 部分的推理：

$$
\begin{aligned}
x_{l}
&=\frac{x_{0}}{D^{l / 2}}+\frac{F_{0}\left(x_{0}\right)}{D^{l / 2}}+\frac{F_{1}\left(x_{1}\right)}{D^{(l-1) / 2}}+\frac{F_{2}\left(x_{2}\right)}{D^{(l-2) / 2}}+\cdots+\frac{F_{l-1}\left(x_{l-1}\right)}{D^{1 / 2}}
\end{aligned}
$$

若 post norm 内部的方差保持在 $D=1$ 左右，那么残差在传递过程中就能保持一定的权重。

> MLM 多加了 Dense

 **一种直观上的解释是：** 额外的 Dense 层减少 BERT 输出层中包含的 MLM 任务相关信息。

越靠近输出的层，学到的 task-Specified 知识更多。类似的，许多对比学习框架也会在训练时候添加额外的 2-3 层全连接层，并且在训练完成后丢弃他们。

 **另一种解释：**  一个现象是：最后一个 LN 层的 gamma 值会明显偏大。联系到上文中讨论的 BERT 采用 0.2 标准差（该标准差相对小）。如果直接乘 Embedding 预测分布的话，logits 太小导致 softmax 分布过于均匀。
如果直接调整 BERT 模型最后一层的 LN，那么模型看起来就不那么一致且优雅了，因此采用一个预训练完就丢弃的 DENSE + LN 是很好的选择。

参考：

苏剑林. (Nov. 08, 2021). 《模型优化漫谈：BERT 的初始标准差为什么是 0.02？ 》[Blog post]. Retrieved from https://kexue.fm/archives/8747
[《ZerO Initialization: Initializing Residual Networks with only Zeros and Ones》](https://arxiv.org/abs/2110.12661)
[《RealFormer：把残差转移到 Attention 矩阵上面去》](https://kexue.fm/archives/8027)

> 补充信息：BERT 的 MLM 层采用了 weight tying

BERT MLM 实际是对 token 进行掩码，所以掩码的对象可能是 `##ing`

BERT MLM 伪代码，接在 BERT MODEL 之后：

```python
## self.transform 部分
# 输入 hidden_states 维度 [*, len_seq, hidden_size]
hidden_states = self.dense(hidden_states)  
# dense: [hidden_size, hidden_size]
hidden_states = self.act_fn(hidden_states)
hidden_states = self.LayerNorm(hidden_states)


## self.decoder 部分
hidden_states = nn.Linear(hidden_size, vocab_size)(hidden_state) + bias(vocab_size)
# hidden_states: [len_seq, vocab_size]
```

其中 `self.decoder` 中的 `nn.Linear` 维度为 `hidden_size, vocab_size` 与 `input_embedding` 共享权重（但 bias 不共享）。

