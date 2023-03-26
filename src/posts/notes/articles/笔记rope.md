---
title: RoPE 旋转位置编码
date: 2023-03-19
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
---

## RoPE 旋转位置编码

对于向量 q，使用 RoPE 添加位置编码后，用复数可以表示为：$qe^{im\theta}$，其中 $\theta_i=10000^{-2i/d}$ 与经典的 Transformer 绝对位置编码相同。通过将向量以复数方式推理， RoPE 巧妙地实现了以添加绝对位置编码的方式，在 attention 中计算了相对位置信息。

<!--more-->

RoPE 命名联系到了复数乘法的几何意义：对于复数  $z = a + bi$，他可以表示为复平面上的向量。其中 x 轴为实部，y 轴为虚部。根据向量与 x 轴的夹角 $\phi$，我们可以将向量表示为 $z = L(\cos \phi + i\sin \phi)$ ，其中 L 为向量模长。因此，向量的乘法变为了：

$$
z_1z_2= L_1L_2(\cos(\phi_1+\phi_2)+\sin(\phi_1+\phi_2))\tag 1
$$

所以，复数乘法也可以看作在复平面上的向量旋转及转换。

#### 大致思路

对于两个二维向量 $q, k$ 的内积，如果通过复数向量内积的方式来计算 $q,k$ 内积的话，那么：

$$
\langle q,k\rangle = Re[qk^*]\tag2
$$

> 引用苏老师的解释：两个二维向量的内积，等于把它们当复数看时，一个复数与另一个复数的共轭的乘积实部。

我们对 $q,k$ 分别乘以 $e^{in\theta},e^{im\theta}$，其中 $n,m$ 表示绝对位置信息:

$$
\langle q_ne^{in\theta},k_me^{in\theta}\rangle = Re[qe^{in\theta}(ke^{im\theta})^*]
=Re[qke^{i(n-m)\theta}]\tag3
$$

因此两个向量的内积也能表示相对位置关系 $n-m$ 了。根据欧拉公式：

$$
e^{x+iy}=e^x[\cos(y)+i\sin(y)]\tag4
$$

我们对 $q_ne^{in\theta}$ 进行拆解：

$$
\begin{aligned}
q_ne^{in\theta} &= (x+yi)[\cos(n\theta)+i\sin(n\theta)] \\&= (x \cos(n\theta)-y\sin(n\theta))+i(x\sin(n\theta) + y\cos(n\theta))
\end{aligned}\tag5
$$

将上式对应到二维向量的话就是：

$$
\left(\begin{array}{l}
x \\
y
\end{array}\right) \rightarrow \left(\begin{array}{c}
x \cos n \theta-y \sin n \theta \\
x \sin n \theta+y \cos n \theta
\end{array}\right)=\left(\begin{array}{l}
x \\
y
\end{array}\right) \cos n \theta+\left(\begin{array}{c}
-y \\
x
\end{array}\right) \sin n \theta\tag6
$$

其中 $\rightarrow$ 表示 $(5)$ 式中乘 $e^{in\theta}$ 的操作。因此，我们如果通过 $6$ 式的方式，来为 $q,k$ 向量内积添加上相对位置信息了。

#### 参考代码实现

#### RoFormer

在 RoFormer 中，只在传统 Multi-Head Attention 中添加一步，对 k, q, v 进行 `RotaryPositionEmbedding` 转换即可：

```python
class RotaryPositionEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2, dtype=paddle.get_default_dtype()) / dim))
        t = paddle.arange(max_position_embeddings, dtype=paddle.get_default_dtype())
        freqs = paddle.matmul(t.unsqueeze(1), inv_freq.unsqueeze(0))
        self.register_buffer("sin", freqs.sin(), persistable=False)
        self.register_buffer("cos", freqs.cos(), persistable=False)

    def forward(self, x, offset=0):
        # x shape [batch_size, num_heads, seqlen, head_dim]
        seqlen = paddle.shape(x)[-2]
        sin, cos = (
            self.sin[offset : offset + seqlen, :],
            self.cos[offset : offset + seqlen, :],
        )
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # [cos_nθ, -sin_nθ] [x1]
        # [sin_nθ,  cos_nθ] [x2]
        # => [x1 * cos_nθ - x2 * sin_nθ, x1 * sin_nθ + x2 * cos_nθ]
        return paddle.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).flatten(-2, -1)
```

#### GlobalPointer

GlobalPointer 中也使用了 RoPE。

```python
class GlobalPointer(nn.Layer):
    def __init__(self, hidden_size, heads, head_size=64, RoPE=True, tril_mask=True, max_length=512):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense1 = nn.Linear(hidden_size, head_size * 2)
        self.dense2 = nn.Linear(head_size * 2, heads * 2)
        if RoPE:
            self.rotary = RotaryPositionEmbedding(head_size, max_length)

    def forward(self, inputs, attention_mask=None):
        inputs = self.dense1(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE 编码
        if self.RoPE:
            qw, kw = self.rotary(qw), self.rotary(kw)

        # 计算内积
        logits = paddle.einsum("bmd,bnd->bmn", qw, kw) / self.head_size**0.5
        bias = paddle.transpose(self.dense2(inputs), [0, 2, 1]) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        # 排除 padding
        attn_mask = 1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
        logits = logits - attn_mask * 1e12

        # # 排除下三角
        if self.tril_mask:
            mask = paddle.tril(paddle.ones_like(logits), diagonal=-1)

            logits = logits - mask * 1e12

        return logits
```





## 参考

[让研究人员绞尽脑汁的 Transformer 位置编码](https://spaces.ac.cn/archives/8130)

[Transformer 升级之路：2、博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265)