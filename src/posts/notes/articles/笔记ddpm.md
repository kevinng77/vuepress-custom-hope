---
title: Diffusion|DDPM 理解、数学、代码
date: 2022-11-28
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- CV
- Diffusion
- AIGC
mathjax: true
toc: true
comments: 笔记

---

## Diffusion

论文：Denoising Diffusion Probabilistic Models

参考[博客](https://kevinng77.github.io/posts/notes/articles/%E7%AC%94%E8%AE%B0ddpm.html)；参考 paddle 版本代码： [aistudio 实践链接](https://aistudio.baidu.com/projectdetail/5219878?contributionType=1&sUid=902220&shared=1&ts=1692196876315)

该文章主要对 DDPM 论文中的公式进行小白推导，并根据 `ppdiffuser` 进行 DDPM 探索。读者能够对论文中的大部分公式如何得来，用在了什么地方有初步的了解。

  **本文将包括以下部分：**  

1. DDPM 总览
2. Forward process： 包括论文中公式推导，以及其在 ppdiffusor 中代码参考
3. Reverse process： 包括论文中公式推导，以及其在 ppdiffusor 中代码参考
4. 优化目标推导：包括论文中公式推导，以及简单的伪代码描述
5. 探索与思考：通过打印，修改 ppdiffusor ddpm 代码，探索 DDPM 模型。

## DDPM 总览

扩散模型在 2015 年已经被提出（[Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)），而 DDPM 将扩散模型应用在了图像生成领域上。

DDPM 的大致思想是：用 AI 构建一个模型，相比于一步到位生成图像，我们让这个模型每次生成一小步，经过 T 步后，图像就完成了。

![img](https://pic2.zhimg.com/80/v2-0387c3ca424a7c3a9c72c7e5c2e2e03d_1440w.webp)

如上图，给定原图片 $x_0$，DDPM 考虑每次在图像 $x_{t-1}$ 上加入噪声，得到图像 $x_t$，在经过多次加噪后，$x_t$ 就几乎变成了噪声。

而后我们训练模型，使其能根据带有噪声的图片 $x_{t}$ 预测 $x_{t-1}$ （具体在 DDPM 中不直接对像素进行预测，该环节可以参考 Reverse process）


### Forward Process

给定原图片 $x_0$，Forward Process 的目标是生成 $x_1, x_2, ..., x_T$ 如上图。方式就是每一次在原图上加上随机的噪声 $\epsilon_t = N(0,1)$ 。论文中假设加噪过程符合分布 $q(x_t|x_{t-1}) =N (x_t; \sqrt {1-\beta_t} x_{t-1}, \beta_tI)$ ，因此我们可以通过以下公式来迭代生成。

$$
x_t = \sqrt\alpha_t x_{t-1} + \sqrt\beta_t\epsilon_t\tag 1
$$

其中 $\alpha_t = 1-\beta_t$。通过 $(1)$ 式不断套娃可以得到：

$$
x_t = \sqrt {\alpha_t...\alpha_1}x_0 +\underbrace {\sqrt {\alpha_t...\alpha_2\beta_1}\epsilon_1 + ... +\sqrt
{\alpha_t\beta_{t-1}}\epsilon_{t-1} + \sqrt\beta_t\epsilon_t}_{\text{ 噪声项 }}\tag 2
$$

因为 $\epsilon$ 为相互独立的正态分布，根据正态分布的叠加性， $(2)$ 式中的噪声项可视为均值为 0，方差为 $1-\alpha_t...\alpha_1$ 的分布。

$$
\begin{aligned}
&\alpha_t...\alpha_1 + \alpha_t...\alpha_2\beta_1 + ...+ \alpha_t\beta_{t-1}+\beta_t
\\ = & (\alpha_1+\beta_1)\alpha_t...\alpha_2 +...+ \alpha_t\beta_{t-1}+\beta_t \\ = &\alpha_{t}+\beta_t
\\=&1
\end{aligned}
$$

因此得出论文的前向扩散公式$(4)$:

$$
q(x_t|x_0) =N (x_t; \sqrt {\bar \alpha_t} x_0, (1-\bar\alpha_t)I),\bar \alpha_t=\prod_T\alpha_t \tag 4
$$

在该步骤中，$\beta$ 被设置成了不可学习参数，范围在 `[1e-4, 0.02]` 之间，随着时间步 `t` 线性变换，这也极大的简化了训练时的优化目标推理。此外，DDPM 设置了 `T=1000`，在加噪 1000 步之后，图像就完全变成了无信号的电视画面。

在 ppdiffusor 中，公式 $(4)$ 对应 `DDPMScheduler.add_noise()`。

如果你参考了 [DDPM 官方文档](https://github.com/hojonathanho/diffusion)，那么公式 $(4)$ 对应的是 `q_sample()` 函数。

## Reverse process

Reverse process 的目的是 能根据带有噪声的图片 $x_{t}$ 预测 $x_{t-1}$。这一步希望拟合的分布是 $p_\theta(x_{t-1}|x_t) = N (x_{t-1}; \mu _\theta (x_t, t), \Sigma_\theta (x_t, t))$。其中，作者假设方差项 $\Sigma_\theta (x_t, t) = \sigma_t^2=\beta_t$。（当然原论文中还提出了其他的方差项，我们不在此讨论）

首先我们能够通过 $(4)$ 推理得到（论文中的公式 $(15)$）：

$$
x_{0} = \frac 1{\sqrt {\bar \alpha_t}}(x_t-\sqrt{1-\bar\alpha_t}\epsilon_t) \tag 5
$$

因此图像采样过程可以定义为 $q(x_{t-1}|x_t)=q(x_{t-1}|x_t, x_0) = N (x_{t-1}; \mu _\theta (x_t, x_0), \sigma_t I)$，采样过程可以视为马尔科夫链，通过 $q(x_{t-1}|x_t, x_0)$ 以及贝叶斯定理，我们能够更好地写出 reverse process 的表达式 。

$$
\begin{aligned}
&q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) =\\


&\exp \left(-\frac{1}{2}\left(\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}\right) \mathbf{x}_{t-1}^2-\left(\frac{2 \sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t+\frac{2 \sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} \mathbf{x}_0\right) \mathbf{x}_{t-1}
+ C\left(\mathbf{x}_t, \mathbf{x}_0\right)\right)\right)
\end{aligned} \tag 6
$$

:::details 公式推导 公式推导

$$
\begin{aligned}
q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) &=q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0\right) \frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_0\right)}{q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)} \\
& \propto \exp \left(-\frac{1}{2}\left(\frac{\left(\mathbf{x}_t-\sqrt{\alpha_t} \mathbf{x}_{t-1}\right)^2}{\beta_t}+\frac{\left(\mathbf{x}_{t-1}-\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0\right)^2}{1-\bar{\alpha}_{t-1}}-\frac{\left(\mathbf{x}_t-\sqrt{\bar{\alpha}_t} \mathbf{x}_0\right)^2}{1-\bar{\alpha}_t}\right)\right) \\
&=\exp \left(-\frac{1}{2}\left(\frac{\mathbf{x}_t^2-2 \sqrt{\alpha_t} \mathbf{x}_t \mathbf{x}_{t-1}+\alpha_t \mathbf{x}_{t-1}^2}{\beta_t}+\frac{\mathbf{x}_{t-1}^2-2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 \mathbf{x}_{t-1}+\bar{\alpha}_{t-1} \mathbf{x}_0^2}{1-\bar{\alpha}_{t-1}}-\frac{\left(\mathbf{x}_t-\sqrt{\bar{\alpha}_t} \mathbf{x}_0\right)^2}{1-\bar{\alpha}_t}\right)\right) \\
&=\exp \left(-\frac{1}{2}\left(\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}\right) \mathbf{x}_{t-1}^2-\left(\frac{2 \sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t+\frac{2 \sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} \mathbf{x}_0\right) \mathbf{x}_{t-1}+C\left(\mathbf{x}_t, \mathbf{x}_0\right)\right)\right)
\end{aligned}
$$

:::

把上式对应到正态分布公式当中，可以得到论文中的公式 $(7)$

$$
\begin{aligned}
\sigma_t^2=\tilde{\beta}_t &=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot \beta_t \\
\tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t, \mathbf{x}_0\right) &=
\frac{\sqrt{\alpha_t}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t} \mathbf{x}_t+\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0
\end{aligned} \tag 7
$$

:::details 公式推导

$$
\begin{aligned}
\sigma_t^2=\tilde{\beta}_t &=1 /\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}\right)\\&=1 /\left(\frac{\alpha_t-\bar{\alpha}_t+\beta_t}{\beta_t\left(1-\bar{\alpha}_{t-1}\right)}\right)\\&=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot \beta_t \\
\tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t, \mathbf{x}_0\right) &=\left(\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t+\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} \mathbf{x}_0\right) /\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}\right) \\
&=\left(\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t+\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} \mathbf{x}_0\right) \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot \beta_t \\
&=\frac{\sqrt{\alpha_t}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t} \mathbf{x}_t+\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0
\end{aligned}
$$

:::

由于在采样过程中我们不知道真实的 $x_0$，所以用 $x_t$ 来预测 $x_0$ ，即本文公式 $(5)$ 。这样采样过程变为：

$$
\begin{aligned}
\hat x_{t-1}
&=\frac 1{\sqrt { \alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)) + \sigma_t z\\ &\text{where }z=N(0,I)
\end{aligned}\tag 8
$$

:::details 公式推导

$$
\begin{aligned}
\hat x_{t-1} &= \mu_\theta(x_t,x_0) + \sigma_tz\\
&= \frac{\sqrt{\alpha_t}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t} \mathbf{x}_t+\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0 + \sqrt { \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t}·z\\
&=\frac{\sqrt{\alpha_t}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t} \mathbf{x}_t
+\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t}(\frac {\sqrt {\bar \alpha_t}}(x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta) )+\sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t}·z\\
&= \frac 1 {\sqrt \alpha _t}x_t\left( \frac{\beta_t+\alpha_t-\bar\alpha_t}{1-\bar \alpha_t}  \right) + \frac {\beta_t}{\sqrt \alpha_t\sqrt {1-\bar\alpha_t}}\epsilon_\theta + \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t}\cdot z\\
&=\frac 1{\sqrt { \alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)) + \sigma_t z\\ &\text{where }z=N(0,I)
\end{aligned}
$$

:::

Reverse process 该部分对应的为 `ppdiffusor.DDPMScheduler.step`。DDPM 论文提供的 Tensorflow 版代码链接为 [link](https://github.com/hojonathanho/diffusion)。

- 上文公式 (5) 在官方代码中对应 `predict_start_from_noise`，在 paddle 中对应 `ppdiffusor.DDPMScheduler.step` 的

```python
pred_original_sample = (sample - beta_prod_t**(0.5) * model_output) / alpha_prod_t**(0.5)
```



- 上文公式 $(7)$ 在官方代码中对应 `q_posterior_mean_variance`，在 `ppdiffusor.DDPMScheduler.step` 对应。

```python
pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
```

细心的朋友们会发现官方给的代码中，sampling 方式分为：

$$
x_t\xrightarrow{model} \epsilon_\theta(x_t,t) \xrightarrow {\text{ 公式 }(5)}\hat x_0(x_t, \epsilon_\theta)  \xrightarrow {\text{ 公式 }(7)}\mu(x_t, \hat x_0),\beta_t\xrightarrow{sampling}x_{t-1}
$$

但其实这等价于：

$$
x_t\xrightarrow{\text{ 公式 } (8)} x_{t-1}
$$

- 上文公式 $(8)$

上文公式 $(8)$ 根据 $(5),(7)$ 推理得来，因此如果在 `ppdiffusor.DDPMScheduler.step` 中将采样过程全部替换为：

```python
prev_sample = (sample - model_output * self.betas[t]/(1-self.alphas_cumprod[t])  **0.5)/self.alphas[t]**  (
           0.5) + variance
```

那么结果会是一样的，我们将在之后的代码探索中尝试验证它。

## 优化目标

在训练过程中，我们只需要对每个 $t$ 步骤添加的噪声 $\epsilon_\theta$ 进行损失优化就行。以下两个角度出发都能够说明拟合 $\epsilon_\theta$ 是有效的。在部分版本的 DDPM 代码中，开可以看到作者们设置的 `pred_noise` 参数，用于选择模型的预测目标为噪声 $\epsilon_\theta$ 或者图像像素 $x_{t-1}$。

### 从论文的变分边界角度出发

我们的目标是获得 $x$ 的生成模型，因此可以优化：

$$
\begin{aligned}
\mathbb E[-\log p_\theta(x_0)] \le \mathbb E \left[-\log p_\theta(x_T) - \sum_{t\ge1}\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}   \right] =L_{VLB}
\end{aligned}\tag{11}
$$

:::details 公式推导

$$
\begin{aligned}
\mathbb E[-\log p_\theta(x_0)] &= \mathbb E\left[-\log p_\theta(x_0) \int p_\theta(x_{1:T})dx_{1:T}\right]  \\
&= \mathbb E \left[-\log \int p_\theta(x_{0:T})dx_{1:T}\right]\\
&= \mathbb E\left[-\log \int \frac {p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}q(x_{1:T}|x_0)dx_{1:T} \right ] \\
&\le \mathbb E\left[- \log\frac{p_{\theta(x_{0:T})}}{\int q(x_{1:T}|x_0)q(x_{1:T}|x_0)dx_{1:T}}\right] \\
&=\mathbb E\left[-\log \frac {p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}   \right]
\\ &= \mathbb E \left[-\log p_\theta(x_T) - \sum_{t\ge1}\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}   \right] =L_{VLB}
\end{aligned}
$$

:::

因此我们得到了论文中的公式 $(3)$，我们只需要对其中的变分边界进行优化即可：

$$
\begin{aligned}
L_{VLB}&= \mathbb{E}_q\left[\underbrace{D_{\mathrm{KL}}\left(q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| p\left(\mathbf{x}_T\right)\right)}_{L_T}+\sum_{t>1} \underbrace{D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\right)}_{L_{t-1}} \underbrace{-\log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}_{L_0}\right]
\end{aligned}\tag {12}
$$

:::details 公式推导

$$
\begin{aligned}
L_{VLB}&=\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}+\log \frac{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}\right] \\
&=\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=2}^T \log \left(\frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)} \cdot \frac{q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)}{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_0\right)}\right)+\log \frac{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}\right] \\
&=\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)}{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_0\right)}+\log \frac{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}\right] \\
&=\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}+\log \frac{q\left(\mathbf{x}_T \mid \mathbf{x}_0\right)}{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}+\log \frac{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}\right] \\
&=\mathbb{E}_q\left[\log \frac{q\left(\mathbf{x}_T \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_T\right)}+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}-\log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)\right]\\
&= \mathbb{E}_q\left[\underbrace{D_{\mathrm{KL}}\left(q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| p\left(\mathbf{x}_T\right)\right)}_{L_T}+\sum_{t>1} \underbrace{D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\right)}_{L_{t-1}} \underbrace{-\log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}_{L_0}\right]
\end{aligned}
$$

:::

因此我们得到了论文中的公式 $(5)$ ，通过以上式子可以看出 $L_T$ 部分为 forward process 分布，在我们之前的设定下是无法优化的，$L_0$ 是固定的噪声，因此我们可以优化 $L_{t-1}$ 项。因为我们在前面假设了 $q(x_{t-1}|x_t,x_0)$ 与 $p_\theta(x_{t-1}|x_t)$ 都服从正态分布，因此根据：


$$
\begin{aligned} K L\left(\mathcal{N}\left(\mu_{1}, \sigma_{1}^{2}\right) \| \mathcal{N}\left(\mu_{2}, \sigma_{2}^{2}\right)\right) &=\log \frac{\sigma_{2}}{\sigma_{1}}-\frac{1}{2}+\frac{\sigma_{1}^{2}+\left(\mu_{1}-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}
\end{aligned}\tag{13}
$$

:::details 公式推导

$$
\begin{aligned} K L\left(\mathcal{N}\left(\mu_{1}, \sigma_{1}^{2}\right) \| \mathcal{N}\left(\mu_{2}, \sigma_{2}^{2}\right)\right) &=\int_{x} \frac{1}{\sqrt{2 \pi} \sigma_{1}} e^{-\frac{\left(x-\mu_{1}\right)^{2}}{2 \sigma_{1}^{2}}} \log \frac{\frac{1}{\sqrt{2 \pi} \sigma_{1}} e^{-\frac{\left(x-\mu_{1}\right)^{2}}{2 \sigma_{1}^{2}}}}{\frac{1}{\sqrt{2 \pi} \sigma_{2}} e^{-\frac{\left(x-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}}} d x \\ &=\int_{x} \frac{1}{\sqrt{2 \pi} \sigma_{1}} e^{-\frac{\left(x-\mu_{1}\right)^{2}}{2 \sigma_{1}^{2}}}\left[\log \frac{\sigma_{2}}{\sigma_{1}}-\frac{\left(x-\mu_{1}\right)^{2}}{2 \sigma_{1}^{2}}+\frac{\left(x-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}\right] d x \\&=\log \frac{\sigma_{2}}{\sigma_{1}}-\frac{1}{2}+\frac{\sigma_{1}^{2}+\left(\mu_{1}-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}
\end{aligned}\tag{13}
$$

其中
    
$$
\begin{aligned}\log \frac{\sigma_{2}}{\sigma_{1}} \int_{x} \frac{1}{\sqrt{2 \pi} \sigma_{1}} e^{-\frac{\left(x-\mu_{1}\right)^{2}}{2 \sigma_{1}^{2}}} d x &=\log \frac{\sigma_{2}}{\sigma_{1}}\\
-\frac{1}{2 \sigma_{1}^{2}} \int_{x}\left(x-\mu_{1}\right)^{2} \frac{1}{\sqrt{2 \pi} \sigma_{1}} e^{-\frac{\left(x-\mu_{1}\right)^{2}}{2 \sigma_{1}^{2}}} d x&=-\frac{1}{2 \sigma_{1}^{2}} \sigma_{1}^{2}=-\frac{1}{2} \\
\frac{1}{2 \sigma_{2}^{2}} \int_{x}\left(x-\mu_{2}\right)^{2} \frac{1}{\sqrt{2 \pi} \sigma_{1}} e^{-\frac{\left(x-\mu_{1}\right)^{2}}{2 \sigma_{1}^{2}}} d x &=\frac{1}{2 \sigma_{2}^{2}} \int_{x}\left(x^{2}-2 \mu_{2} x+\mu_{2}^{2}\right) \frac{1}{\sqrt{2 \pi} \sigma_{1}} e^{-\frac{\left(x-\mu_{1}\right)^{2}}{2 \sigma_{1}^{2}}} d x\\ &=\frac{\sigma_{1}^{2}+\mu_{1}^{2}-2 \mu_{1} \mu_{2}+\mu_{2}^{2}}{2 \sigma_{2}^{2}}=\frac{\sigma_{1}^{2}+\left(\mu_{1}-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}
\end{aligned}\tag{14}
$$

:::

因此：

$$
\begin{aligned}
L_{t-1} &=
D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\right) \\&=D_{\mathrm{KL}}\left(\mathcal{N}\left(\mathbf{x}_{t-1} ; \tilde{\boldsymbol{\mu}}\left(\mathbf{x}_t, \mathbf{x}_0\right), \sigma_t^2 \mathbf{I}\right) \| \mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \sigma_t^2 \mathbf{I}\right)\right) \\
&=\frac{1}{2 \sigma_t^2}\left\|\tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t, \mathbf{x}_0\right)-\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)\right\|^2 + C
\end{aligned}\tag{15}
$$

所以我们得到了论文中的公式 $(8)$。

由于我们在前面假设了  $q(x_{t-1}|x_t,x_0)$ 与 $p_\theta(x_{t-1}|x_t)$ 的方差值相同，因此上式中 $C=0$。将公式 $(5)$ 带入，得到：

$$
\begin{aligned}
L_t &=\mathbb{E}\left[\frac{\left(1-\alpha_t\right)^2}{2 \alpha_t\left(1-\bar{\alpha}_t\right)\sigma^2}\left\|\boldsymbol{\epsilon}_t-\boldsymbol{\epsilon}_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_t, t\right)\right\|^2\right]
\end{aligned}\tag {16}
$$

:::details 公式推导

$$
\begin{aligned}
L_t &=\mathbb{E}\left[\frac{1}{2\sigma^2}\left\|\tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t, \mathbf{x}_0\right)-\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)\right\|^2\right] \\
&=\mathbb{E}\left[\frac{1}{2\sigma^2}\left\|\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_t\right)-\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right)\right\|^2\right] \\
&=\mathbb{E}\left[\frac{\left(1-\alpha_t\right)^2}{2 \alpha_t\left(1-\bar{\alpha}_t\right)\sigma^2}\left\|\boldsymbol{\epsilon}_t-\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right\|^2\right] \\
&=\mathbb{E}\left[\frac{\left(1-\alpha_t\right)^2}{2 \alpha_t\left(1-\bar{\alpha}_t\right)\sigma^2}\left\|\boldsymbol{\epsilon}_t-\boldsymbol{\epsilon}_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_t, t\right)\right\|^2\right]
\end{aligned}\tag {16}
$$

:::

因此我们得到了论文中的公式 $(12)$，在训练时直接对噪声 $\epsilon$ 进行优化即可，论文也提出在优化时，忽略公式 $(12)$ 前面的系数，效果更好。

### 从优化像素的角度出发

[生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼](https://spaces.ac.cn/archives/9119) 从优化像素的角度出发进行了推理，得到了与论文相似的优化函数形式。

大致思路是直接对图像进行优化：

$$
L = ||x_{t-1} - \hat x_{t-1}||^2\tag 9
$$

由于在预测时候我们不知道原先噪声，因此使用预测的噪声 $\epsilon_\theta$ 来预测图像 $\hat x_{t-1} = \frac 1{\sqrt {\alpha_t}}(x_t-\sqrt{1-\alpha}\epsilon_\theta(x_t,t))$ ，带入 $(9)$ 式得到：

$$
\begin{aligned}
L &= ||\frac 1{\sqrt {\alpha_t}}(x_t-\sqrt{1-\alpha_t}\epsilon) - \frac 1{\sqrt { \alpha_t}}(x_t-\sqrt{1-\alpha_t}\epsilon_\theta(x_t, t))||^2\\
&= \frac {1-\alpha_t}{\alpha_t}||\epsilon_t - \epsilon_\theta(x_t, t)  ||^2\\
&= \frac {1-\alpha_t}{\alpha_t}||\epsilon_t - \epsilon_\theta\left(\sqrt{\alpha_t} \mathbf{x}_0+\sqrt{1-{\alpha}_t} \boldsymbol{\epsilon}_t, t\right)  ||^2
\end{aligned}\tag {10}
$$


当然以上只是进行了大致流程概括，实际推理过程还需要考虑方差过大等细节问题，详细请参考 [生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼](https://spaces.ac.cn/archives/9119)。

### 模型优化过程

根据上述的公式，我们只需要建立模型，对噪声进行拟合即可。以下伪代码参考了 DDPM 论文提供的代码，展示 DDPM 优化过程逻辑：

```python
def train_losses(self, model, x_start, t):
    noise = torch.randn_like(x_start)
    x_noisy = self.q_sample(x_start, t, noise=noise)
    predicted_noise = model(x_noisy, t)
    loss = F.mse_loss(noise, predicted_noise)
    # 部分网友提到此处使用 mse 可能导致 loss 太小，在低精度训练情况下，模型先收敛后发散
    return loss
```

DDPM 论文中采用的 model 为 UNET（并做了一些优化配置），我们不展开讨论。其中 $t$ 为时间步。在真实训练中并非对一张图片的 1000 个时间布都进行学习，而是随机选取时间步

```python
for epoch in range(epochs):
    for step, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        batch_size = images.shape[0]
        images = images.to(device)
        
        # sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        
        loss = gaussian_diffusion.train_losses(model, images, t)
        
        if step % 200 == 0:
            print("Loss:", loss.item())
            
        loss.backward()
        optimizer.step()
```

## 探索与思考

为什么 DDPM 效果更好？

笔者猜想是否因为优化目标从图片像素便到了噪声，优化目标变小，更好拟合了？？此外，DDPM 相比于单步的 VAE 效果更好，可能因为：

> VAE 同样假设建模对象符合正态分布，对于微小变化来说，可以用正态分布足够近似地建模，类似于曲线在小范围内可以用直线近似，多步分解就有点像用分段线性函数拟合复杂曲线，因此理论上可以突破传统单步 VAE 的拟合能力限制。
> -- 引用来源 [生成扩散模型漫谈（二）：DDPM = 自回归式 VAE](https://spaces.ac.cn/archives/9152)

## 代码（torch 版本）

参考代码 [TF-DDPM](https://github.com/hojonathanho/diffusion)  [torch-DDPM ](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py):

其中函数分别以及对应的公式：

+ `q_sample` 对应本文公式 $(4)$： $x_t = \sqrt {\bar\alpha_t} x_0 + \sqrt {1 - \bar\alpha_t} \epsilon$.
+ `predict_start_from_noise` 对应本文公式 $(5)$: $x_{0} = \frac 1{\sqrt {\bar \alpha_t}}(x_t-\sqrt{1-\bar\alpha_t}\epsilon)$. 
+ `q_posterior_mean_variance` 对应本文公式 $(7)$: 


$$
\begin{aligned}
\tilde{\beta}_t &=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot \beta_t\\
\mu _t(x_t, x_0) &= \frac{\sqrt {\hat\alpha_{t-1}}\beta }{1-\hat\alpha_t}x_0 + \frac {\sqrt \alpha_t (1-\hat\alpha_{t-1})}{1-\hat\alpha_t}x_t
\end{aligned}
$$

+ `p_mean_variance` 对应本文公式 $(5)+(7)$.
+ `p_sample` 对应 `p_mean_variance`  + 本文公式 $(8)$.

细心的朋友们会发现官方给的代码中，sampling 方式分为：

$$
x_t\xrightarrow{model} \epsilon_\theta(x_t,t) \xrightarrow {\text{ 公式 }(5)}\hat x_0(x_t, \epsilon_\theta)  \xrightarrow {\text{ 公式 }(7)}\mu(x_t, \hat x_0),\beta_t\xrightarrow{sampling}x_{t-1}
$$

但其实这等价于：

$$
x_t\xrightarrow{\text{ 公式 } (8)} x_{t-1}
$$

  **经过测试，将 `p_sample` 部分的代码换成上面这个公式后，采样生成图片的结果是一样的。**  

模型方面 DDPM 采用了 UNET 作为 backbone，在传播过程中加入了三角函数位置编码，用于传递采样步骤 $t$ 的信息。在训练过程中，图像的像素被缩放到了 `[-1, 1]` 的区间进行模型学习，在预测编码的时候映射回到 `[0, 255]`。

此外论文中的 UNET 还加入了 attention 等操作，能够提高打榜分数，但如果采用基础的自编码器效果也是够好的。

### 训练过程

根据官方的代码，优化时直接对噪声进行优化，即：

```python
def train_losses(self, model, x_start, t):
    noise = torch.randn_like(x_start)
    x_noisy = self.q_sample(x_start, t, noise=noise)
    predicted_noise = model(x_noisy, t)
    loss = F.mse_loss(noise, predicted_noise)
    # 部分网友提到此处使用 mse 可能导致 loss 太小，在低精度训练情况下，模型先收敛后发散
    return loss

```

其中 $t$ 为时间步。在真实训练中并非对一张图片的 1000 个时间布都进行学习，而是随机选取时间步：

```python
for epoch in range(epochs):
    for step, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        batch_size = images.shape[0]
        images = images.to(device)
        
        # sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        
        loss = gaussian_diffusion.train_losses(model, images, t)
        
        if step % 200 == 0:
            print("Loss:", loss.item())
            
        loss.backward()
        optimizer.step()
```

## 代码（ppdiffuser 版本）

### 查看采样过程中的渐变图片

我们需要在 `DDPMScheduler.step` 中，将 `prev_sample` 打印出来，首先运行一个图片采样过程：

```python
import sys
sys.path.append("ppdiffusers")
sys.path.append("ppdiffusers/ppdiffusers")

from ppdiffusers import DDPMPipeline

# 加载模型和 scheduler
pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
pipe.scheduler.config.clip_sample =False
# 执行 pipeline 进行推理
output = pipe(seed=777)

images = output[0].images
image = images[0]
# 保存图片
all_images = output[1]  # 保存了所有预测过程中的 x_t
all_x_0 = output[2]  # 保存了所有预测过程中的 x_0，参考本文公式 5
image.show()
```

我们打印所有过程图片 $x_t$，看到了论文中描述的从噪声逐步采样到完成图片的过程：

```python
from matplotlib import pyplot as plt

plt.figure(figsize=(10,5))
count = 0
for i in range(1,1000,50):
    img = all_images[i][0].resize((64,64))
    count += 1
    plt.subplot(5,10,count)
    plt.imshow(img)
    plt.axis("off")
    
plt.show()
```

![image-20230816225157005](https://pic3.zhimg.com/80/v2-9bbb5b1bb04c27b0c9cb7a4916905ce6_1440w.webp)

接下来我们打印所有中间预测过的 $x_0$（参考本文公式 $(5)$），能够发现模型在一开始对 $x_0$ 仅停留在一个模糊的预测：比如颜色，大致位置，轮廓等。而后在面的过程中，对图片的一些细节才逐步有了刻画。

```python
plt.figure(figsize=(10,5))
count = 0
for i in range(1,1000,50):
    img = all_x_0[i][0].resize((64,64))
    count += 1
    plt.subplot(5,10,count)
    plt.imshow(img)
    plt.axis("off")
plt.show()
```

![image-20230816225220451](https://pic4.zhimg.com/80/v2-8fefe59ea7afbe74f38a0df6b2f93be7_1440w.webp)

验证将

$$
x_t\xrightarrow{model} \epsilon_\theta(x_t,t) \xrightarrow {\text{ 公式 }(5)}\hat x_0(x_t, \epsilon_\theta)  \xrightarrow {\text{ 公式 }(7)}\mu(x_t, \hat x_0),\beta_t\xrightarrow{sampling}x_{t-1}
$$

替换为：

$$
x_t\xrightarrow{\text{ 公式 } (8)} x_{t-1}
$$

后的结果（参考本文 Reverse process 部分）

将 `ppdiffusers/ppdiffusers/schedulers/DDPMScheduler.step` 中 `pred_prev_sample` 预测方式改为

```python
pred_prev_sample = (sample - model_output * self.betas[t]/(1-self.alphas_cumprod[t])  **0.5)/self.alphas[t]**  (
           0.5) + variance
```

得出与原先相近的图片。由于采样过程中存在对预测的 $x_0$ clip 的情况（见 `DDPMScheduler` 中的 `config.clip_sample` 参数）。因此两者在代码上来说，并不是完全等价的。这个影响在 DDIM （DENOISING DIFFUSION IMPLICIT MODELS）中会相对严重，笔者将在下一个笔记中一起来探讨 DDIM。

## 参考

[科学空间 - 生成扩散模型漫谈系列博客](https://spaces.ac.cn/search/%E7%94%9F%E6%88%90%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B/)

[小小将 - 扩散模型之 DDPM](https://zhuanlan.zhihu.com/p/563661713)

[DDPM 论文代码 TF 版](https://github.com/hojonathanho/diffusion)  

[DDPM Torch 版](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py)

