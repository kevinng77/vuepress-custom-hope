---
title: DIFFUSION 系列笔记 | 扩散模型加速推理
date: 2023-12-16
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- CV
- Diffusion
- AIGC
---

## CM，LCM，LCM-Lora 

对于 Consistency Model，Latent Consistency Model 及 LCM-LoRA 的原理解读，十分推荐这篇文章：

https://wrong.wang/blog/20231111-consistency-is-all-you-need/

具体细节建议参考上面推荐的文章链接，以下对大致思路进行总结：

- Consistency Model 基于扩散模型，增加了一个推导约束：每个样本到噪声的加噪轨迹上的每个点都可以通过一个函数 $f(x_t, t)$ 映射回轨迹的起点。同时 CM 也对模型训练时的损失、sample 方案等进行了改动，以允许我们使用 2-4 个 step 来生成高质量图片。

![img = x400](/assets/img/speed_sd/image-20231216210637171.png)

+ LCM 在 SD 基础上进行蒸馏，在蒸馏过程中加入了 CM 中的 Consistency 约束。官方的用词是 `distill`：

  > “LCMs can be distilled from any pre-trained Stable Diffusion (SD) in only 4,000 training steps，

  但经过比较 stabilityai/stable-diffusion-xl-base-1.0 和 latent-consistency/lcm-sdxl 的 UNET 参数配置。 LCM 蒸馏似乎并未改变模型的大小。经过蒸馏后，我们可以使用 2-4 个 step 来生成媲美教师模型的图片。

- LCM-Lora：因为 LCM 也可以看做一种 finetune，因此我们也可以通过 lora 的方式进行高效参数微调，来蒸馏出一个 LCM-Lora。使用任何 SD 模型配合该 LCM-Lora，均能实现 2-4 step 的高质量成图。

::: important

参考 [Performing inference with LCM](https://huggingface.co/docs/diffusers/main/en/using-diffusers/lcm)。`guidance_scale` 对于 LCM-Lora，建议设置为 0（也可以试试 1-2）。对于 LCM 建议取值范围 `[3., 13.]`

:::

笔者测试了官方提供的 [Real-Time Latent Consistency Model SDv1.5](https://huggingface.co/spaces/latent-consistency/Real-Time-LCM-Text-to-Image-Lora-SD1.5)，将其中的 LCMScheduler 改为 DPMSolverMultistepScheduler，似乎效果更好（从 CM 算法角度看，应该是 LCM Scheduler 效果更好？）。但在本地尝试 SDXL LCM Lora 时，` LCMScheduler ` 的效果还是会好于其他 schedular 的。

## SDXL Turbo

参考 [技术报告： Adversarial Diffusion Distillation](https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1701197769659/adversarial_diffusion_distillation.pdf) 和 [huggingface diffusors 代码](https://huggingface.co/docs/diffusers/using-diffusers/sdxl_turbo)。

对于原理解析，可以参考文章：

https://zhuanlan.zhihu.com/p/669353808

SDXL Turbo 通过蒸馏 SDXL，根据官方描述，在 A100 上，生成一张 `512 * 512` 图片只需要 207ms，其中 UNET 推理用了 67 ms。

以下为大致训练方式：

![image-20231216222100682](/assets/img/speed_sd/image-20231216222100682.png =x400)

其中 ADD-student 的输入由 $x_0$ 经过 forward diffusion process 生成，而 DM-teacher 的输入由 ADD-Student 的输出经过  forward diffusion process 生成。

与 LCM 类似，SDXL-turbo 似乎也能像 LCM 一样出一个 LORA 版。

## 参考

https://wrong.wang/blog/20231111-consistency-is-all-you-need/

[Consistency Models](https://arxiv.org/pdf/2303.01469.pdf)

[Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/pdf/2310.04378.pdf)

[LCM-LoRA: A Universal Stable-Diffusion Acceleration Module](https://arxiv.org/pdf/2311.05556.pdf) 

