---
title: Instruction Tuning 后时代的模型笔记（二）
date: 2023-03-25
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
- AIGC
---

Alpaca，ChatGLM 6B 等模型的效果可以接受，下文总结部分笔记，为训练自定义小型化（7B）模型提供点知识储备。包括模型论文 LLAMA, PaLM, BLOOM, BLOOMZ-mT

<!--more-->

###  **LLAMA** 

LLaMA: Open and Efficient Foundation Language Models

论文的重点在于预训练，虽然也尝试了使用 instruction tuning 进行模型测试，但介绍部分并不多。

 **数据：** 约 1.4 T token 预训练，多语言但就是没有中文；

 **模型：** 

- 经典的大模型 Pre-normalization，采用 RMSNorm normalizing。使用 SwiGLU 激活函数，使用 ROPE。
- 模型规模有：7B, 13B, 33B, 65B

 **训练：** 

- 预训练：2048 A100 GPU 21 天

![img](https://picx.zhimg.com/80/v2-7817c721628b82eeaf46f2cd3436362e_1440w.png?source=d16d100b)



 **结果：** 

LLAMA 对标 GPT-3, PALM, OPT 等预训练模型。根据论文中各种指标表格，直观上来看，在 0-shot 以及 few-shot 效果上， **13B 的 LLAMA 与 GPT-3 可比，7B 的 LLAMA 不会落后 GPT-3 太多。** 

###  **PaLM** 

论文： **Scaling Language Modeling with Pathways**  

论文主要以研究预训练模型为主，但部分 instruction tuning 模型会与 PaLM 作比较，因此记录一些关注点：

 **模型架构上：** 

- SwiGLU 激活函数，采用 RoPE，共享 input, output embedding，所有 layer 不用  biases，更改 Transformer Block 中 Layernorm 的并行方式，使用 multi-query attention 取代 multi_head attention（仅将 Q 映射到 [k, h]，K,V 保持 [1, h]）

 **训练** ：

- 预训练 780 billion tokens
- 针对训练过程中，loss 爆炸的情况，采用回滚的方式：在 loss 爆炸点前 100 steps 开始，跳过 200 到 500 个 data batches，这样使得训练更加稳定

结果：总体看来 540 B 的 PaLM 在大部分任务上式新 SOTA

![img](https://pic1.zhimg.com/80/v2-7eaf758246c61d6efbba74c7ddb5a1ee_1440w.png?source=d16d100b)

同时 PaLM 62B 的效果与 GPT-3 相近。

![img](https://pic1.zhimg.com/80/v2-2b63c2e776d91396cfe5ed040ba9c471_1440w.png?source=d16d100b)



###  **Bloom** 

论文：BLOOM: A 176B-Parameter Open-Access Multilingual Language Model

Bloom 偏向于研究预训练的文章，文章将较大比重放在了多语言能力的实验上，文中的 Bloom 和 BLOOMz 两个模型也都开源。

 **模型：**  Vocab size 比 LLAMA 等其他模型大很多。位置编码采用了 ALiBi Positional Embeddings。

![img](https://picx.zhimg.com/80/v2-02d9ad6320c588eb1bb2c6eeb1bb2f61_1440w.png?source=d16d100b)



 **数据：** 预训练：498 个 huggingface 数据集，1.6 TB 文本(涵盖 46 种语言和 13 中编程语言)；BLOOMz 在多语言多任务数据集上额外进行了训练。

 **效果：** 

superGLUE 的评分图：

![img](https://picx.zhimg.com/80/v2-e70699a70a2063ccec31478a310aa25d_1440w.png?source=d16d100b)



文中还有其他评分任务，总体看来， BLOOM 效果好于 OPT。 **不同规格的 BLOOM 效果差距还是很大的。** 

![img](https://picx.zhimg.com/80/v2-df64ca91877f42fba19ecf421eb33f6c_1440w.png?source=d16d100b)



###  **Bloomz/mT0** 

论文：Crosslingual Generalization through Multitask Finetuning

在 BLOOM 和 mT5 上实验了 Multitask prompted finetuning 的效果。

模型：实验的都是多语言模型，文中使用了 BLOOM 和 mT5

数据：仍然采用公开数据集

![img](https://picx.zhimg.com/80/v2-18f861ab27ba5a008df73f73a274cba6_1440w.png?source=d16d100b)



prompt finetuning 采用的 prompt 也被分成三种形式进行测试：

![img](https://picx.zhimg.com/80/v2-9b8c14eaf80df62241e92632b1c3d476_1440w.png?source=d16d100b)



 **结果：** 

对照上图文中对比了三种模型的结果：BLOOMZ-P3 （prompt 数据为 P3 数据），BLOOMZ（prompt 数据为 xP3 数据），BLOOMZ-MT（prompt 数据为 xP3mt 数据）。

![img](https://pic1.zhimg.com/80/v2-ac598e6954b3b62f293ff7c75b962ba1_1440w.png?source=d16d100b)



论文中没有找到直接与 GPT 对比的实验，但根据下面的 code 续写成绩来看，BLOOM 系列和 GPT 差距应该不小。

![img](https://picx.zhimg.com/80/v2-343aa555e421ffcdee3dd85541095a06_1440w.png?source=d16d100b =x500)



###  **SELF-INSTRUCT** 

论文：Aligning Language Model with Self Generated Instructions

![self-instrcut 生成数据流程图](https://pic1.zhimg.com/80/v2-712c8d81e1e26485936877bfa520bb24_1440w.png?source=d16d100b)





 **大致数据生成流程：** 

1. 首先人工提供好 175 个 [instruct, input, output] 或 [instruct, null, output] 形式的数据。
2. 通过 LLM 的 ICL 能力，随机提取 6 个人工数据和 2 个，自动生成 2 个 instruction。
3. 通过 few shot （19 个负样本+12 个正样本）的方式，使用 GPT-3 来判断生成的 instruction 是否属于分类任务。
4. 通过 instruct 生成 input 以及 output，可以分为两个类别：
5. 对于非 classification 类型 instruction，先生成 input，而后生成 output。
6. 对于 classification 类型 instruction 生成 output，而后再试 input。
7. 通过 few-shot 的样本来控制 output 或者 input 的生成顺序，如要先生成 input，那么 prompt 可能如： Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn’t require additional input, you can generate the output directly.  Task: Which exercises are best for reducing belly fat at home?  Output:  - Lying Leg Raises  - Leg In And Out - Plank  - Side Plank - Sit-ups
8. 筛选：
9. 当生成的 instruction 与已有的任意 instruction 的 ROUGE-L 小于 0.7。
10. 句子中不包含 image， picture 等非语言内容的信息。
11. 过滤掉相同 input，但是不同 output 的案例。
12. 数据分布可视化：通过 Berkeley Neural Parser 查看数据类型分布。

 **数据：** 82k sample，52k instruction

论文中基于 ROUGE-L 进行的结果对比不太公平，毕竟数据集是冲着低 ROUGE-L 去做的。还是参考下图的人工评测结果好点。 self-instruct 后 GPT-3 效果能够接近 InstructGPT-001，同时在 self-instruct 数据集上训练的效果会好于在公开数据集上训练的效果：

![img](https://pic1.zhimg.com/80/v2-a682858cf0c48988729cdb77129e08ae_1440w.png?source=d16d100b)





[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) 中稍微修改了 self-instruct 生成数据的方式。一共使用 52K instruction 即 sample 对 LLAMA 进行训练，花费 3 hours on 8 80GB A100s。

[BELLE](https://github.com/LianjiaTech/BELLE) 基于 BLOOMz-mt 进行了中文微调，根据论文实验的结果看来，想要达到 ALPACA 类似的成绩，可能有点悬。