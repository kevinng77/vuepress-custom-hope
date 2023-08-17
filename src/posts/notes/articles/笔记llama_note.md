---
title: LLaMa 零散笔记
date: 2023-08-05
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
- AIGC
---

## llama note

### LLaMa 系列模型

baichuan，qwen 等与 llama 架构类似。不同点在于中文的这几个模型对此表进行了扩充，预训练方式不同，instruction tuning prompt template 不同，baichuan，qwen 分别采用 `w_proj` 和 `c_proj` 来代替 hf llama 官方的 `k,q,v_proj`。因此除了 lora 训练时需要映射一下位置，GPTQ 也需要做一下调整。

lora 有些人直接给设置成对 `q_proj, k_proj, v_proj, W_proj` 等等一系列不同模型采用的权重名称，似乎也不是不行。

### SwiGLU

官方实现方式为：

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act] 
    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
```

因此，尽管 llama 的 config 文件中，写了 hidden_act 为 silu，但是实际上用的的确时 SwiGLU。

$$
Silu = \sigma (x) * x
$$

SwiGLU 融合了 GLU 的思想，结合 LLaMa 中 FFN 没有 bias，因此可以写成：

$$
SwiGLU = Silu(xW)\times (xV)
$$

其中 $W,V$ 分别表示 `gate_proj` 和 `up_proj`

### 训练

LLaMa 出了名的，很多仓库做好了适配，没必要去自己根据 huggingface 写一个（当然，除非有很好的改进想法，如优化 loss 计算方式等）。现阶段完全采用 CasualLM 的训练优化方案了。

::: warning

很重要的一点，HF 的模型在计算 loss 时候，会自动对 label 进行 shift，以此计算 next token prediction 的 loss。因此我们输入中的 label 不需要进行提前的 shift。

此外 可以通过 ignore token 来进行多轮对话训练（参考 fschat 的代码）

:::

单卡 7B-lora 训练可以用：

+ [llama-efficient-tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning)

多卡多轮对话训练：

+ FastChat

