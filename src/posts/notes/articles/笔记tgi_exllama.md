---
title: TGI + exllama llama 量化部署方案
date: 2023-07-29
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
- AIGC
---

本文对 Text generation inference + exllama 的 LLaMa 量化服务方案进行单卡 4090 部署测试。

上期内容：[vllm vs TGI 部署 llama v2 7B 踩坑笔记](https://zhuanlan.zhihu.com/p/645732302)

在上期中我们提到了 TGI 和 vllm 的对比测试，在使用 vllm 和 TGI 对 float16 模型进行部署后，我们能够在单卡 4090 上达到 3.5+ request/秒的吞吐量。

就在几天前 TGI 优化了 exllama，基于之前量化 LLaMa 的测试，exllama 能在控制精度损失的情况下，将模型的推理速度提升。

以下参考 TGI 的官方手册对采用 AUTOGPTQ 量化后的 LLaMa v2 gptq4 权重进行部署:

```bash
docker run --rm --name tgi \
    --runtime=nvidia \
    --gpus all \
    -p 5001:5001 \
    -v /home/kevin/models:/models \
    ghcr.io/huggingface/text-generation-inference:1.0.0 \
    --model-id /models/llama2-7b-chat-gptq-int4 \
    --hostname 0.0.0.0 \
    --port 5001 \
    --max-concurrent-requests 256  \
    --quantize gptq \
    --trust-remote-code \
    --max-batch-total-tokens 30000 \
    --sharded false \
    --max-input-length 1024 \
    --validation-workers 4
```

其中，`llama2-7b-chat-gptq-int4` 量化采用 AUTOGPTQ 提供的示例量化代码进行量化，量化数据集选择 wikitext：

```bash
# git clone AUTOGPTQ 仓库后进入 `examples/quantization` 文件夹
# 修改以下 pretrained_model_dir 和 quantized_model_dir 选择用 Llama-2-7b-chat-hf 量化
python basic_usage_wikitext2.py
```

当然 TGI 和 GPTQ-for-LLaMa 也提供了 llama 量化脚本，但是对 llama v2 进行 GPTQ 量化时，AUTOGPTQ 的损失总是比较小， **量化后的模型输出也更稳定一些** ，原因未知。

目前 TGI 版本（1.0.0）对本地加载 exllama 模型仍有不少问题，如果遇到了 weight gptq_bits not found 的话，在模型文件夹下加入一个 safetensor，其中储存好 gptq_bits 和 group 就行。具体 exllama 权重加载逻辑可以看 `utils.weight.Wight`。

```python
import torch
from safetensors.torch import save_file

tensors = {
    "gptq_bits": torch.tensor(4),
    "gptq_groupsize": torch.tensor(128)
}
save_file(tensors, "/home/kevin/models/llama2-7b-chat-gptq-int4/gptq_config.safetensors")
```

目前 TGI 中对量化权重的处理方法不是很兼容 AUTOGPTQ 等 GPTQ 采用的数据储存方式，但有几个 PR 中已经在对此优化。

## TGI + EXLLAMA 测试

部署后，发送单一的请求进行速度测试：

```bash
###
POST http://127.0.0.1:5001/generate
Content-Type: application/json

{
    "inputs": "Once a upon time,",
    "parameters":{"max_new_tokens":100,"tempareture":0.6}
}
```

发送请求后 853ms 得到预测结果，平均 117.23 tokens/s。比 TGI + AUTOGPTQ （约 80 tokens/s）快。但还是不如 exllama 官方的推理速度（140+ tokens/s）。

通过 vllm 提供的 server benchmark 文件 `benchmark/benchmark_serving.py`  进行测试，

- 测试数据集：ShareGPT_V3_unfiltered_cleaned_split.json
- num prompt: 100 （随机从 ShareGPT 提供的用户和 GPT 对话数据当中，筛选 100 个问题进行测试）
- 默认设备为：单卡 4090 + inter i9-13900K。（采用 3070 测试的数据有标注）
- request 间隔: 每个 request 发送的间隔。

|                    | request 间隔（秒） | Throughtput (request/s) | average speed (tokens/s) | lowest speed (tokens/s) |
| ------------------ | ------------------ | ----------------------- | ------------------------ | ----------------------- |
| vllm               | 1                  | 0.95                    | 51                       | 39.4                    |
| vllm               | 0.5                | 1.66                    | 44.96                    | 29.41                   |
| vllm               | 0.25               | 2.48                    | 37.6                     | 24.05                   |
| vllm               | 0.05               | 3.24                    | 26.31                    | 4.13                    |
| TGI float16        | 1                  | 0.96                    | 80.15                    | 40.91                   |
| TGI float16        | 0.5                | 1.81                    | 74.62                    | 32.97                   |
| TGI float16        | 0.25               | 2.67                    | 59.36                    | 22.59                   |
| TGI float16        | 0.05               | 3.6                     | 37.39                    | 4.12                    |
| TGI EXLLAMA        | 1                  | 1.01                    | 131.87                   | 70.47                   |
| TGI EXLLAMA        | 0.5                | 1.86                    | 97.28                    | 44.22                   |
| TGI EXLLAMA        | 0.25               | 2.91                    | 59.70                    | 16.66                   |
| TGI EXLLAMA        | 0.02               | 4.89                    | 41.41                    | 14.88                   |
| TGI Exllama (3070) | 1                  | 0.42                    | 8.58                     | 0.72                    |
| TGI Exllama (3070) | 0.25               | 0.35                    | 2.39                     | 0.16                    |
| TGI Exllama (3070) | 0.02               | 0.43                    | 2.61                     | 0.19                    |

TGI + exllama 使得我们能够在一些小显存上部署模型，如 3070 (8GB)。期待 TGI，vllm 等部署服务对量化部署 llm 的持续优化，让私有部署模型有更好的体验。



