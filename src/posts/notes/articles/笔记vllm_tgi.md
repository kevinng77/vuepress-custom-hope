---
title: vllm vs TGI 踩坑笔记
date: 2023-07-27
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
- AIGC
---

LLM 高并发部署是个难题，具备高吞吐量的服务，能够让用户有更好的体验（比如模型生成文字速度提升，用户排队时间缩短）。本文对 vllm 和 TGI 两个开源方案进行了实践测试，并整理了一些部署的坑。

测试环境：单卡 4090 + i9-13900K。限制于设备条件，本文仅对单卡部署 llama v2 7B 模型进行了测试。

 **小结：** TGI (0.9.3) 优于 vllm (v0.1.2)。最新版本的 TGI 在加入了 PagedAttention 之后，吞吐量和 vllm 差不多。

##  **vllm** 

github: https://github.com/vllm-project/vllm/tree/main

###  **安装** 

根据官方指南安装，可能出现各种问题（网络、依赖问题等）。笔者最终采用了以下 dockerfile 进行构建：

```bash
ARG CUDA_VERSION="11.8.0"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"

# Base NVidia CUDA Ubuntu image
FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION AS base

EXPOSE 22/tcp
EXPOSE 8000/tcp

USER root
# Install Python plus openssh, which is our minimum set of required packages.
# Install useful command line utility software
ARG APTPKGS="zsh sudo wget tmux nvtop vim neovim curl rsync less"
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip python3-venv && \
    apt-get install -y --no-install-recommends openssh-server openssh-client git git-lfs && \
    python3 -m pip install --upgrade pip && \
    apt-get install -y --no-install-recommends $APTPKGS && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda/bin:${PATH}"

ARG USERNAME=vllm
ENV USERNAME=$USERNAME
ARG VOLUME=/workspace
ENV VOLUME=$VOLUME

# Create user, change shell to ZSH, make a volume which they own
RUN useradd -m -u 1000 $USERNAME && \
    chsh -s /usr/bin/zsh $USERNAME && \
    mkdir -p "$VOLUME" && \
    chown $USERNAME:$USERNAME "$VOLUME" && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-docker-users

USER $USERNAME
ENV HOME=/home/$USERNAME
ENV PATH=$HOME/.local/bin:$PATH
WORKDIR $HOME

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX;8.9;9.0"

# 你可以考虑添加 pip 清华源或其他国内 pip 源
RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    pip3 install -e . && \
    pip3 install ray == 2.5.1 && \ 
    pip3 install git+https://github.com/huggingface/transformers accelerate==0.21.0 && \
    pip3 cache purge
```

以上 dockerfile 参考了 Thebloke 的 vllm 安装推荐（[link](https://github.com/vllm-project/vllm/issues/537)）。如果构建 docker image 过程遇到了问题，也可以直接 pull 笔者构建好的 vllm (v0.1.2) 镜像：

```bash
docker pull kevinng77/vllm
```

###  **快速开始** 

1. 除了安装 vllm 外，我们需要 vllm 官方 github 上的一些测试代码：

```bash
git clone https://github.com/vllm-project/vllm.git

docker run -it  --runtime=nvidia --gpus=all --net=host --name=vllm -v ./vllm:/workspace/vllm kevinng77/vllm /bin/bash
```

\2. 启动模型服务

我们使用 LLaMa v2 进行测试部署：

```bash
# cd /workspace/vllm
python3 -m vllm.entrypoints.api_server \
    --model /models/Llama-2-7b-chat-hf --swap-space 16 \
    --disable-log-requests --host 0.0.0.0 --port 8080 --max-num-seqs 256
```

更多参数可以在 vllm/engine/arg_utils.py 找到，其中比较重要的有：

- max-num-seqs：默认 256，当 max-num-seqs 比较小时，较迟接收到的 request 会进入 waiting_list，直到前面有 request 结束后再被添加进生成队列。当 max-num-seqs 太大时，会出现一部分 request 在生成了 3-4 个 tokens 之后，被加入到 waiting_list（有些用户出现生成到一半卡住的情况）。过大或过小的 max-num-seqs 都会影响用户体验。
- max-num-batched-tokens： **很重要的配置** ，比如你配置了 max-num-batched-tokens=1000 那么你大概能在一个 batch 里面处理 10 条平均长度约为 100 tokens 的 inputs。max-num-batched-tokens 应尽可能大，来充分发挥 continuous batching 的优势。不过似乎（对于 TGI 是这样，vllm 不太确定），在提供 HF 模型时，该 max-num-batched-tokens 能够被自动推导出来。

部署后，发送 post 请求到 http://{host}:{port}/generate ，body 为 ：

```json
{
    "prompt": "Once a upon time,",
    "max_tokens": output_len,
    "stream": false,
    "top_p": 1.0
    // 其他参数
}
```

vllm 也提供了 OpenAI-compatible API server，vllm 调用了 fastchat 的 conversation template 来构建对话输入的 prompt，但 v0.1.2 的 vllm 与最新版的 fastchat 有冲突，为了保证使用 llama v2 时用上对应的 prompt template，可以手动修改以下 entrypoints.openai.api_server 中对 get_conversation_template 的引入方式（[link](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py#L33)）。修改后执行以下代码启动：

```bash
python3 -m vllm.entrypoints.openai.api_server \
        --model /models/Llama-2-7b-chat-hf \
        --disable-log-requests --host 0.0.0.0 --port 5001 --max-num-seqs 20 --served-model-name llama-2
```

###  **一些碎碎念** 

- 采用 stream: true 进行 steaming 请求时，vllm 提供的默认 API 每次的回复并不是新预测的单词，而是目前所有已预测的全部文本内容（history_tokens + new_tokens），这导致回复的内容中包含很多冗余的信息。
- 当前 vllm 版本（v0.1.2）加载 *.safetensor 模型时存在问题，请尽量加载 *.bin 格式的模型。对于文件夹下同时存放 bin 和 safetensor 权重时，vllm 优先加载 .bin 权重。
- vllm 的 OpenAI-compatible API server 依赖 fschat 提供 prompt template，由于 LLM 更新进度快，如果遇到模型 prompt template 在 fschat 中未找到的情况（通常报错为 keyError），可以重新安装下 fschat 和 transformers。

关于 vllm 的 paged attention 机制，网上已经有不少解析，比如可以参考 [NLP（十七）：从 FlashAttention 到 PagedAttention, 如何进一步优化 Attention 性能](https://zhuanlan.zhihu.com/p/638468472)。除了 paged attention 之外，vllm 中的 Continuous batching 和 Scheduling 机制也挺有意思。从接收 request，到返回回复，大致的过程如下：

1. vllm 接收到 request 之后，会发放 request_uuid，并将 request 分配到 running, swap, waiting 三个队列当中。（参考  vllm.core.scheduler.Scheduler._schedule ）
2. 根据用户等待的时间进行 request 优先级排序。从 running 和 swap 队列中选择优先级高的 request 来生成对应的回复，由于 decoding 阶段，每次前项传播只预测一个 token，因此 vllm 在进行完一次前项传播（即 one decoding iteration）之后，会返回所有新生成的 tokens 保存在每个 request_uuid 下。（参考 vllm.engine.llm_engine.LLMEngine.step）
3. 如果 request 完成了所有的 decoding 步骤，那么将其移除，并返回结果给用户。
4. 更新 running, swap 和 waiting 的 request。
5. 循环执行 2,3,4。

##  **Text Generation Inference** 

TGI github: https://github.com/huggingface/text-generation-inference

TGI 支持了：

- 和 vllm 类似的 continuous batching
- 支持了  [flash-attention](https://github.com/HazyResearch/flash-attention) 和 [Paged Attention](https://github.com/vllm-project/vllm)。
- 支持了 [Safetensors](https://github.com/huggingface/safetensors) 权重加载。（目前版本的 vllm 在加载部分模型的 safetensors 有问题（比如 llama-2-7B-chat）。
- TGI 支持部署 GPTQ 模型服务，这使得我们可以在单卡上部署拥有 continous batching 功能的，更大的模型。
- 支持采用 Tensor Parallelism 部署多 GPU 服务，模型水印等其他功能

###  **安装** 

如果想直接部署模型的话，建议通过 docker 安装，省去不必要的环境问题。目前 TGI 只提供了 0.9.3 的：

```bash
docker pull ghcr.io/huggingface/text-generation-inference:0.9.3
```

如果要进行本地测试，可以通过源码安装（以下在 ubuntu 上安装）：

1. 依赖安装

```bash
# 如果没有网络加速的话，建议添加 pip 清华源或其他国内 pip 源
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
apt-get install cargo  pkg-config git
```

\2. 下载 protoc

```bash
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

\3. 如果没有网络加速的话，建议修改 cargo 源。有网络加速可略过。

```bash
# vim ~/.cargo/config
[source.crates-io]
registry = "https://github.com/rust-lang/crates.io-index"

replace-with = 'tuna'

[source.tuna]
registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"

[net]
git-fetch-with-cli=true
```

\4. TGI 根目录下执行安装：

```bash
BUILD_EXTENSIONS=True make install # Install repository and HF/transformer fork with CUDA kernels
```

\5. 安装成功，添加环境变量到 .bashrc 中 export PATH=/root/.cargo/bin:$PATH

\6. 执行 text-generation-launcher --help ，有输出表示安装成功。

###  **快速开始** 

将 Llama-2-7b-chat-hf 部署成服务：

```bash
# 建议将模型下载到本地，而后挂载到 docker 中，避免在 docker 中重复下载。
docker run --rm \
	--name tgi \
	--runtime=nvidia \
	--gpus all \
	-p 5001:5001 \
	-v /home/kevin/models:/data \
	ghcr.io/huggingface/text-generation-inference:0.9.3 \
	--model-id /data/Llama-2-7b-chat-hf \
	--hostname 0.0.0.0 \
	--port 5001 \
	--dtype float16 \
	--sharded false \
	--max-batch-total-tokens 30000 # 24GB 显卡加载 7B llama v2 可参考
```

可以通过 text-generation-launcher --help 查看到可配置参数，相对 vllm 来说，TGI 在服务部署上的参数配置更丰富一些，其中比较重要的有：

- model-id：模型 path 或者 hf.co 的 model_id。
- revision：模型版本，比如 hf.co 里仓库的 branch 名称。
- quantize：TGI 支持使用 GPTQ 来部署模型。
- max-concurrent-requests：当服务处于高峰期时，对于更多新的请求，系统会直接返回拒绝请求，（比如返回服务器正忙，请稍后重试），而不是将新请求加入到 waiting list 当中。改配置可以有效环节后台服务压力。默认为 128。
- max-total-tokens：相当于模型的 max-tokens
- max-batch-total-tokens： **非常重要的参数，他极大影响了服务的吞吐量。** 该参数与 vllm 中的 max-num-batched-tokens 类似。比如你配置了 max-num-batched-tokens=1000：那么你大概能在一个 batch 里面处理 10 条平均长度约为 100 tokens 的 inputs。 **对于传入的 HF 模型，TGI 会自动推理该参数的最大上限** ，如果你加载了一个 7B 的模型到 24GB 显存的显卡当中，你会看到你的显存占用基本上被用满了，而不是只占用了 13GB（7B 模型常见显存占用），那是因为 TGI 根据 max-batch-total-tokens 提前对显存进行规划和占用。 **但对于量化模型，该参数需要自己设定，** 设定时可以根据显存占用情况，推测改参数的上限。

##  **Serving 测试** 

上文中分别采用 vllm 以及 TGI 部署了 Llama-2-7b-chat-hf 模型，以下对两者进行测试：

###  **vllm**   **benchmark**   **测试** 

部署后通过 benchmark/benchmark_serving.py 进行测试：

- 测试数据集：ShareGPT_V3_unfiltered_cleaned_split.json
- num prompt:  100 （随机从 ShareGPT 提供的用户和 GPT 对话数据当中，筛选 100 个问题进行测试）
- 设备：单卡 4090 + inter i9-13900K
- request 间隔: 每个 request 发送的间隔。

|           | request 间隔（秒） | Throughput (request/s) | average speed tokens/s | lowest speed tokens/s |
| --------- | ------------------ | ---------------------- | ---------------------- | --------------------- |
| vllm      | 1                  | 0.95                   | 51                     | 39.4                  |
| vllm      | 0.5                | 1.66                   | 44.96                  | 29.41                 |
| vllm      | 0.25               | 2.48                   | 37.6                   | 24.05                 |
| vllm      | 0.05               | 3.24                   | 26.31                  | 4.13                  |
| TGI（HF） | 1                  | 0.96                   | 70.67                  | 37.15                 |
| TGI（HF） | 0.5                | 1.68                   | 62                     | 31.97                 |
| TGI（HF） | 0.25               | 2.71                   | 56.47                  | 25.59                 |
| TGI（HF） | 0.05               | 3.66                   | 37.39                  | 4.71                  |

1. 通过以上数据，可以看出 TGI 更新后比 vllm 的吞吐量优秀一些。
2. 由于 continuous batchinglowest speed。
3. 测试时候 random seed 是一样的。因此输入样本是一样的。预测是，TGI 和 VLLM 的 tempareture 都设置为 0，均采用 sampling。

通过 vllm 提供的 benchmark/benchmark_throughput ，在 4090 测试得到结果：平均输出速度 1973.84 tokens/s，Throughput 为 4.13 requests/s，相当于 240.7 seq/min。（benchmark_throughput 相当于以上测试中 request 间隔为 0）

###  **JMeter 模拟** 

采用 Apache JMeter 进行请求模拟测试。模拟 100 个用户分别在 10 秒钟内开始他们的聊天，持续 10 轮。结果如下：

![View result in Table 结果截图（根据 Latency 排序）](https://pic1.zhimg.com/80/v2-da49d3cab8579431a4e0a3e28cfdc661_1440w.png?source=d16d100b)



![Aggregate Report 结果截图](https://pic1.zhimg.com/80/v2-5e67960f7accb27ae3725338798e79a1_1440w.png?source=d16d100b)

实验结果表示，每个用户发送消息后，接收到 LLM 回复的延迟在 152 ms 以下（接收到 第一个 token 的延迟）。平均每个对话的回复速度在 33-50 tokens/s。因此，使用 4090 单卡，可以部署一个供约 100 人正常使用的 7B LLM 模型。

JMeter 模拟配置如下：

Thread Group：添加 Number of Threads = 100 个用户，所有用户在 Ramp-up period=10 秒内完成请求发送。 我们假设每个用户进行了 Loop count=10 轮对话。

- HTTP Request（sampler）
- constant timer = 2: 每个用户接受到 LLM 回复后，会在 2 秒后发送新的请求（模拟打字速度）。
- HTTP Header Manager: 添加 content-type=application/json，post 的 body 统一设置为：

```
{
    "messages":[
        {"role": "user", "content": "Once a upon time,"}
    ],
    "model":"llama-2",
    "temperature": 0.6,
    "stream": true,
    "max_tokens": 2000
}
```

- Listener 包括: 
- View results tree: 查看每个请求返回结果，确认 LLM 生成的回复是正确的。
- View results in Table: 查看 request 延迟时间最大值等数据。
- Aggregate Report: 查看平均请求时间等数据。

## 其他

除了 vllm，TGI 外，仍有不少 llm 服务部署仓库，如 lmdeploy 等。限制于设备条件，本文仅对单卡部署 7B 模型进行了测试。

Kevin 吴嘉文：LLaMa 量化部署 20 赞同 · 5 评论文章

在上期 LLaMa 量化文章中，我们讨论了 LLaMa 使用 GPTQ 量化的情况下，进行推理（batch_size=1）的速度可提高近 3 倍。但由于当 batch size 大时， GPTQ 的 batch inference 效率没有 fp16 高，本地测试采用 batch inference 可以达到的推理速度：

| type    | batch size | tokens/s |
| ------- | ---------- | -------- |
| exllama | 1          | 144.2    |
| exllama | 2          | 255.73   |
| exllama | 4          | 373.44   |
| exllama | 8          | 304      |
| fp16    | 1          | 52       |
| fp16    | 4          | 205      |
| fp16    | 8          | 390      |
| vllm    | 1          | 59       |
| vllm    | 4          | 222      |
| vllm    | 8          | 432      |

因此采用 TGI + GPTQ 的吞吐量可能不会有很大提升。目前 TGI 对 exllama 的支持还不是很好，等过段时间再来测试 TGI + exllama 的吞吐量。

