---
title: 教程|Huggingface.co 模型上传与使用
date: 2023-05-15
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
- AIGC
---

将本地模型训练完后，除了用 huggingface transformers 自带的 `push_to_hub` 功能之外，我们还可以实用 `git lfs` 等工具将模型上传到 huggingface.co 上。

本文对 huggingface 模型上传，下载进行介绍。模型上传成功后，我们就可以使用 transformers 的 `from_pretrained()` 功能加载我们的模型了，如：

```python
from transformers import AutoTokenizer, AutoModel
model_id = "kevinng77/unsup_bert_L3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

question = "周末玩什么？"
inputs = tokenzier(question, return_tensor="pt")
outputs = model(**inputs)
```

## HF 模型上传快速上手

### 1.  **创建账户与仓库** 

首先在 huggingface 上注册账户，而后点击用户头像选择添加新模型: `+ New Model`

同创建 github 仓库一样，选择创建一个用于储存模型的仓库，如 `kevinng77/text_to_sql_t5_distill`

![HF 用户界面示例](/assets/img/hf_co_model_note/image-20230618104618316.png =x300)

### 2. **链接到你创建的仓库** 

创建仓库后，在仓库主页下，点击 Train 旁边的按钮，选择 clone。

![HF 模型仓库界面示例](/assets/img/hf_co_model_note/image-20230618105000372.png)

点击 clone repository 后，复制对应的代码，并在本地上执行。

![点击 clone repository 后弹出界面](/assets/img/hf_co_model_note/image-20230618105310988.png =x300)

Huggingface 可以支持用 http 连接（使用该 http 连接时，上传文件需要输入用户和 Access Token），如：

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/kevinng77/text_to_sql_t5_distill

```

:::info

如果没有安装 git lfs，可以用 `sudo apt-get install git-lfs`

:::

也可以用 ssh 链接，用 ssh 的话，需要更换连接地址为：

```bash
git lfs install
git clone ssh://git@hf.co/kevinng77/text_to_sql_t5_distill
```

:::warning

用 ssh 时，需要在 huggingface 上传你的 ssh 公钥。

:::

### 3. commit 你的文件

在 `git add` 你的模型权重之前，请先 track 你要上传的大文件（一般为模型权重）：

```bash
git lfs track *.bin
```

完成后执行：

```bash
git add .
git commit -m "your comments"
```

### 4. 推送到 huggingface.co

:::tip

如果你选择通过 HTTP 连接到 huggignface，在 push 时候使用的密码，需要在 huggingface 上生成（settings->access token->generate access token），使用默认用户密码是登录不上去的。

:::

```bash
# git push origin 本机 branch:远端 branch
git push origin main:main
```

### 5. 添加 model card 等信息

model card 信息统一在 `READMD.md` 中配置，README 在 huggingface.co 网站上可以很方便地编辑。官方提供了 metadata UI 界面，可以轻松的配置模型的一些信息。

![metadata UI 示例](/assets/img/hf_co_model_note/image-20230618104108012.png)

以上配置的 metadata，会通过 `yaml` 的形式添加在 `README.md` 文件中。

如下图，如果我们想要配置 Hosted inference API 中的 examples，可以在 `README.md` 中手动添加一些 yaml 配置。

![Hosted inference API 示例](/assets/img/hf_co_model_note/image-20230618110257421.png)

添加的 yaml 配置示例：

```md
---
license: apache-2.0
language:
- en
pipeline_tag: text2text-generation
library_name: transformers
tags:
- text-generation-inference
widget:
- text: >
    Given a SQL table named 'price_data' with the following columns:
    
    Transaction_ID, Platform, Product_ID, User_ID, Transaction_Amount
    
    Construct a SQL query to answer the following question:
    
    Q: How many rows are there

  example_title: "How many rows are there?"
---
```

## 模型权重下载与使用

### 1. 使用模型

模型上传到 huggingface 之后，通过 transformers 的 `.from_pretrained` 可以直接拉取模型并使用。当然，huggingface.co 储存模型的本质是使用 `git lfs`，因此我们可以在 huggingface.co 上保存其他类型的模型，如 paddle, onnx, tensorflow 等等。至于加载非 huggingface 模型的方式，需要参考各个模型其对应的文档了。如使用 onnx 的话，参考 [HF: export to ONNX](https://huggingface.co/docs/transformers/serialization) 。以下为加载 Onnx 模型示例：

```bash
# transformers==4.29.1

from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification

onnx_model_path = "kevinng77/unsup_bert_L3"
tokenizer = AutoTokenizer.from_pretrained(onnx_model_path)

# 需要上传 .onnx 模型权重到 huggingface.co 上
onnx_model = ORTModelForSequenceClassification.from_pretrained(onnx_model_path)
onnx_pipe = pipeline(task="text-classification", model=onnx_model, tokenizer=tokenizer)
onnx_pipe("How many rows are there in the table?")
```

### 2. 私有权重

我们可以将模型仓库设置为 `private`，这样子别人就不能用我们的模型了。设置成 private 之后，我们需要在本地登录 huggingface cli：

```bash
huggingface-cli login
```

输入用户名以及 `Access Token`（注意，是 access token，不是登录密码）。登录成功后，在你的 python 文件中修改：

```python
private_model = AutoModel.from_pretrained(model_id, use_auth_token=True)
private_tokenizer = AutoTokenzier.from_pretrained(model_id, use_auth_token=True)
```

### 3. 下载权重到本地

由于网络问题，我们通常会遇到使用 `.from_pretrained()` 无法正常加载模型的问题。这时候可以将模型先下载到本地：

```bash
git lfs install
git clone https://huggingface.co/your_model_id
# 如 git clone https://huggingface.co/kevinng77/chat-table-flan-t5
```

而后通过 `.from_pretrained(模型绝对路径)` 的方式加载。

## 一些坑

1. 使用 vscode 终端时，可能出现各种网络问题（如 connection error , SSH 验证错误等），可以尝试使用终端 ssh（不用 vscode 提供的终端）。

2. 使用 ssh 时，可能出现 `GIT LFS Locking API Error`，可以使用通过暂时移除 lfs lock verify 来解决：

```bash
# Remote "origin" does not support the Git LFS locking API. Consider disabling it with:
git config lfs.https://hf.co/kevinng77/text_to_sql_t5_distill.git/info/lfs.locksverify false
```

3. 在通过 `git clone https://huggingface.co/xxx` 时，出现 `git-lfs filter-process failed` 错误。参考 https://github.com/git-lfs/git-lfs/issues/911：

   ```bash
   # Skip smudge - We'll download binary files later in a faster batch
   git lfs install --skip-smudge
   
   # Do git clone here
   # 这一步不会下载 lfs 文件
   git clone ...
   
   # Fetch all the binary files in the new clone
   # 单独下载 lfs 文件
   git lfs pull
   
   # Reinstate smudge
   git lfs install --force
   ```

   如果在 `git lfs pull` 遇到了 `git config credential.helper manager` 等 credential 问题，可以使用：

   ```bash
   git config credential.helper manager
   ```

   或者 

   ```bash
   git config --global credential.helper cache
   git config --global crendential.helper wincred
   ```

   