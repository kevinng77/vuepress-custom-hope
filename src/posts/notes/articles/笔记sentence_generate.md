---
title: Beam Search、cache 机制笔记
date: 2021-09-03
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
- 对话系统
mathjax: true
toc: true
comments: 笔记
---

> paddlenlp 源码中的 Beam Search、cache 机制笔记

<!--more-->

## Beam Search

### paddle beam search 实现分析

 **整体逻辑** 
模型执行生成代码： `model.generate( input_ids,...)` 
执行解码通用操作：

+ 获取相关输入数据，如 input_ids，bos_token_id，eos_token_id 等数据
+ 准备`cache` 相关数据，配置`logits_processors` 函数用于每个时间步的 logits 调整。（如重复字符惩罚等）

而后准备 `beam_search` 过程中需要的数据：

+ 使用`expand_inputs_for_generation` 将 系列的输入如`input_ds`等拓展为 `[batch_size *num_beam, len_seq]`维度，包括 `token_type_ids`，`position_ids`，`encoder_output` 等参数。
+ 准备 `beam_scorer` - 用来计算 beam search 分数

最后，进行 `beam_search` 解码

```python
# 初始化 beam_scores
while cur_len < max_length:
    # 输入格式处理
    # 模型前向传导，计算 logits
    # logits 调整与处理
    # 更新整体 score，选择 top K
```

####  **输入格式处理**  

由模型下的 `prepare_inputs_for_generation` 实现。可以在每个时间步的前向传导前，调整模型的输入。如一个 encoder-decoder 架构的模型使用`cache`时，我们在每个时间步需要更新 `cache` 并且将 decoder 的相关输入（如`decoder_input_ids`, `decoder_attention_mask`等）截取保留最后一个时间步的数据。
例：`paddlenlp.transformers.BartForConditionalGeneration`

```python
    def prepare_inputs_for_generation(self,
                                      decoder_input_ids,
                                      attention_mask=None,
                                      decoder_attention_mask=None,
                                      cache=None,
                                      use_cache=False,
                                      encoder_output=None,
                                      **kwargs):
        # cut decoder_input_ids if past is used
        if cache is not None:
            decoder_input_ids = decoder_input_ids[:, -1].unsqueeze(-1)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask[:, :,
                                                                -1, :].unsqueeze(
                                                                    2)

        return {
            "input_ids": None,
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_output,
            "decoder_attention_mask": decoder_attention_mask,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache
        }
```

#### logits 调整与处理

在执行 `outputs = self(**model_inputs)`前向传导后，在计算 `beam_score` 前针对输出的 logits 进行了调整。主要包括两部分：

+ 静态信息调整，如禁用某些固定的词或 token。模型通过重写`adjust_logits_during_generation`函数来实现。例 `paddlenlp.transformers.UNIMOLMHeadModel`：

```python
    def adjust_logits_during_generation(self, logits):
        # pre-process distribution
        logits[:, self.unimo.unk_token_id] = -1e9
        logits[:, self.unimo.pad_token_id] = -1e9
        logits[:, self.unimo.bos_token_id] = -1e9
        return logits
```

+ 动态信息调整，如根据当前生成句子长度修改 `eos_token` 的概率。对重复出现的 token 基于惩罚等。通过 `paddlenlp.transformers.generation_utils.get_logits_processor` 实现，当前支持的功能有最大最小长度控制，重复字符惩罚。 **想要实现更多功能的话需要直接修改该文件。** 

#### 计算 beam score

1.  **通过处理后的 logits 计算下一个 token 的概率，筛选出 top 候选。** 

一开始的时候，每个 batch 中有 `num_beams`个一样的输入，所以应该对 `beam_score` 配置掩码。

```python
beam_scores = paddle.zeros(
    (batch_size, num_beams), dtype=paddle.get_default_dtype())
beam_scores[:, 1:] = -1e9
beam_scores = paddle.reshape(beam_scores, [-1])
```

有了掩码，在后续的 broadcast 相加中，就不会出现同 batch 下多个相同语句的情况了：

```python
# next_scores [batch_size * num_beams, vocab_size]
next_scores = F.softmax(logits)
next_scores = paddle.log(next_scores)

next_scores = next_scores + beam_scores.unsqueeze(-1)
```

将候选序列重新 reshape 为 `[batch_size, num_beams * vocab_size]` ，然后取 top `2*beams`？# todo

```python
next_scores = next_scores.reshape(
    [batch_size, num_beams * vocab_size])

next_scores, next_tokens = paddle.topk(
    next_scores, 2 * num_beams, axis=1)

next_indices = next_tokens // vocab_size
# next_indices 得分高的，被选中的候选句子在 next_scores 中的 idx

next_tokens = next_tokens % vocab_size  
# 排除掉前面 n*vocab_size 个 token。
```

2.  **由 `beam_scorer` 计算新时间步的 beam search 输出。`beam_scorer.process`** 

上一步中，代码执行后得到了当前时间步下最后的 top `2*num_beams` 候选词。而在这步中，beam score 根据这些候选词，整理出全局最优的最多 `num_beams` 个 `beam hypothesis` 候选句子。具体整理思路比较特别，可参考 `paddlenlp.transformers.generation_utils 129-210` 行代码。

共有 `batch_size `个 `beam hypothesis `，若当前时间步的候选句子编码结束（eos token），那么就添加到 `beam hypothesis` 中。对于编码未结束的候选句子，挑选 top `num_beams` 个作为下一个时间步计算的 `beam_scores` (`shape[batch_size, num_beams]`)与 新生成的编码 相加。

beam_scores 的 group_size 干什么用？

+ 控制 beam_scores 的大小为 `[batch_size, group_size]` 

num_beam_hyps_to_keep 呢?

+ 每个 batch 中，beam search 结束后，返回 `num_beam_hyps_to_keep` 个结果。

3.  **根据是否使用 `cache`，更新 `token_type_ids`，`position_ids`，`attention_mask` 等储存在 `model_kwargs` 中的变量。** 

通过`paddlenlp.transformers.generation_utils.GenerationMixin.update_model_kwargs_for_generation`实现。该操作与 [ **输入格式处理**  ](#输入格式处理) 中的输入处理相似。尽量避免两个函数出现累赘操作。

#### 输出结果

最后由 `finalize()` 总结所有 beam search 候选句子并输出结果。

## 效率优化 - cache

#### cache 主要思想：

储存模型传导中以固定的变量，如 attention 中的 key、value、query 等

#### cache 整体逻辑框架：

对于 encoder decoder 的结构，cache 只能用在 decoder 解码部分。以下根据 paddle 源码进行描述：
对于每个时间步：
`prepare_inputs_for_generation`，若使用 cache 则对 input_ids 和对应的 attentionmask，token_type_ids，position_ids 进行裁剪。
`adjust_logits_during_generation` 修改一些静态 logits，如将 unk_token，pad_token 或禁用词的概率设置为 `-1e9
logits_processors` 修改一些动态的 logits，如对重复的 token 进行惩罚，根据 min_length 设置 eos_token 的大小。
对与每个时间步，都需要提取上一步的 cache 记录，然后计算当前步的结果。

```python
    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      token_type_ids,
                                      position_ids,
                                      attention_mask,
                                      use_cache=False,
                                      cache=None,
                                      **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)
            attention_mask = attention_mask[:, :, -1, :].unsqueeze(2)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache
        }
```

通过下面代码，可以看出储存 `k,v,q` 是 cache 机制的核心。

```python
def _prepare_qkv(self, query, key, value, cache=None):
    """当 cache 存在时，将 k,v 缓存下来。
        Cache = collections.namedtuple("Cache", ["k", "v"])
    	StaticCache = collections.namedtuple("StaticCache", ["k", "v"])
    """
    q = self.q_proj(query)
    q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
    q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

    if isinstance(cache, self.StaticCache):
        # for encoder-decoder attention in inference and has cached
        k, v = cache.k, cache.v
    else:
        k, v = self.compute_kv(key, value)

    if isinstance(cache, self.Cache):
        # for decoder self-attention in inference
        k = tensor.concat([cache.k, k], axis=2)
        v = tensor.concat([cache.v, v], axis=2)
        cache = self.Cache(k, v)

    return (q, k, v) if cache is None else (q, k, v, cache)
```

对于 decoder 中的 cross attention，由于 k,v 都来自于 encoder 的 output，所以我们在准备 k，v 的时候，每个 time step 的对应的映射矩阵都是一样的。因此可以将 k，v 储存为上述的 `StaticCache`。在解码每一个 time step 的时候直接提取使用。

而对于 decoder 中的 self attention。在解码时，我们通常一个一个词输入，k 和 v 为当前所有的 decoder 的 output。为了避免重复计算 t 时刻之前的 k 与 v，我们可以将他们储存在上述的 `Cache` 当中。k，v 的形状为  `[batch_size, num_heads, sequence_length, embed_dim // num_heads]`，在计算 self-attention 的时，只需要计算最近一步的 k，v 然后根据第 2 维度拼接在 `Cache` 中即可。

```python
def gen_cache(self, memory):
    incremental_cache = self.self_attn.gen_cache(
        memory, type=self.self_attn.Cache)
    static_cache = self.cross_attn.gen_cache(
        memory, memory, type=self.cross_attn.StaticCache)
    return incremental_cache, static_cache
```

 **paddle debug** 

```python
# encoder_output = self(input_ids=input_ids,return_encoder_output=True)[1]
# model_kwargs["encoder_output"] = encoder_output
# model_kwargs["decoder_input_ids"] = 1
```

## 其他链接

[beam search 优化](https://opennmt.net/OpenNMT/translation/beam_search/)
[repetition penalty](https://arxiv.org/pdf/1909.05858.pdf) 
[diversity rate](https://arxiv.org/abs/1611.08562)

## 其他论文

1. A Neural Attention Model for Abstractive Sentence Summarization
2. A Deep Reinforced Model for Abstractive Summarization
3. Incorporating Copying Mechanism in Sequence-to-Sequence Learning
4. Get To The Point: Summarization with Pointer-Generator Networks
5. Constructing literature abstracts by computer: Techniques and prospects
6. Recent automatic text summarization techniques: a survey
7. Jointly Learning to Align and Summarize for Neural Cross-Lingual Summarization
8. VMSMO: Learning to Generate Multimodal Summary for Video-based News Articles
9. Q-learning with Language Model for Edit-based Unsupervised Summarization
10. Multi-Fact Correction in Abstractive Text Summarization
11. Incorporating Commonsense Knowledge into Abstractive Dialogue Summarization via Heterogeneous Graph Networks
12. On extractive and abstractive neural document summarization with transformer language models
13. Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation
14. Re-evaluating Evaluation in Text Summarization
15. The Mathematics of Statistical Machine Translation: Parameter Estimation
16. BLEU: a Method for Automatic Evaluation of Machine Translation
17. Statistical Phrase-Based Translation
18. Hierarchical Phrase-Based Translation
19. Sequence to Sequence Learning with Neural Networks
20. Neural Machine Translation by Jointly Learning to Align and Translate
21. Adam: A Method for Stochastic Optimization
22. Neural Machine Translation of Rare Words with Subword Units
23. Attention is All You Need.

