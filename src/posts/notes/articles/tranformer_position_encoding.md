---
title: 信息抽取小综述|开放抽取、事件抽取
date: 2022-04-13
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
mathjax: true
toc: true
---

# 事件抽取

事件抽取应用广泛：可用于拓展知识图谱、知识库；舆情监控、分析；金融风险判断；蛋白质相互作用、生理和发病机制等。

 **不同行业有不同的事件抽取综述，如：** 

+ 通用逻辑推理：2-3

Building event-centric knowledge graphs from news. 2016
Constructing narrative event evolutionary graph for script event prediction. 2018

+ 公共舆情监控：4-8

Terrorism information extraction from online reports. 2015
Automated event extraction in the domain of border security. 2009
Techniques for multilingual security-related event extraction from online news. 2013
Real-time news event extraction for global crisis monitoring. 2008
Extracting violent events from on-line news for ontology population. 2007

+ 金融风控：9-11

An automated framework for incorporating news into stock trading strategies. 2014
A risk assessment system with automatic extraction of event types. 2008
Developing and executing electronic commerce applications with occurences. 2002

+ 蛋白质相互作用：12

An overview of biomolecular event extraction from scientific documents. 2015

比较权威的自动内容抽取公开评价项目如： ACE、TAC 等。ACE 2005 语料库共包含 599 个标注文档和约 6000 个标注事件，包括英语、阿拉伯语和汉语等不同媒体来源的事件，如新闻热线文章、广播新闻等

## 任务形式

事件抽取，即从一段非结构体描述（Event Mention）中，根据预先指定的事件类型和论元角色（也成为 Schema），识别句子中所有目标事件类型的事件，并根据相应的论元角色集合抽取事件所对应的论元（Event Arguments）。

如对于金融事件抽取：宁波容百新能源科技股份有限公司(简称“容百科技”，证券代码：688005)在科创板上市。

给定的 Schema 为：` (event_type：上市，role：上市时间、上市板块、上市企业、融资金额)`。抽取的事件结果为：

```json
{"event_list":[
    {
        "event_type":"上市",
        "arguments":[
            {
                "role":"上市板块",
                "argument":"科创板"
            },
            {
                "role":"上市企业",
                "argument":"宁波容百新能源科技股份有限公司"
            }
        ]
    }
]}
```

因此定义好事件的架构（schema）是事件抽取的前提。事件抽取通常也分为事件判断与事件元素抽取两步。事件判断可能涉及到 **事件触发词**  （Event Trigger）。

不同领域中通常存在一些较有权威性的 Schema 定义，如 A document-level chinese financial event extraction system based on automatically labeled training data 中便定义了 9 中金融事件类型。

### 任务关注点

事件抽取中事件可能以多种描述形式，分散出现在多个句子和段落中。如何判断事件是否属于同一个，该任务称为事件共指。

通常一个事件不是一两个句子就能够表述清楚的，如何从长篇文档中抽取出事件？

如何学习到一个事件的语义表征？

如何基于少量的标注数据来训练一个事件抽取模型？

## 事件抽取方案

### 基于匹配

基于模式的匹配依赖于词典与人工构造的模版。例：恐怖事件抽取 AutoSlog；生物医学事件 OpenDMAP、Kybots；新闻商业事件 BEECON。如何拓展模版是个问题，并且模版在领域之间的迁移性很差，相关拓展模版的方式有 PALKA(Parallel Automatic Language Knowledge Acquisition)、Booststrapping 等。

### 基于传统机械学习

该方案大致流程分为 ：

1. 检测触发器是否存在，若存在检测触发器类型。
2. 单词特征提取（NER,POS TAGGING,句法分析等）
3. 构建词表征（包括分词，构建词向量）
4. 对事件中的名词分别进行分类

该方案下注重文本特征的质量。

#### 句子级事件抽取

#### 文档级事件抽取

文档级事件抽取可以使用全局信息来辅助句子级事件的抽取。应用什么全局信息，主题特征等？如何使用他们？

### 基于深度学习

深度学习大致采用端到端解决方案，使用预训练 embedding + 编码器（CNN/RNN/GNN）- 解码器分类的模型结构。

PMCNN - Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks

并行多池卷积神经网络(PMCNN)，该网络可以捕获句子的成分语义特征，用于生物医学事件的提取

DLRNN - Jointly Extracting Event Triggers and Arguments by Dependency-Bridge RNN and Tensor-Based Argument Interaction

RNN 中加入句法依赖信息；通过使用分布式文档表示向量来提取跨句子甚至跨文档的线索，即通过无监督学习 PV-DM 模型来捕捉文档的主题分布。

Abstract meaning representaiton for sembanking

规范化文本中的许多词汇和语法变化，并输出一个有向无环图，以捕获文本中“谁对谁做了什么”的概念。

Jointly multiple events extraction via attention-based graph information aggregation

通过句法依赖创建图

Joint entity and event extraction with generative adversarial imitation learning

Self-regulation: Employing a generative adversarial network to improve event detection

Exploiting the ground-truth: An adversarial imitation based knowledge distillation approach for event detection,

GAN 事件提取

### 半监督学习

半监督学习意在使用规则，拓展训练集的数量，类似于数据增强。

#### 联合数据拓展

S. Abney, ‘‘Bootstrapping,’’ in Proc. 40th Annu. Meeting Assoc. Comput. Linguistics, 2002,

首先用一小部分有标签的数据训练分类器，对新的无标签数据进行分类。除了分类标签，分类器还输出新数据的分类置信度。然后，将置信度较高的数据加入到训练数据中，用于下一轮模型训练。

Can document selection help semi-supervised learning?: A case study on event extraction

Semi-supervised event extraction with paraphrase clusters

Adversarial training for weakly supervised event detection,

Using prediction from sentential scope to build a pseudo co-testing learner for event extraction

A two-stage approach for extending event detection to new types via neural networks

Zero-shot transfer learning for event extraction

#### 知识库数据拓展

Scale up event extraction learning via automatic training data generation

Doc2EDAG: An end-to-end document-level framework for chinese financial event extraction

Open-domain event detection using distant supervision

Dcfee: A document-level chinese financial event extraction system based on automatically labeled training data

金融知识库进行数据拓展

#### 多语言数据拓展

Multilingual entity, relation, event and human value extraction,

### 无监督学习

采用检索的方式，如 TF-IDF 表示文档，将其与实现给定的与事件密切相关的文章表征进行匹配。

相似度搜索、马尔科夫聚类

可以采用规则进行事件抽取，如动词视为触发器，而后通过句法分析，判断与触发器对应的名词、宾语等作为论元。







### 挑战

社交媒体内容措辞不一，相对于官方新闻内容更具有抽取事件挑战性。

## 事件的语义表征

Extracting Events and Their Relations from Texts: A Survey on Recent Research Progress and Challenges. 26 页总结了不同类型

Event extraction via dynamic multipooling convolutional neural networks. (DMCNN)

特点：根据 argument 和 trigger 进行 dynamic multi-pooling

# 开放抽取

主题检测与评估项目 TDT，其中任务包括

Story segmentation：故事分割，从新闻文章中检测故事边界；
First story detection：第一故事检测，检测新闻流中讨论新话题的故事；
Topic detection：话题检测，根据故事讨论的主题对故事进行分组；
Topic tracking：话题跟踪，检测讨论以前已知话题的报道
Story link detection：故事链接检测，判断一对故事是否讨论同一主题。

除了事件检测（任务一、二）和事件聚类（任务 3、4、5），还有提取事件关键词，如 A multiple instance learning framework for identifying key sentences and detecting events。









### 资源

[nlp 中的实体关系抽取方法总结](https://zhuanlan.zhihu.com/p/77868938)

- Beyond Word Attention: Using Segment Attention in Neural Relation Extraction. IJCAI 2019
- From What to Why: Improving Relation Extraction with Rationale Graph. Findings of ACL 2021



## 遗留问题

图神经网络在关系抽取的应用，提供更好的实体、实体与关系、关系与关系之间的信息交互

对信息抽取的低资源、复杂样本、数据质量等问题进行理解

低资源训练：增加预训练任务，领域预训练

1. [^](https://zhuanlan.zhihu.com/p/77868938#ref_43_0)Knowledge-Augmented Language Model and its Application to Unsupervised Named-Entity Recognition
2. [^](https://zhuanlan.zhihu.com/p/77868938#ref_44_0)Description-Based Zero-shot Fine-Grained Entity Typing
3. [^](https://zhuanlan.zhihu.com/p/77868938#ref_45_0)Zero-Shot Entity Linking by Reading Entity Descriptions
4. [^](https://zhuanlan.zhihu.com/p/77868938#ref_46_0)Multi-Level Matching and Aggregation Network for Few-Shot Relation Classification
5. [^](https://zhuanlan.zhihu.com/p/77868938#ref_47_0)Exploiting Entity BIO Tag Embeddings and Multi-task Learning for Relation Extraction with Imbalanced Data
6. [^](https://zhuanlan.zhihu.com/p/77868938#ref_48_0)Massively Multilingual Transfer for NER



# 参考链接

[工业界如何解决 NER 问题？12 个 trick，与你分享～](https://zhuanlan.zhihu.com/p/152463745)

[2020 语言与智能技术竞赛：关系抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/31/0/task-definition)

[2021 语言与智能技术竞赛：多形态信息抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/65/0/task-definition)

[CCKS2022 通用信息抽取竞赛](https://aistudio.baidu.com/aistudio/competition/detail/161/0/task-definition) 

[中文学术数据集](https://www.luge.ai/#/) - DuIE2.0 关系抽取、DuEE1.0 事件抽取、DuEE-fin 金融领域篇章级事件抽取

[请查收！一份关于事件抽取综述的综述~](https://zhuanlan.zhihu.com/p/434676352)



[事件抽取论文整理 2014-2021](https://github.com/carrie0307/DL_EventExtractionPapers)

[事件演化挖掘开篇：故事森林 storyforest 系统中的 keygraph 算法思想与实现细节剖析](https://mp.weixin.qq.com/s/ABlQLdnPMjFY23dQqB293Q)

[刘焕勇 - 如何进行事件标注：Duee 等代表性事件标注数据集解析与 Marktool 事件标注动手实现](https://mp.weixin.qq.com/s/_Fg-h9ByY1GL4gyjBoKulw)

[刘焕勇 - 事件抽取系列](https://github.com/liuhuanyong/ComplexEventExtraction)

[如何解决 NLP 分类任务的 11 个关键问题：类别不平衡&低耗时计算&小样本&鲁棒性&测试检验&长文本分类](https://zhuanlan.zhihu.com/p/183852900)

[中科院柳厅文：面向非结构化文本的信息抽取](https://zhuanlan.zhihu.com/p/455700987)





[刷爆 3 路榜单，信息抽取冠军方案分享：嵌套 NER+关系抽取+实体标准化](https://zhuanlan.zhihu.com/p/326302618)



[知识图谱从哪里来：实体关系抽取的现状与未来](https://zhuanlan.zhihu.com/p/91762831)



# 论文清单





