---
title: 人工智能 计算智能体基础|智能体结构与分层 第二章
date: 2022-06-23
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- AI 必读
mathjax: true
toc: true
comments: 笔记

---

> 印象最深的 AI 书籍：人工智能 计算智能体基础笔记。[书籍全文链接](http://www.artint.info/2e/html/ArtInt2e.Ch1.S1.html)

# 智能体结构与分层

在大自然中，我们也会常识性的对事物的结构进行分层，直到某些最基础的子系统，如物理中的基本粒子。



智能体的架构应该是什么样子的？首先我们来考虑使用单一的函数来实现一个智能体：

### 智能体函数

<img src="http://www.artint.info/2e/html/x354.png">

上图是智能体的架构，以购买自动交易机为例子。 percepts 为股票价格、成交量等，command 为机器想要购买的股票数量。environment 即为股市环境。

交易机通过 Body 接受股票的信息，并且执行股市具体交易。而 Controller 则通过股票信息决定采取什么交易策略。

Percepts 感知得到的信息是会随时间变化的，比如某个交易高峰时刻，机器可能会遭受一定的延迟，Commands 同理。

此外 Commands 发出的交易策略并不一定被执行，比如当 Body 接收到以 12 块钱买入 10 手股票时，股票价格涨到了 14 元。 但这种情况比较少见，因为机器的信息传递速度还是足够快的。

Controller 类似我们的大脑，身体传递视觉、触觉等 Percepts 到大脑中，大脑在通过这些知觉，发出命令来控制身体。

类似大脑中存有记忆，机器也能够获取过去的经验来辅助目前的决策。 如果将记忆表示为单一的状态 S（brief state），那么 S 随着时间以及我们的感知 Percepts 更新。

### 分级控制

<img src="http://www.artint.info/2e/html/x355.png">

事实证明前面提到的智能体架构太慢了，很难调和对复杂的高级目标的缓慢推理和低级任务所需要的快速反应。 

考虑人的快速反应来源，大部分的快速反应都是条件反射带动的，因此我们可以考虑一个分层的智能体，每层都会接收如上节中提到的命令、内省和记忆。

低层处理如直觉，本能，情绪化等，并且不接受内省 

 **在较高层的推理通常是离散的和定性的，而在较低层执行的低级推理通常是连续的和定量的。**  这也是分层智能体和上一节中智能体最主要的不同之处。每个层只需要处理他所负责的参数形式，从而大大减少了运算时间（联想下 python 高性能编程，当我们对 Cython 函数输入进行类型限定时，性能能够有极大的提升正是因为减少了很多中间不必要的数据转换）。

<img src="http://www.artint.info/2e/html/x357.png"> 

### 智能体记忆力的世界

提示：我们可以通过收集一些感知信息，如视觉传感器，同时采用贝叶斯法则来过滤现实生活中的噪声。

### 知识和应用

知识的学习和应用大致逻辑：

<img src="http://www.artint.info/2e/html/x362.png">

一个例子就是 ELBO。



Ontology（本体）: 系统中，符号的特殊意义（一般指和现实世界的关联）。如果一个模型是 state-based 的，那么他的本体就是解释 state 中数字显示意义的映射。



Knowledge Engineer 是协助 domain expert 构建知识库的人员（如码农）。

## 寻找答案

寻找答案的过程，可以视为在有向图中寻找起始节点到目标节点的路径的数学问题，若将答案表示成路径，那该路径将由不同的状态（state）组成。

本书第三章探索了部分路径搜索算法。

### 路径搜索

最佳路线可能意味着：

+ 最短路线
+ 最快路线
+ 考虑到时间、金钱和路线吸引力的最低成本路线。

路径搜索的挑战：

+ 路径中的噪声，会影响路径的实际决策。
+ 计算路径时间时，需要考虑到其他智能体选择的路径

对于路径搜索，找到最佳的解决方案往往是 NP hard 的。但是我们能够找到令人满意的解决方案。

我们应该用有关特殊情况的知识来指导智能体找到解决方案，这些额外的知识也被称为启发式知识。

#### 用图定义任务

如果用图来表示一个问题（书中称为 graph searching），那么我们有以下定义：

+ 节点 n 表示 State
+ 边表示 action
+ goal 表示 `goal(n)` 布尔值
+ cost：通常每条边都会带有一个 cost
+ solution：solution 是一条总 cost 最小的路径

对于许多问题，搜索图并不是明确给定的，需要动态地进行调整。

定义：

+ branching factor：number of outgoing/incoming arcs of node.



#### 3.4  A Generic Searching Algorithm

<img src="http://www.artint.info/2e/html/x227.png">

 



算法通常是具有不确定性的：

+ choose：算法不会知道当前如何选择，才能够达到目标。因此我们需要对所有结果进行遍历。
+ Select：在某些路径上，不论如何探索，最终的结局都不会是好的。虽然在这些无用的路径上探索不会影响算法最后结果的准确性，但是却会大大影响算法的效率。因此，启发式是一个能够用来辨别无用选项的方法。

书中对 select 和 choose 两个词进行了不同的定义，以表示以上两种不同的，虽然都有选择的意思，但 choose 更偏向于中性的选择，而 select 则蕴含有选出好的的意思。

重点：P 与 NP

#### 3.5 Uninformed Search Strategies 

信息缺乏情况下的搜素方案

BFS

DFS

iterative deepening(有深度限制的 DFS)

lowest cost first search

A* search（implemented using generic search algorithm）

Heuristic Function：启发式方程是对未来 cost 下限的一个估算。如对于旅游问题，两个景点的最小消耗就是他们之间的直线距离。



cycle pruning

验证一个图是否为有环图

multiple-path pruning

当我们的目的是只要返回一个可行解时，可以采用。我们只需要维持所有 frontier 中，消耗最小地那个就行。



 
