---
title: 机械学习|决策树 到 XGBoost
date: 2021-08-15
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- Machine Learning
mathjax: true
toc: true
comments: 
---

# 机械学习|决策树 到 XGBoost

> 关键词：决策树、ID3、C4.5、CART、随机森林、Adaboost、GBDT、XGBOOST

<!--more-->

## 决策树

|      | 划分标准   | 缺失值 | 剪枝               | 目标      | 问题（相对其他两者）         |
| ---- | ---------- | ------ | ------------------ | --------- | ---------------------------- |
| ID3  | 信息增益   | -      | 预剪枝             | 分类      | 倾向于选择对数量多的特征     |
| C4.5 | 信息增益率 | 有考虑 | 预/后剪枝          | 分类      | 多叉树，效率低；对数运算耗时 |
| CART | 基尼系数   | 有考虑 | 基于代价复杂度剪枝 | 分类/回归 | -                            |

#### ID3

ID3 主要思想：贪婪地选择当前带来信息熵增益最大的特征进行分裂。

 **信息熵增益：** 如特征 A 对数据集 D 的信息增益表示为：

$$
g(D, A)=H(D)-H(D \mid A)
$$

也就是 D 的熵与 特征 A 对数据集 D 的经验条件熵的差。

$$
H(D)=-\sum_{k=1}^{K} \frac{\left|C_{k}\right|}{|D|} \log _{2} \frac{\left|C_{k}\right|}{|D|}
$$

$$
H(D \mid A)=\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{|D|} H\left(D_{i}\right)=-\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{|D|} \sum_{k=1}^{K} \frac{\left|D_{i k}\right|}{\left|D_{i}\right|} \log _{2} \frac{\left|D_{i k}\right|}{\left|D_{i}\right|}
$$

算法 $\mathbf{5 . 2}(\mathbf{I D} 3$ 算法）
输入：训练数据集 $D$, 特征集 $A$ 阈值 $\varepsilon ;$ 输出：决策树 $T$ 。
(1) 若 $D$ 中所有实例属于同一类 $C_{k}$, 则 $T$ 为单结点树, 并将类 $C_{k}$ 作为该结点 的类标记, 返回 $T$;
(2) 若 $A=\varnothing$, 则 $T$ 为单结点树, 并将 $D$ 中实例数最大的类 $C_{k}$ 作为该结点的类 标记, 返回 $T$;
(3)否则, 计算 $A$ 中各特征对 $D$ 的信息增益, 选择信息增益最大的特 征 $A_{g}$;
(4) 如果 $A_{g}$ 的信息增益小于阈值 $\varepsilon$, 则置 $T$ 为单结点树, 并将 $D$ 中实例数最大的类 $C_{k}$ 作为该结点的类标记, 返回 $T$;
(5)否则, 对 $A_{g}$ 的每一可能值 $a_{i}$, 依 $A_{g}=a_{i}$ 将 $D$ 分割为若干非空子集 $D_{i}$, 将$D_{i}$ 中实例数最大的类作为标记, 构建子结点, 由结点及其子结点构成树 $T$, 返回 $T$;
(6) 对第 $i$ 个子结点, 以 $D_{i}$ 为训练集, 以 $A-\left\{A_{g}\right\}$ 为特征集, 递归地调用步 $(1) \sim$ 步 (5), 得到子树 $T_{i}$, 返回 $T_{i}$​ 。

 **预剪枝：** 

可以考虑以下几种剪枝策略：

1. 样本数量低于阀值
2. 信息增益低于阀值
3. 所有节点特征都已分裂
4. 引用验证集，节点划分后准确率比节点划分前低。

#### C4.5

类似 ID3，但特征选择采用了信息熵增益率，改善了 ID3 倾向于选择样本数量多的特征进行分裂的问题。

$$
\begin{aligned}
\operatorname{Gain}_{\text {ratio }}(D, A) &=\frac{\operatorname{Gain}(D, A)}{H_{A}(D)} \\
H_{A}(D) &=-\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{|D|} \log _{2} \frac{\left|D_{i}\right|}{|D|}
\end{aligned}
$$

 **后剪枝策略：** 

在树构造好之后，从下往上依次遍历非叶子节点。剪枝后与剪枝前相比其错误率是保持或者下降，则这棵子树就可以被替换掉。

 **缺失值：** 

对于具有缺失值特征，采用未缺失样本所占比重来折算；

选定某一特征进行划分，对于缺失该特征值的样本，将样本以不同概率划分到不同节点中。（样本可能同时存在于多个子节点中）

#### CART

对比前两者，CART 采用 **二叉树** 而非多叉树。对回归树用平方误差最小化准
则，对分类树用基尼指数(Gini index) 最小化准则，进行特征选择。

对 $K$ 分类问题，使用基尼指数最小化（基尼系数越大，类别不确定性越高。）。对回归问题使用平方误差最小化。

$$
\operatorname{Gini}(p)=\sum_{k=1}^{K} p_{k}\left(1-p_{k}\right)=1-\sum_{k=1}^{K} p_{k}^{2}
$$

$$
\operatorname{Gini}(D, A)=\frac{\left|D_{1}\right|}{|D|} \operatorname{Gini}\left(D_{1}\right)+\frac{\left|D_{2}\right|}{|D|} \operatorname{Gini}\left(D_{2}\right) \tag1
$$

其中 $D_{1}=\{(x, y) \in D \mid A(x)=a\}, D_{2}=D-D_{1}$

 **CART 算法：** 

输入: 训练数据集 $D$, 停止计算的条件;
输出: CART 决策树。
根据训练数据集, 从根结点开始, 递归地对每个结点进行以下操作, 构建二叉决策树:
(1) 设结点的训练数据集为 $D$, 计算现有特征对该数据集的基尼指数。此时, 对每一个特征 $A$, 对其可能取的每个值 $a$, 根据样本点对 $A=a$ 的测试为 “是” 或 “否” 将 $D$ 分割成 $D_{1}$ 和 $D_{2}$ 两部分, 利用上式 $(1)$ 计算 $A=a$ 时的基尼指数。
(2) 在所有可能的特征 $A$ 以及它们所有可能的切分点 $a$ 中, 选择基尼指数最小的特征及其对应的切分点作为最优特征与最优切分点。依最优特征与最优切分点, 从现 结点生成两个子结点, 将训练数据集依特征分配到两个子结点中去。
(3) 对两个子结点递归地调用 (1), (2), 直至满足停止条件。
(4) 生成 CART 决策树。

 **缺失值：** 

对于具有缺失值特征：使用了一种惩罚机制来抑制提升值，从而反映出缺失值的影响（例如，如果一个特征在节点的 20% 的记录是缺失的，那么这个特征就会减少 20% 或者其他数值）

选定某一特征进行划分，对于缺失该特征值的样本：对于每个节点（不论有无缺失值），选择一个特征作为代理，代替缺失值特征作为划分样本的依据。通常要求代理特征超过某规则。如果样本缺少第一代理特征，就采用第二代理特征。若所有代理特征缺失，则划分到较大的子节点中。

 **剪枝：** 

保留准确度的同时，担心树复杂度过大。因此引入了代价复杂度剪枝算法。为了衡量准确度与复杂度，定义损失函数为：

$$
C_{\alpha}(T)=C(T)+\alpha|T|
$$

其中 $C()$ 为预测误差，$\alpha$ 为超参，$T$ 为叶子节点数量。因此剪枝后的损失函数为：

$$
C_{\alpha}(T)=C(T)+\alpha
$$

随着 $\alpha$ 增大，损失函数的惩罚就越大，总有一个 $\alpha$ 使得剪枝前后的损失函数相等，即从这个时刻开始，就有必要剪枝了。

[cart 树怎么进行剪枝？](https://www.zhihu.com/question/22697086)

 **采用类别比例作为节点分类依据** 

不同与 ID3，CART 叶子节点的类别采用子节点类别数量占该类别样本总数的 **百分比** 。而非子节点类别的数量。

## 随机森林

在 ensembling 中，Bagging 与随机森林较为类似，都是由多个基分类器进行聚合。

 **随机森林怎么随机？** 

与 bagging 相似的，随机森林采用  **Bootstrap**  随机地进行样本抽样以构建基分类器的训练集；不同与 bagging ， **随机森林首先随机选择特征 m 个特征，而后从这些特征中计算分裂点** ，特征数量 $1\le m \le M$，$M$ 为样本的全部特征数量。

随机选择特征的目的是为了防止过拟合，通常单个决策树会倾向于过拟合。

## adaboost

个人认为 adaboost 的概念很好，但似乎他被使用的频率不高。他的主要思想是：在学习的过程中，根据误差率不断调整对每个样本的权重，来学习出不同的基本分类器。最后将这些基本分类器进行聚合。

+ 算法 $8.1$​ ( AdaBoost)

输入: 训练数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$​, 其中 $x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}, y_{i} \in \mathcal{Y}=\{-1,+1\} ;$ 弱学习算法;
输出: 最终分类器 $G(x)$​ 。
首先，初始化训练数据的权值分布

$$
D_{1}=\left(w_{11}, \cdots, w_{1 i}, \cdots, w_{1 N}\right), \quad w_{1 i}=\frac{1}{N}, \quad i=1,2, \cdots, N
$$

而后，对 $m=1,2, \cdots, M$
（a）使用具有权值分布 $D_{m}$ 的训练数据集学习，得到基本分类器

$$
G_{m}(x): \mathcal{X} \rightarrow\{-1,+1\}
$$

（b）计算 $G_{m}(x)$ 在训练数据集上的分类误差率

$$
e_{m}=\sum_{i=1}^{N} P\left(G_{m}\left(x_{i}\right) \neq y_{i}\right)=\sum_{i=1}^{N} w_{m i} I\left(G_{m}\left(x_{i}\right) \neq y_{i}\right)
$$

(c) 计算 $G_{m}(x)$ 的系数

$$
\alpha_{m}=\frac{1}{2} \log \frac{1-e_{m}}{e_{m}}
$$

这里的对数是自然对数。
(d) 更新训练数据集的权值分布

$$
\begin{array}{c}
D_{m+1}=\left(w_{m+1,1}, \cdots, w_{m+1, i}, \cdots, w_{m+1, N}\right) \\
w_{m+1, i}=\frac{w_{m i}}{Z_{m}} \exp \left(-\alpha_{m} y_{i} G_{m}\left(x_{i}\right)\right), \quad i=1,2, \cdots, N
\end{array}
$$

最后我们得到的模型就是：$\hat y = \sum_{m\in M}\alpha_m G_{m}(x)$

 **缺点是对异常点非常敏感** 

## 回归提升树

回归提升树的思想是，我们使用新的树来拟合前一个树产生的误差。这样就能将误差不断减小。

已知一个训练数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}, x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}, \mathcal{X}$ 为
输入空间, $y_{i} \in \mathcal{Y} \subseteq \mathbf{R}, \mathcal{Y}$ 为输出空间。如果将 输入空间 $\mathcal{X}$ 划分为 $J$ 个互不相交的区域 $R_{1}, R_{2}, \cdots, R_{J}$, 并且在每个区域上确定输 出的常量 $c_{j}$, 那么树可表示为

$$
T(x ; \Theta)=\sum_{j=1}^{J} c_{j} I\left(x \in R_{j}\right)
$$

其中, 参数 $\Theta=\left\{\left(R_{1}, c_{1}\right),\left(R_{2}, c_{2}\right), \cdots,\left(R_{J}, c_{J}\right)\right\}$ 表示树的区域划分和各区域上的常数。 $J$ 是回归树的复杂度即叶结点个数。

回归问题提升树使用以下前向分步算法:

$$
\begin{array}{l}
f_{0}(x)=0 \\
f_{m}(x)=f_{m-1}(x)+T\left(x ; \Theta_{m}\right), \quad m=1,2, \cdots, M \\
f_{M}(x)=\sum_{m=1}^{M} T\left(x ; \Theta_{m}\right)
\end{array}
$$

在前向分步算法的第 $m$ 步, 给定当前模型 $f_{m-1}(x)$, 需求解

$$
\hat{\Theta}_{m}=\arg \min _{\Theta_{m}} \sum_{i=1}^{N} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+T\left(x_{i} ; \Theta_{m}\right)\right)
$$

得到 $\hat{\Theta}_{m}$, 即第 $m$ 棵树的参数。

## 梯度提升树

相比于回归树的对预测误差进行拟合，梯度提升树对损失梯度进行拟合。联系到梯度上升/下降优化法，梯度提升树所拟合的目标也可以看做是（反向）梯度，损失函数一般为 sigmoid 或者 exp。

输入: 训练数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}, x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}, y_{i} \in \mathcal{Y} \subseteq \mathbf{R} ;$
损失函数 $L(y, f(x))$; 输出：回归树 $\hat{f}(x)$ 。
(1) 初始化

$$
f_{0}(x)=\arg \min _{c} \sum_{i=1}^{N} L\left(y_{i}, c\right)
$$

(2) 对 $m=1,2, \cdots, M$
(a) 对 $i=1,2, \cdots, N$​, 计算

$$
r_{m i}=-\left[\frac{\partial L\left(y_{i}, f\left(x_{i}\right)\right)}{\partial f\left(x_{i}\right)}\right]_{f(x)=f_{m-1}(x)}
$$

(b)  **对**  $r_{m i}$  **拟合一个回归树** , 得到第 $m$ 棵树的叶结点区域 $R_{m j}, j=1,2, \cdots, J$ 。
(c) 对 $j=1,2, \cdots, J$, 计算

$$
c_{m j}=\arg \min _{c} \sum_{x_{i} \in R_{m j}} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+c\right)
$$

(d) 更新 $f_{m}(x)=f_{m-1}(x)+\sum_{j=1}^{J} c_{m j} I\left(x \in R_{m j}\right)$
（3）得到回归树

$$
\hat{f}(x)=f_{M}(x)=\sum_{m=1}^{M} \sum_{i=1}^{J} c_{m j} I\left(x \in R_{m j}\right)
$$

## XGBoost 

与 GBDT 对比，XGBOOST 加入了惩罚项，损失中添加了二阶导数项。

过拟合方面，XGBOOST 除了目标函数添加正则项，也考虑到了随机森林的列抽样（feature subsampling）和缩减（Shrinkage，相当于学习率）

主要特点：算法可并行，训练效率高，实际效率好，可控参数多，可以灵活调整 

####  **目标函数** 

定义一棵树： $f_t(x)=w_{q(x)}$ 树由叶子的权重 $w$ 和实例到叶子节点的映射关系 $q(x)$ 决定，映射关系即实例在第几个叶子上。

假设模型由 $K$ 个字数组成，总体目标函数为 $\mathcal{L}(\phi)=\sum_{i} l\left(\hat{y}_{i}, y_{i}\right)+\sum_{k} \Omega\left(f_{k}\right)$

其中惩罚项 $\Omega(f)=\gamma T+\frac{1}{2} \lambda\|w\|^{2}$ 取决于树复杂度，由树的叶子数量 $T$ 和所有叶子的总权重 $w$ 决定

由于这是个加性模型（additive manner），因此给定第 $t-1$ 棵树的预测值 $\hat y _i^{(t-1)}$ ，第 $t$ 棵树的损失为：

$$
\mathcal{L}^{(t)}=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}^{(t-1)}+f_{t}\left(\mathbf{x}_{i}\right)\right)+\Omega\left(f_{t}\right)
$$

利用泰勒公式展开两项：

$$
\mathcal{L}^{(t)} \simeq \sum_{i=1}^{n}\left[l\left(y_{i}, \hat{y}^{(t-1)}\right)+g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right)
$$

移除常数项得到

$$
\mathcal{L}^{(t)}=\sum_{i=1}^{n}\left[g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right) \tag 1
$$

其中 $g_{i}=\partial_{\hat{y}(t-1)} l\left(y_{i}, \hat{y}^{(t-1)}\right)$，$h_{i}=\partial_{\hat{y}^{(t-1)}}^{2} l\left(y_{i}, \hat{y}^{(t-1)}\right)$

在基模型为 CART 的情况下，总的损失可表示为：

$$
\begin{aligned}
\tilde{\mathcal{L}}^{(t)} &=\sum_{i=1}^{n}\left[g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\gamma T+\frac{1}{2} \lambda \sum_{j=1}^{T} w_{j}^{2} \\
&=\sum_{j=1}^{T}\left[\left(\sum_{i \in I_{j}} g_{i}\right) w_{j}+\frac{1}{2}\left(\sum_{i \in I_{j}} h_{i}+\lambda\right) w_{j}^{2}\right]+\gamma T
\end{aligned}
$$

其中 $g$ 和 $h$ 从前一棵树计算结果得来，视为常数，于是每个节点的最优权重可以记为：

$$
w_{j}^{*}=-\frac{\sum_{i \in I_{j}} g_{i}}{\sum_{i \in I_{j}} h_{i}+\lambda}
$$

带入总损失，可以求得在树结构已知情况下的最优损失为：

$$
\tilde{\mathcal{L}}^{(t)}(q)=-\frac{1}{2} \sum_{j=1}^{T} \frac{\left(\sum_{i \in I_{j}} g_{i}\right)^{2}}{\sum_{i \in I_{j}} h_{i}+\lambda}+\gamma T
$$

#### 分裂算法

假设树被分为左右两部分，对应的样本子集为：$I=I_L \cup I_R$，则 **分裂收益**  loss reduction after the split 为分裂成两棵树后的总损失增益：

$$
\mathcal{L}_{\text {split }}=\frac{1}{2}\left[\frac{\left(\sum_{i \in I_{L}} g_{i}\right)^{2}}{\sum_{i \in I_{L}} h_{i}+\lambda}+\frac{\left(\sum_{i \in I_{R}} g_{i}\right)^{2}}{\sum_{i \in I_{R}} h_{i}+\lambda}-\frac{\left(\sum_{i \in I} g_{i}\right)^{2}}{\sum_{i \in I} h_{i}+\lambda}\right]-\gamma
$$

 **贪婪算法：** 

在每个树选择分支的时候，选择 **分裂收益** 最高的节点拓展方案。

这个方案需要遍历所有的特征，遍历所有特征的分裂点，选最优的样本和对应分裂点。为加速计算，应实现将样本根据特征进行排序。

![image-20220317200448405](/assets/img/mltree/image-20220317200448405.png)

 **近似算法:** 

解决数据太大无法读入内存进行计算。

在计算最优分裂点前，先筛选出一些候选分裂点。可以采用 global 方式（每个树初始化时候选一次候选分裂点。）或 local 方式（每次分裂的时候选一次）

![image-20220317200505757](/assets/img/mltree/image-20220317200505757.png)

得到候选分裂点后，只需要计算每个分裂点的收益，从中选取最大的即可。

候选分裂点筛选方式：Weighted Quantile Sketch

$$
r_{k}(z)=\frac{1}{\sum_{(x, h) \in \mathcal{D}_{k}} h} \sum_{(x, h) \in \mathcal{D}_{k}, x<z} h
$$

其中 $x$ 为样本对应特征值，$h$ 为二阶导数值，$r \in (0,1$ )为分裂点评分。分裂点需满足：

$$
\left|r_{k}\left(s_{k, j}\right)-r_{k}\left(s_{k, j+1}\right)\right|<\epsilon, \quad s_{k 1}=\min _{i} \mathbf{x}_{i k}, s_{k l}=\max _{i} \mathbf{x}_{i k}
$$

因此大约会有 $1/\epsilon$ 个分裂点。使用二阶导数作为评分权重，因为通过重写每棵树的损失函数，即式 $(1)$：

$$
\sum_{i=1}^{n} \frac{1}{2} h_{i}\left(f_{t}\left(\mathbf{x}_{i}\right)-g_{i} / h_{i}\right)^{2}+\Omega\left(f_{t}\right)+\text { constant }
$$

可以看出二阶导数就是损失函数中的权重。

 **稀疏感知算法** 

在稀疏数据中，对于缺少的数值， **每个节点都有一个默认的分配稀疏值方向** 。这样所需要遍历样本量大大减小，论文中表示稀疏感知算法比基础算法速度块了超过 50 倍。

寻找默认分配方向需要遍历所有缺失值分配到左右两个节点对应的最大收益，选取收益最大的方向。

### 工程设计

XGBOOST 最大的特点就是，真的考虑了很多工程实践上的问题！

####  **为并行计算设计的块** 

从上面的算法可以看出，XGBOOST 有大量排序计算。因此文章提出了使用块（block）来加速计算。将排好序的数据采用 CSC 格式储存在快中，每个特征储存指向样本梯度的索引。

块之间相互独立，以支持并行计算。

#### 缓存访问优化

（内存读取方面的优化）为每个线程提供一个缓存区，填充梯度信息。减少非连续内存的读取，提高运算效率。

#### Blocks for Out-of-core Computation

为达到 scalable learning，数据将被分为不同的块 block，储存与不同的磁盘上。对于 block 采取两种优化方案：

Block Compression：对数据进行压缩，读入内存时调用一个独立的线程进行解压。

Block Sharding:在不同的磁盘上储存多分 block 的副本，提供条用时候的吞吐量。

 **XGBOOT 的优点：** 

 **目标函数升级**  - 对比 GBDT，优化函数考虑了泰勒二阶展开，加入正则项，有助于防止过拟合，这个比 GBDT 好。

 **训练优化**  - 快速的缺省值处理、采用列抽样、学习率等、适配性强、可对不同任务，采用不同目标函数和基分类器。

 **工程适配好**  - 效率高、并行计算、内存优化、分布式数据。

## Lightgbm

 **histogram optimization**  - 将连续性数据通过 bin 映射到离散型数据上

 **leaf-wise learning**  - 在其他树中，大部分都是每一层选择一个特征，然后对所有节点都进行分裂（level-wise learning）而在 lightgbm 中是选择一个叶节点进行分裂，因此 lightgb 增加了最大深度限制，防止过拟合。

参数分析

```python
boosting_type (string, optional (default='gbdt')) # ‘gbdt’,
num_leaves (int, optional (default=31)) # 叶子总数量，用于控制树的复杂度,2^max_depth
max_depth (int, optional (default=-1)) # 防止 leaf-wise 过拟合
learning_rate (float, optional (default=0.1)) – # callbacks parameter of fit method to shrink/adapt learning rate in training using reset_parameter callback.
subsample_for_bin (int, optional (default=200000)) # 用多少 sample 来构建我们 histogram optimization 的 bins.越少算越快。
objective (string, callable or None, optional (default=None)) # binary 或者 multiclass
class_weight (dict, 'balanced' or None, optional (default=None)) # multiclass 的不同权重
min_split_gain (float, optional (default=0.))
min_child_weight (float, optional (default=1e-3))
min_child_samples (int, optional (default=20)) # 一个叶子上最少的样本数，防止过拟合
subsample (float, optional (default=1.)) # 随机抽取多少作为样本
subsample_freq (int, optional (default=0)) # 表示每训练 n 次就进行一次 bagging, <=0 表示禁用 bagging.
colsample_bytree (float, optional (default=1.)) # 类似与随机森林，开局选一些比例的特征来训练
reg_alpha (float, optional (default=0.)) # L1 regularization term on weights.
reg_lambda (float, optional (default=0.)) # L2 regularization term on weights.
importance_type (string, optional (default='split')) # The type of feature importance to be filled into feature_importances_. If ‘split’, result contains numbers of times the feature is used in a model. If ‘gain’, result contains total gains of splits which use the feature.
```

