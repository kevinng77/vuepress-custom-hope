---
title: 图卷积神经网络 GCN 
date: 2021-08-18
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- 图网络
mathjax: true
toc: true
comments: 笔记
---

> 不同的数据结构使得学者们创造了不同的特征提取和编码方式，例如 CNN 的卷积核很好地应用在了图像编码上，RNN 的结构更好地提取了文本或者其他序列数据的时间步信息。GCN 的设计是为了对图谱，社交网络等图结构数据有更好的解析能力。
>
> GCN 可以做 **节点分类（node classification）、图分类（graph classification）、边预测（link prediction）** ，同时也可以提取 **图相关的嵌入（graph embedding）** 

<!--more-->

## 简单讲讲图

#### 图论历史 - 三次里程碑

1735 年 8 月 26 日，欧拉向当时俄国的圣彼得堡科学院递交了一篇名为《有关位置几何的一个问题的解》的论文，阐述了他是如何否定哥尼斯堡七桥问题能一次走完的。而后人们认识到可以用点与线来描述具体问题，以此引发了对图论的继续研究。[引用-从七桥问题开始说图论](https://zhuanlan.zhihu.com/p/38378095)

在 1959 和 1968 年期间，数学家 Paul Erdos 和 Alfred Renyi 发表了关于随机图（ **Random Graph** ）的一系列论文，在图论的研究中融入了组合数学和概率论，建立了一个全新的数学领域分支---随机图论。[引用-随机图模型](https://zhuanlan.zhihu.com/p/300861555)

1999 年 Barabasi 提出了 Scalefree network 无标度网络。[引用-什么是无标度网络 | 集智百科](https://zhuanlan.zhihu.com/p/138216968)，[无标度网络模型开山之作：随机网络中标度的涌现](https://mp.weixin.qq.com/s?__biz=MzIzMjQyNzQ5MA==&mid=2247502769&idx=1&sn=25e1cc11eea37ac07276ac6efc3126e6&chksm=e897913cdfe0182af17d1153cc5954620cbdbdf5e3617d89ded60124ddd964cf29bbf958edf3&scene=21#wechat_redirect) 

### 图基础

图根据是否有向 directed 与是否有权重 weighted 可以分成四类图。现实中的数据，大部分都是有权重的，无权重的图论主要在理论界发展。

####  **度分布** 

节点度通常定义为该节点所有连接边的数量，网络的度分布即为网络中各节点的度的概率分布或者频率分布。

 **对于随机图，度分布为** 

$$
p_{d}=C(N-1, d) \cdot p^{d} \cdot(1-p)^{N-1-d}
$$

其中 $p_d$ 为 一个定点恰好度数为 $d$ 的概率，当 $d \ll N$ 时，度分布近似与泊松分布。

$$
p_{d} \sim\langle k\rangle^{d} e^{-\langle k\rangle} / d !
$$

其中点平均度数 $\langle k\rangle=p(N-1) \sim p N$ 

 **对于无标度网络，度分布呈幂率分布** 

$$
P(k) \sim k^{-\gamma}
$$

通常 $\gamma$ 在 2 到 3 之间。

#### 距离衡量 Distance 

GCN 对距离的要求：1. 点与点之间的权重为正。2. 节点之间的距离计算，需要在线性时间复杂度内。

希尔伯特空间为完备空间，他的内积运算满足：

(1) $(\varphi, \psi)=(\psi, \varphi)^{*}$
(2) $(\varphi, \psi+\chi)=(\varphi, \psi)+(\varphi, \chi)$
(3) $(\varphi, a \psi)=(\varphi, \psi) a$
$(a \varphi, \psi)=a^{*}(\varphi, \psi)$
(4) $(\varphi, \varphi) \geq 0$
若 $(\varphi, \varphi)=0$, 则 $\hat{\varphi}=0$

其中 a 为复数域上的一个数，$\phi, \psi, \chi$ 为线性空间中的任意三个矢量。

[知乎相关 - 什么是希尔伯特空间？](https://zhuanlan.zhihu.com/p/88946250)

#### 聚类系数 Clustering Coefficient

聚类系数 Clustering coefficient 可用于衡量网络的稠密程度。

局部聚类系数（面向节点）

$$
C(i) = \frac {\#\ of\ triangles\ with\ i}
{\# of\ triples\ with\ i}
$$

![相关图片](/assets/img/gcn/image-20210811153319735.png )

例如，对于节点 3，他的 closed triplet （triangles） 为 {3，(1,2)}，所有的 triplet 为 {3，（1,2）}，{3，（2,4）}，{3，（4,5）}，{3，（1,5）}，{3，（2,5）}，{3，（1,4）}，因此他的 3 节点的局部聚类系数为：1/6。

全局聚类系数则面向所有的节点

$$
C = \frac {\#\ of\ triangles }
{\# of\ triples}
$$


（计算 triple 时候，i 需要作为中间（枢纽）节点）

[聚类系数（clustering coefficient）计算](https://www.cnblogs.com/startover/p/3141646.html)

#### 中介度 Betweeness

 **点界度 node Betweeness：** 

 节点 i 的点界度计算方式为： $B(i) = \sum \frac {L_{jL}(i)}{L_{jL}},i\ne j,i\ne L,j\ne L$ 其中 L 为最短路径

![相关图片](/assets/img/gcn/image-20210808100729282.png =x300)
以点 1 点界度计算为例：

$$
\begin{aligned}
B(1)=& \frac{(5,1,4)}{(5,1,4)}+\frac{0}{(5,3)}+\frac{(5,1,2)}{(5,1,2)+(5,3,2)} \\
&+\frac{(4,1,2,3)+(4,1,5,3)}{(4,1,2,3)+(4,1,5,3)}+\frac{(4,1,2)}{(4,1,2)}+\frac{0}{(3,2)}
\end{aligned}\\
= 1+0+1/2+2/2+2/2+1
$$

计算点 1 的点界度时，依次计算所有两点组合的最短路径数量，以及最短路径中有经过点 1 的路径数量。

 **边界度 edge betweeness** 

边界度与点界度的计算方式相似，边 $e_{ij}$ 的计算方法为：

$$
B\left(e_{i j}\right)=\sum \frac{L_{L q}\left(e_{i j}\right)}{L_{1} q} \quad(i, j) \neq(L, q)
$$

![相关图片](/assets/img/gcn/image-20210808101211909.png =x300)
以 $e_{12}$ 的计算为例：

$$
B(e_{12}) = \frac{(3,2,1,4)}{(3,2,1,4)}=\frac{1}{1}
$$

betweeness 用来衡量节点或者边的流通性，对于搜索桥接点十分有帮助。

#### 其他基本概念

walk，trail， **path** ， **tree** ，complementary gragh，perfect match，bridge，Eulerian Gragh，Hamiltonian graph， **Motif / subgragh** ， **coreness** ，hyper gragh (HGCN)

#### 拉普拉斯算子

![img](https://pic4.zhimg.com/80/v2-9203a8d068c794ea0e03e9761d2451cb_1440w.jpg)

$$
\Delta f=\sum_{i=1}^{n} \frac{\partial^{2} f}{\partial x_{i}^{2}}
$$

对于基函数 $e^{-j \omega t}$，其拉普拉斯算子为：

$$
\Delta e^{-j \omega t}=\frac{\partial^{2} e^{-j \omega} t}{\partial^{2} t}=-\omega^{2} e^{-j \omega t}
$$

即 $e^{-j \omega t}$ 为自身拉普拉斯算子的特征向量。

 **图上节点拉普拉斯算子定义** 

首先定义 节点特征向量  $f_i \in f = (f_1, f_2,...,f_N)$，

而后定义节点 $i$ 的拉普拉斯算子为：$i$ 节点与邻上节点的差值总和。[推导]() 

$$
\Delta f_{i}=\sum_{i \in N_{i}} W_{i j}\left(f_{i}-f_{j}\right) \\
w_{i j}=0\text{ if  i, j 不相邻} \\
\Delta f_{i}=\sum_j W_{ij}f_i - \sum_j W_{ij} f_j\\
\text{ 令 }i\text{ 的度 }= d_i = \sum_jw_{ij}\\
\text{ 行向量 } w_i=[w_{i1},w_{i2},...]\\
\text{ 列向量 }f = [f_1,f_2,...]\\
\Delta f_{i} = d_if_i-fw_i\\
=(D-A)f
$$

所以 $L=D-A$

 **图上傅里叶变化的定义** 

因为拉普拉斯算子的特征向量为 $e^{-jwt}$ ，对拉普拉斯算子进行 SVD 特征展开后，得到 $L=U \wedge U^{\top}$ ，因此图的傅里叶变化为$\hat f=U^Tf$。此处并没有严格的数学推导，存在部分的近似代替与逻辑推理。

### GCN

考虑一个图结构的数据，其中有 N 个节点，$X\in \mathbb{R}^{N \times D}$ 为节点的特征矩阵，节点之间的邻接矩阵（adjacency matrix）为 $A\in \mathbb{R}^{N \times N}$。

GCN 目的是建立一个模型 $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ ，使得给定输入 $X,A$，输出 $Z \in \mathbb{R}^{N \times F}$, 其中 $F$ 为我们所需要的每个节点的特征数量。

首先考虑这个简单的模型结构：

$$
H^{(l+1)} =f\left(H^{(l)}, A\right)=\sigma\left(A H^{(l)} W^{(l)}\right)
$$

其中 $H$ 是每一层的 activation，$H^{(l)} \in \mathbb{R}^{N \times D},H^0=X,
\sigma(.)$\text{ 是激活函数 },\text{ 如 }$Relu(.)$

以上模型有两个缺点：

+ 特征矩阵与对角线全为 0 的邻接矩阵 $A$ 点积，这意味着每个节点都可以考虑到其他节点的信息，缺遗漏了节点本身的信息。解决方式就是给他加上一个单位矩阵 $I$。
+ A 一般都是没有归一化的，所以特征矩阵与 A 点积将使得节点特征的规模(scale) 产生变化。可以使用  **random walk normalization**  $D^{-1}A$，其中 $D$ 是度矩阵 (degree matrix)。然而 GCN 使用的是  **symmetric normalization**  $D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$。symmetric normalization 同时考虑了 edge 上两端节点的度。

在改进后，GCN 有了以下层与层之间的结构：

$$
H^{(l+1)}=\sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)
$$

其中

$$
\tilde{A} = A + I,I \text{ 是单位矩阵 }\\
\tilde{D}\text{ 为 }\tilde{A}\text{ 的 }degree\ maxtix, \tilde{D} i i=\sum j \tilde{A}_{i j}\\
$$

l 论文中以下面这个双层网络结构解释了 GCN 的优化过程：

$$
Z=f(X, A)=\operatorname{softmax}\left(\hat{A} \operatorname{ReLU}\left(\hat{A} X W^{(0)}\right) W^{(1)}\right)
$$

![相关图片](/assets/img/gcn/image-20210806082631965.png )

左图为我们的网络结构，右图在 Cora 数据集上训练后的 activation 可视化（t-SNE）

由于数据集中只有小部分有标注，因此我们只对有标注的序列计算交叉熵。

$$
\mathcal{L}=-\sum_{l \in \mathcal{Y}_{L}} \sum_{f=1}^{F} Y_{l f} \ln Z_{l f},\mathcal{Y}_{L}\text{ 为有标记的数据 }
$$

 **图卷积的演变** 

第一代的 GCN 卷积核为

$$
(f * g)=\sigma(Ug_{\theta}U^Tx)
$$

其中 x 为特征矩阵。因为没有归一化，存在收敛问题。同时也没有考虑点对自身的影响。计算复杂度也在 $O(N^2)$ 。

第二代的卷积核解决了自身权重与归一化的问题。二代使用了切比雪夫展开，

$$
\begin{aligned}
g_{\theta} * x &=U\sum_{k=0}^{\infty} \theta_{k} \Lambda^{k} U^{\top} x \\
&=\sum_{k=0}^{k} \theta_{k}^{\prime}\left(U \Lambda^{k} U^{\top}\right) x \\
&=\sum_{k=0} \theta_{k}^{\prime}\left(U \Lambda U^{T}\right)^{k} x \\
&=\sum^k \theta_{k}^{\prime} L^{k} x
\end{aligned}
$$

而后第三代卷积核加入了切比雪夫展开的前两项的近似优化，并对其中某些参数进行了定义，解决了时间复杂度问题，最终形成了目前使用的公式。

## 代码实例

官方 torch 代码 [repo](https://github.com/tkipf/pygcn) 

### 数据处理

cora 数据集有 cora.content 与 cora.cities 两个文件。其中 content 文件格式为 `paper_id x_1 x_2 ... x_i y` x 为 1433 维向量的 0/1 整数，代表词表中对应编码的词是否出现在该文中，y 为论文的类别。cities 文件格式为 `paper_id1 paper_id2` 表示 这两个论文之间存在引用关系。

##### 邻接矩阵

邻接矩阵的维度为 `(论文数，论文数)` ，若论文 `i` 引用了 `j` ，则`A[i,j]=1` 。实际代码中采用了稀疏矩阵进行储存，并且需要将临街矩阵的 `idx` 与特征矩阵对应。

```python
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),dtype=np.int32)
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                 dtype=np.int32).reshape(edges_unordered.shape)
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(labels.shape[0], labels.shape[0]),
                    dtype=np.float32)
```

得到邻接矩阵 $A$ 后， **将有向图转变为无向图，保留权重最大的边** 。而后再处理成 $\hat A$​ :

```python
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = normalize(adj + sp.eye(adj.shape[0]))
```

这边的归一化可以简单的按行处理，也可以根据上面公式进行对称归一化 (symmetric normalization)。

##### 特征矩阵

特征矩阵的处理十分朴素，唯一要注意的就是这边做了归一化。

```python
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
features = normalize(features)
```

##### 标签 Label

```python
labels = encode_onehot(idx_features_labels[:, -1])
labels = torch.LongTensor(np.where(labels)[1])
```

##### 辅助函数

```python
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
```

### 模型结构

模型中比较特别的是，采用了 sparse mm 的计算方法。

```python
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
```

```python
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # Sparse matrix multiplication，https://github.com/tkipf/pygcn/issues/19
        # output = torch.spmm(adj, support)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
```

### 训练细节

为了保持邻接矩阵的完整，在训练中我们使用 idx 的方式来分割训练，验证和测试集。

```python
idx_train = range(140)
idx_val = range(200, 500)
idx_test = range(500, 1500)
```

更新梯度或者计算指标时:

```python
output = model(features, adj)
loss_train = F.nll_loss(output[idx_train], labels[idx_train])
acc_train = accuracy(output[idx_train], labels[idx_train])
```

## 参考

[GCN 的 normalization](https://blog.csdn.net/qq_30636613/article/details/105449531)

[论文作者自己的博客解读](http://tkipf.github.io/graph-convolutional-networks/)

[知乎 何时能懂你的心——图卷积神经网络（GCN）](https://zhuanlan.zhihu.com/p/71200936)

