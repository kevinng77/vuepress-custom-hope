---
title: NLP 模型复现经验总结
date: 2022-06-20
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
mathjax: true
---

以 canine 复现为例，整理总结复现过程中重要的深度学习知识点。

## 简介

本文章对 NLP 论文复现的流程进行总结，包括模型编写、预训练权重转换；微调时候的大型数据处理、对卡训练、混合精度训练；模型推理部署。完整复现仓库：[link](https://github.com/kevinng77/canine_paddle)

## 模型实现

### 模型定位

在一切复现之前，最重要的事情就是判断这篇文章是否值得复现，内容包括：

+  **复现难度：** 你是否具备复现所需要的设备？训练模型所需要的时间是否充足？
+  **模型成就水平：** 模型的指标是否可信，模型拿了哪些 SOTA？评测的数据库是否权威？算法是否优雅？
+  **论文合理性：** 尽量避开有坑的论文。复现前检查论文是否具备完整的源码，浏览下论文仓库 issue 区。过一下论文大致框架。时间允许的话可以跑以下官方源码。

不要迷恋大厂光环，如 google 的 CANINE 论文就充满了槽点，模型架构没什么创新，模型指标也一般般。CANINE 论文中声明该模型比 TydiQA 基线采用的 mBert 高了约 2%，看似不错，但同年发布的其他 TydiQA Top 5 模型比 CANINE 指标还要高出约 7% 到 13%。再者 TydiQA 数据集较为小众，排行榜发布两年到现在也就个位数的模型投榜。因此个人认为 CANINE 有点水论文的嫌疑了。

此外从 TydiQA 源码中的算法来看，该团队的作风有些诡异。如[官方仓库](https://github.com/google-research-datasets/tydiqa) 中的：

+ `run_tydi_lib.py` 中在 GPU 训练过程中插入了频繁的 CPU 计算，大大降低显卡使用率；
+ `postproc.py` 中内存管理极不合理，实际运行官方源码，你需要 120G+ 的内存；然而经过笔者的优化测试，在程序效率不变的情况下，10G+的内存就可以搞定了。
+ `postproc.py` 中计算效率极为缓慢，文件中存在诸多与 TydiQA 任务结果无关的计算，并且没有任何优化计算的方案，笔者通过加入多线程、清理无用中间变量，将数据处理时间从官方文件的 3 小时减少到了仅 20 分钟。

### 模型编写

复现一定不是从绝对的零开始，大部分复现都是基于已有的算子、模型框架进行编写。如 CANINE 采用了 Transformer Encoder 作为主编码器，因此若基于 Bert 模型进行修改，1 天便能完成模型架构的编写。若是从 0 开始自己拼算子，只怕需要花上个一周甚至更久。

算子也可能存在 bug，如 paddle 的 `repeat_interleave` 就存在反向传播时候的 `segmentation fault` 问题

### 预训练权重转换

使用 paddle 或者其他框架时，可以考虑转换已有的 huggingface 预训练权重而非自己训练预训练权重。转换好的权重可以上传至 huggingface.co （记得使用 git lfs）

### 静态图，动态图 

个人喜欢使用动态图构建框架，在实现后转为静态图进行服务部署。

相关链接：[动态图，静态图](https://zhuanlan.zhihu.com/p/191648279)，[飞桨 动态图转静态图 ](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/jit/index_cn.html)，[飞桨产业级实践深度学习-](https://www.paddlepaddle.org.cn/tutorials/projectdetail/1499114)

根据 operator 算子解析执行方式不同，模型可以分为动态图和静态图。

|          | 动态图                                                       | 静态图                                                       |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 编程范式 | 命令式编程范式                                               | 声明式编程范式                                               |
| 执行方式 | 用户无需预先定义完整的网络结构，每写一行网络代码，即可同时获得计算结果。 | 先编译后执行，用户需预先定义完整的网络结构，再对网络结构进行编译优化后，才能执行获得计算结果。 |
| 编程体验 | 可以使用 python 原生的控制流，容易开发、调试                   | 调试不方便，开发有一定门槛，过程中的算子需要有 action（如 .run()）才会执行。 |
| 性能     | 动态图需要在 python 与 C++计算库之间频繁切换，导致了更大的时间开销。 | 一般采用 C++ 性能更优。                                      |
| 模型架构 | 无需使用占位符                                               | 静态图组网阶段并没有实际运行网络，因此并不读入数据，所以需要使用“占位符”（如 paddle.data）指明输入数据的类型、shape 等信息，以完成组网。 |

 **动态图转静态** 

除了手动编写静态图代码外，部分框架也提供了动转静的 API，如 paddle 只需要采用 `paddle.jit.to_static()` 。动态图转静态的一部分优化内容在于使用 python 定义的控制流，如 `for`, `while` 等。

 **paddle 动态图转静态图注意点：** 

+ 需要注意控制流的使用方式，如 for range 等，详细可查看 [支持语法](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/jit/grammar_list_cn.html#8)。

for range 中不支持参数传递 step 值，如 `for x in range(0,n, step):`，可改用 while 替代。

`for a,b,c,d in xs:` 报错，采用 `for x in xs: a,b,c,d = x` 代替

+ 模型中比较常见的控制流转写大多数与 `batch_size` 或者 `x.shape` 相关。

  `x.shape[i]` 的返回值可能是固定的值，也可能是 `None` ，表示动态 shape（如 batch_size）。

  如果比较明确 `x.shape[i]` 对应的是  **动态 shape** ，推荐使用 `paddle.shape(x)[i]`，特别是在生成 position_ids 的时候。

+ 错误：Intel MKL function load error: cpu specific dynamic library is not loaded。环境问题，尝试 `conda install nomkl`

+ `paddle.jit.set_code_level()` 打印转换后的静态图模型代码。

+ 若出现推理时候的维度错误，但模型动态转静态无报错，那么大概错误在于 reshape 或者 unsqeeze 时候维度出问题。1. 尽量少用 `axis=-1`；2. 检查所有的维度变换是否正确；3.检查是否使用 `paddle.shape(x)[i]`来获取动态维度，如 `batch_size`, `len_seq` 等。

## 数据处理

背景：训练数据集太大，一次性加载不进内存中。

### 数据加载

场景：NLP 预训练任务，文本数据集大。

| 储存方式     | 文件格式            | Dataset                      | 备注                                                         | shuffle                     |
| ------------ | ------------------- | ---------------------------- | ------------------------------------------------------------ | --------------------------- |
| 单个大文件   | pickle/jsonl/txt 等 | map dataset/iterable dataset | 使用更好的机器，将所有样本加载到内存中。                     | 可以                        |
| 单个大文件   | pickle/jsonl/txt 等 | map dataset                  | [知乎链接](https://zhuanlan.zhihu.com/p/460012052) ,计算每个样本的 offset，移动指针截取样本。 | 可以                        |
| 多个中文件   | pickle/jsonl/txt 等 | iterable dataset             | 每个文件储存一定数量的样本，内次加载部分样本到内存中，样本加载速度远大于楼下。 | 不能全局 shuffle             |
| 超多个小文件 | pickle/jsonl/txt 等 | map dataset                  | 将每个样本储存在一个文件中，通过索引 文件名获取样本。I/O 的开销非常大。 | 支持全局 shuffle             |
| 数据库       | h5df, tfrecord 等   | map dataset                  | 所有样本储存在同一数据库中，通过索引数据库获取样本。         | 支持全局 shuffle，但影响性能 |

### 相关 API

`Dataset` 

常用的 dataset 有 `MapDataset` 和 `IterableDataset`

```python
class TydiDataset(paddle.io.IterableDataset):
    """
    Construct Dataset Class for Canine TydiQA task.
Args:
    file_names (List[int]): the names of input files.
    sample_dir (str): The directory of folder storing input sample files, which contains a single
        training sample respectively.
"""
    def __init__(self,
                 file_names,
                 sample_dir: str = "/data/tydi/train_samples",
                 ):
        super(TydiDataset, self).__init__()
        self.all_file_path = []
        for file_name in file_names:
            self.all_file_path.append(os.path.join(sample_dir, file_name))
    def __iter__(self):
        if paddle.distributed.get_world_size() == 1:
            file_list = self.all_file_path
        else:
            worker_info = paddle.io.get_worker_info()
            num_files = len(self.all_file_path)
            files_per_worker = int(
                math.ceil(num_files / float(
                    worker_info.num_workers)))

            worker_id = worker_info.id
            iter_start = worker_id * files_per_worker
            iter_end = min(iter_start + files_per_worker, num_files)
            file_list = self.all_file_path[iter_start:iter_end]

        for file_name in file_list:
            with open(file_name,"rb") as fp:
                for sample in pickle.load(fp):
                    yield sample
```

`DataLoader `

+ [num workers 和 dataloader](https://www.zhihu.com/question/422160231/answer/1484767204) - 似乎不太起作用？个人测试对于小 batch size，提高 `num_worker` 会有部分效果提升。

`DistrubutedBatchSampler`

多卡训练下，数据的分配是个关键问题。采用 dataset 时可以手动设置每个卡读取的样本，如上述案例代码。若使用 `MapDataset`，则可以考虑使用 `DistributedBatchSampler` 来自动分配每个卡的样本，以保证样本不重叠。

### H5DF 

Canine 的指标是根据 TydiQA 数据集进行评测的，其中 TydiQA 数据集在数据处理过程中，使用了 tfrecord + tftensor 进行数据存储。为了适配 Paddle 的训练，笔者尝试了使用 H5DF 代替 tfrecord，在数据处理过程中，H5DF 的空间占用与 tfrecord 旗鼓相当，训练过程中，H5DF 也能提供足够的速度，以保证训练效率上与从内存加载数据集的效率相近。

相比于使用 pickle 或者 jsonl + 压缩的方式储存文件。H5DF 的数据处理方式更佳优雅，笔者个人也是推荐采用 h5df 的。关于 H5DF 的经验分享，欢迎参考我的博客 [H5DF | H5py 文档小整理](http://wujiawen.xyz/archives/h5dfh5py%E6%96%87%E6%A1%A3%E5%B0%8F%E6%95%B4%E7%90%86)。更多详细，请参考 [HDF5 官方文档链接](https://docs.h5py.org/en/stable/high/dataset.html#creating-datasets)

## 训练

### 混合精度训练

混合精度训练，短短的几行代码，在节省显存占用 40%+，训练速度翻倍的前提下，能够做到模型准确率几乎不减少！该部分笔者也在个人博客 [混合精度训练](http://wujiawen.xyz/archives/%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83) 中进行了整理。

### 梯度累加

可以近似地模拟大 batch：

```python
global_step = 0
for i, (inputs, labels) in enumerate(training_set):
  loss = model(inputs, labels)                    
  loss = loss / accumulation_steps                
  loss.backward()  
  global_step += 1
  if global_step % accumulation_steps == 0:             
      optimizer.step()                            
      model.zero_grad()                           
```

### 单机多卡训练

需要注意几个概念：模型并行与数据并行，Parameter Server 与 Ring All-Reduce，同步训练与异步训练。通常单机多卡采用数据并行，GPU 之间大多使用 Ring-All-reduce 进行同步。可以参考：[一文说清楚 Tensorflow 分布式训练必备知识](https://zhuanlan.zhihu.com/p/56991108) 等。

对于 torch，单机多卡可以使用 `DataParallel`（PS 架构，异步训练）。或者 `DistributedDataParallel` （Ring All-Reduce，同步训练）；对于 paddle，其中的 `DataParallel` 默认采用的已经是 Ring All-Reduce 了。使用单机多卡训练的操作也比较简介，通常只需要初始化多进程多卡，模型分配等环节即可，以下以 paddle 为例总结（torch 类似）。paddle 可以采用 paddle 的 spawn 或者通过 `paddle.distributed.launch` 开启多进程多卡训练。只要对单机单卡进行简单的修改即可：

```python
# 添加以下语句
from paddle import distributed as dist
if dist.get_world_size() > 1:
    dist.init_parallel_env()
    model = paddle.DataParallel(model)
```

通常多卡训练时的日志操作比较麻烦，常见的方法是使用 `get_rank()` 选择发布日志的进程，而后进行操作，如：

```python
if dist.get_rank() == 0:
    paddle.save(model.state_dict(), os.path.join(self.output_dir, name))
    # 只需要一个进程保存模型即可
    
if dist.get_world_size() > 1:  
    # 对所有进程的数据进行汇总，这边使用 ALL_GATHER,也可以用别的算子， 如 ALL_REDUCE
    dist.all_gather(loss_list, local_loss)
    dist.all_gather(dev_loss_list, dev_loss_tensor)
    dist.all_gather(acc_list, acc)

    if dist.get_rank() == 0:
        logging_loss = (paddle.stack(loss_list).sum() / len(
            loss_list)).item()
        dev_loss = (paddle.stack(dev_loss_list).sum() / len(
            dev_loss_list)).item()
        logging_acc = (paddle.stack(acc_list).sum() / len(
            acc_list)).item()

        logger.info(f"Step {global_step}/{num_train_steps} train loss {logging_loss:.4f}"
                                            f" dev loss {dev_loss:.4f} acc {logging_acc:.2f}% diff {logging_diff:.2f}"
                                            f" time {(time.time() - time1) / 60:.2f}min"
                                            )
        dist.barrier()  # 阻塞其他进程，等待 0 号进程处理完毕。
```

多卡学习需要注意：

+ batch size 与 学习率的调整
+ 多卡下需要注意数据集的分配，可以使用 `DistributeBatchSampler` 来自动分配样本。
+ 混合精度+多卡训练可能要预留一部分的显存出来，不然可能训练到一半发现 OOM 了

## 推理部署

[paddle 产业级推理部署](https://www.paddlepaddle.org.cn/tutorials/projectdetail/3952715)

## 其他

### pdb 调试

> pdb 调试更加灵活，可以使用条件判断语句，在代码中任意选择断点。这通常是 IDE 做不到的。

 **step1：** 在想要进行调试的代码前插入`import pdb; pdb.set_trace()`开启 pdb 调试。

 **step2：** 正常运行.py 文件，在终端会出现下面类似结果，在`(Pdb)`位置后输入相应的 pdb 命令进行调试。

```
> /tmp/tmpm0iw5b5d.py(9)func()
-> two = paddle.full(shape=[1], fill_value=2, dtype='int32')
(Pdb)
```

 **step3：** 在 pdb 交互模式下输入 l、p 等命令可以查看相应的代码、变量，进而排查相关的问题。[pdb 官方](https://docs.python.org/zh-cn/3/library/pdb.html)

```shell
l # 查看当前位置的源代码
p expression # 查看上下文打印 expression 的值，如 p x
s # 执行下一行，进入函数内部
n # 执行下一行，不进入函数
r # 执行代码到函数返回处
b 30 # 在 30 行处设置断点
c # 执行代码，直到下一个断点
q # 退出调试
```

### segmentation fault 报错分析

```shell
ulimit -c  # 查看 core 限制大小
# 0
cat /proc/sys/kernel/core_pattern #  查看 core 生成路径
# core
```

现象：

  我们执行生成 core 的文件并不是在 linux 的目录下，而是在 windows 和 linux 共享的 hgfs 下，导致生成的 core.xxx 都是 0 字节大小。

解决： 把需要运行的程序拷贝到 linux 的[根目录](https://so.csdn.net/so/search?q=根目录&spm=1001.2101.3001.7020)下运行即可。[方案 1](https://zhuanlan.zhihu.com/p/201330829), [方案 2](https://blog.csdn.net/dzhongjie/article/details/80280192)

+ 修改 core 文件大小限制`ulimit -c unlimit` 

+ 重新运行会 segmentation fault 的程序。

+ 目录下生成 core 文件，检查 core 文件大小不为 0

+ ```shell
  gdb `whichi python` core 
  # 用 python 解释器来进行 gdb core 分析
  ```


由于 docker 容器的权限问题，默认无法产生 core 文件，需要做一些配置修改。

在宿主机上修改 core 路径

```bash
echo '/tmp/core.%t.%e.%p' | sudo tee /proc/sys/kernel/core_pattern
```

这是因为系统在产生 Core Dump 文件的时候是根据 /proc/sys/kernel/core_pattern 的设定。而默认的设定是 |/usr/share/apport/apport %p %s %c %P，也就是用管道传给 apport。然而 Docker 里面的系统不一定有装 apport，并且 /proc 又是直接挂到 Docker 里面的，所以我们就得改成放到固定的位置去，也就是 /tmp。

另外，在 docker run 的时候要加上以下参数

```shell
--ulimit core=-1 --security-opt seccomp=unconfined
```

### windows 环境还是很多坑

尝试了 WSL2 下进行开发，还是感觉原先纯 LINUX 的环境更适应一点。Windows WSL 下存在 git 使用不方便，文件磁盘格式问题，文件权限有限等。

linux 和 windows 换行符：导致各种错误，如 pre-commit， sh 文件解析错误，markdown 文件解析错误等。解决方法： vim 中使用 `:set ff=unix` 或者 vscode 等编辑器中设置换行符为 lr