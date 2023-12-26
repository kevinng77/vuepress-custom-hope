---
title: 深度学习多卡训练
date: 2021-06-07
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
---
## 单机单卡

### 梯度累加

单机单卡下可以通过梯度累加来增大训练时候的理论 batch size。如：

```python
for i, (inputs, labels) in enumerate(training_set):
  loss = model(inputs, labels)                   
  loss = loss / accumulation_steps                
  loss.backward()                                 
  if (i+1) % accumulation_steps == 0: 
      global_step += 1
      optimizer.step()                            
      optimizer.clean_grad() 
```

假设目标复现论文采用 `batch size=64`，`lr=4e-5`。复现过程中，在单机单卡上训练时的`batch size=16`。此时有两种方案达到论文复现效果：

+ 方案一：小`batch=16`，小学习率 `lr=1e-5`，高频率 `accumulation_steps=1`。这种情况下，假设每个样本平均损失为 `m` ，并且我们对每个 batch 的 loss 取平均，那么 64 个样本训练完后，模型得以更新的损失为 `64/16 * 1e-5 * m= m*4e-5`
+ 方案二：小 `batch=16`， 大学习率 `lr=4e-5`，低频率`accumulation_steps=4`。这种情况下，每个 step 处理 16 个样本后，得到的损失为 `loss/accumulation_steps=m/4 * 4e-5`；每 64 个样本提供的损失为 `4e-5 * m`

### 混合精度训练

混合精度训练可以为计算密集型任务带来效率上的提升，若任务为 IO 密集型（Batch 小），那么提升注定是有限的。

amp 库下采用混合精度训练，大致流程为：

 **FP32 权重 -> FP16 权重 -> FP16 计算前向 -> FP32 的 loss，扩大 -> 转为 FP16 -> FP16 反向计算梯度 -> 缩放为 FP32 的梯度更新权重**  

[一文搞懂神经网络混合精度训练](https://zhuanlan.zhihu.com/p/84219777)

## 单机多卡

### 多 GPU 操作总结

> 以下为单卡多 GPU，数据并行计算总结

启动训练：

```shell
python -m paddle.distributed.launch --gpus=0,1,2,3 run_train.py
```

代码中调用训练函数：

+ 可以通过 spawn 启动，

```python
n_gpu = len(os.getenv("CUDA_VISIBLE_DEVICES", "").split(","))
paddle.distributed.spawn(self.train, nprocs=n_gpu)
```

在训练中需要初始化平行训练

```python
if paddle.distributed.get_world_size() > 1:
    paddle.distributed.init_parallel_env()
```

模型加载 `model = paddle.DataParellel(model)`，在模型加载好权重之后运行。torch 下采用 `DDP` 加载。

模型加载权重方式不变，同样可以采用 `state_dict` 方式加载

```python
if paddle.distributed.get_world_size() > 1:
    paddle.distributed.init_parallel_env()
```

distrubutedbatchsampler 在多进程启动的时候，会将数据集根据进程数量进行划分。

```python
state_dict = emb.state_dict()
paddle.save(state_dict, "paddle_dy.pdparams")

para_state_dict = paddle.load("paddle_dy.pdparams")
emb.set_state_dict(para_state_dict)
```

在并行训练下，每个 dataloader 的大小都会变为 `total_samples/n_gpu` 因此总的训练 step 也会相应的减少，对于 `lr_scheduler` 会有所影响。

```python
lr_scheduler = paddlenlp.transformers.LinearDecayWithWarmup(
    learning_rate=self.learning_rate,
    total_steps=num_train_steps,
    warmup=int(num_train_steps * self.warmup_proportion))
```

### 进程间通信

进程间通信包括 all_gather, all_reduce 等方法。

如以下通过 all_gather 方式通信后，每个卡上会有一份所有 GPU 上 loss 的数据，此时仅需要在一个 GPU 上进行操作，如打印日志等。避免重复操作。

```python
# 合并多进程数据，进行 logging loss_list=[] losses=paddle.tensor
loss_list = []
if (paddle.distributed.get_world_size() > 1):
    paddle.distributed.all_gather(loss_list, losses)
    if paddle.distributed.get_rank() == 0:
        losses = paddle.stack(loss_list).sum() / len(
            loss_list)
```

### 多卡训练提示

+ batch size 大小

由于异步训练与同步训练对梯度的更新方式不同，因此两者训练时候的理论 bathc size 也是不同的，在每张卡获取部分数据->获取模型参数->各自前项传播->各自反向传播计算梯度后：

+ 异步训练：采取每张卡直接对模型参数进行更新（问：每张卡储存的模型参数一直都是一致的吗？）
+ 同步训练：采取汇总所有的梯度后，一起同步更新，因此同步训练理论上的 batch size 是更大的。

异步训练相当于采用了更频繁的梯度更新，以及较小的 batch size，同时训练中的 loss 波动更大。而同步训练更新较慢，同时还会受到 gpu 间通信同步的影响，但该方案比异步训练更流行。

在 torch 中，DDP 默认采用同步训练。在 paddle 中的 dataparellel 相似，若需要异步训练可以考虑使用 `DataParallel` 下的 `no_sync()` 。

## 其他参考

[单机多卡的正确打开方式（三）：PyTorch](https://zhuanlan.zhihu.com/p/74792767)

[tensorflow 分布式训练必备知识](https://zhuanlan.zhihu.com/p/56991108)

[深度学习多机多卡 batchsize 和学习率的关系](https://blog.csdn.net/qq_37668436/article/details/124293378)

[单机多卡数据并行-DataParallel(DP)](https://support.huaweicloud.com/develop-modelarts/modelarts-distributed-0007.html)

[单机多卡理论](https://zhuanlan.zhihu.com/p/72939003)

[NVIDIA/apex](https://link.zhihu.com/?target=https%3A//github.com/nvidia/apex)：封装了 DistributedDataParallel，AllReduce 架构，建议

[较详细的 torch DDP 解说](https://zhuanlan.zhihu.com/p/467103734)