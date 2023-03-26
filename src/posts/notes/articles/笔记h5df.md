---
title: H5DF | H5py 文档小整理
date: 2022-06-20
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- 大数据
- NLP
mathjax: true
---

### H5DF 简介

[HDF5 官方文档链接](https://docs.h5py.org/en/stable/high/dataset.html#creating-datasets)，以下对 H5PY 基础操作进行总结。安装：

```shell
conda install h5py
```

#### File

```python
f = h5py.File('myfile.hdf5','w')
```

文件模式 `w/a/r`等于 python io 类似。同时 H5PY 还提供了 SWMR 接口，可以实现但进程写入的同时多进程读取。此外，文件还提供不同的 driver 形式，已针对不同的数据场景：

```python
f = h5py.File('myfile.hdf5', driver=<driver name>, <driver_kwds>)
```

使用结束后需要关闭：`f.close()`。对于一次性读取的可以使用 `with h5py.File() as fp:`

#### Group

group 的功能类似于菜单，Group 下面可以储存 Group 和 dataset 类型数据。

创建组对象：

```python
grp = f.create_group("bar")
grp.name  # '/bar'
subgrp = grp.create_group("baz")
subgrp.name  # '/bar/baz'
```

引用组对象：

```python
grp3 = f['/some/long']
grp3.name  # '/some/long'
```

可以采用 `keys()`, `values()` 遍历对象。但是更推荐使用 `group.visit()` 或 `group.visititems()`。

`group.get()` 类似于字典的 `get`。判断某个 group 是否存在可以尝试使用：`if f['group_name']`，但部分 H5PY 版本不支持。

#### Datasets

h5py dataset 中的数据 API 与 numpy 类似。

创建 DATASET，必须指定 dataset 的 `name`, `shape`, `dtype`。默认的`dtype` 是 `f`。（除非创建 Empty dataset）

```python
dset = f.create_dataset("ints", (100,), dtype='i8')
```

##### dataset 导入数据

创建时候导入

```python
arr = np.arange(100)
dset = f.create_dataset("init", data=arr)
```

对以有 dataset 导入：

```python
dset[0, 1] = 3.0  # print(dset[0, 1])
dset[0,:] = other_data
```

##### dataset 引用数据

注意：不能使用 numpy 中的双层索引，如`dset[0][3]`

```python
dset = f.create_dataset("MyDataset", (10,10,10), 'f')
dset[0,0,0]
dset[0,2:10,1:9:3]
dset[:,::2,5]
dset[0]
dset[1,5]
dset[0,...]
dset[...,6]
dset[()]
```

如果要遍历 dataset 的话，采用 `dataset.len()` 代替 `len(dataset)`

#### chunked

默认 h5df dataset 采用连续空间存储。但可以采用 chunks 来指定数据存储的连续性。如下：

```python
dset = f.create_dataset("chunked", (1000, 1000), chunks=(100, 100))
```

案例中表示 `dset[0:100,0:100]`, `dset[200:300, 400:500]` 将被储存再连续空间。官方建议设置 chunk 块大小为 10kb - 1MB。chunked 的特点就是，当 chunked 中的一个数据被读取的时候，整个 chunked 的数据都会被读取。可以让 h5df 自动设置 chunk 的大小。

```python
dset = f.create_dataset("autochunk", (1000, 1000), chunks=True)
```

如对于深度学习，你可能希望将 chunks 手动设置为每个 samples 的大小，这样会比让系统自动设置快很多。（个人测试最快时候可以差 10 倍读取速度）

#### resizable dataset

```python
dset = f.create_dataset("unlimited", (10, 10), maxshape=(None, 10))
```

需要注意的是 resize 不是像 numpy 的 reshape，缩小维度上的数据会被直接丢弃掉

####  compression

compression 压缩有时候甚至可以提高读取的速度！压缩可以选择不同的方式，以及不同的等级。

```python
df.create_dataset(dset_name, data=data, maxshape=tuple(shape_list), dtype='int64',
                          chunks=tuple(data_shapes[dset_name]), compression="gzip")
```

#### Attibute

向 group 或者 dataset 添加 attibute。通常为 64kb 以下的小数据

```python
dset.attrs["myAttr1"] = [100, 200]
dset.attrs.get("myAttr1")
```

#### 其他

对于字符串内容，需要定义特殊类型

```python
f = h5py.File('foo.hdf5')
>>> dt = h5py.string_dtype(encoding='utf-8')
>>> ds = f.create_dataset('VLDS', (100,100), dtype=dt)
>>> ds.dtype.kind
'O'
>>> h5py.check_string_dtype(ds.dtype)
string_info(encoding='utf-8', length=None)
```

+ 顺序读取 h5df 数据库的速度会比随机读取快很多！快将近 40%

+ 使用 H5DF 配合 dataloader 的时候，建议添加 dataloader `num_workers > 0`，个人测试可以提升最高 5%的训练速度。

+ 如果需要 shuffle 的话，建议同时使用 batchsampler，因为 h5df 的随机访问效率能够被提高。

+ 似乎 h5df 1.10 的多线程处理效率更好？[h5df dataloader 相关](https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/14)

+ group 的访问也是需要时间的（命名空间上的搜索需要花费大量时间），因此如果要遍历 dataset 的话，最好直接采用变量引用到 dataset，而非在 group 基础上使用 group['dset'] 来遍历。如：

  ```python
  # 建议这样写，节省命名空间搜索
  self.feature_dataset = h5py.File(self.h5df_path, 'r', swmr=True)[feature_group_name]
  features = self.feature_dataset[self.sample_ids[index]]
  data = {
      'input_ids': features[0],
      'input_mask': features[1],
      'segment_ids': features[2],
  }
  ```

  上面方式会比下面方式速度快很多：

  ```python
  self.dataset = h5py.File(self.h5df_path, 'r', swmr=True)
  features = self.dataset[feature_group_name][self.sample_ids[index]]
  data = {
      'input_ids': features[0],
      'input_mask': features[1],
      'segment_ids': features[2],
  } 
  ```

