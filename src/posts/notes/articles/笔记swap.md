---
title: 内存不够解决方案|Ubuntu 下 swap 空间配置
date: 2023-06-18
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- Linux
---

通常转换大模型时，会出现系统内存不够用的情况，除了购买额外的内存条之外，还可以使用 swap 来暂时提高内存空间。但 swap 占用硬盘空间，并且硬盘读写速度远不如内存，因此还是要谨慎使用。

## 扩大 swap 空间

1. 查看内存使用情况：

```bash
free -h
```

2. 查看 swap 使用情况：

```bash
swapon --show
NAME      TYPE SIZE USED PRIO
/swapfile file   2G   0B   -2
```

3. 如果要提升 swap 的空间，首先关闭 swap：

```bash
sudo swapoff /swapfile
```

4. 分配空间给 swapfile：

```bash
sudo fallocate -l 50G /swapfile
```

5. 配置并开启 swap：

```bash
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 重新配置 swap 空间

如果要减小 swap 空间的话，只能删除 swap 文件，然后重新添加 swapfile：

1. 停用 swap

```bash
sudo swapoff /swapfile
```

2. 移除 swapfile，可以通过 `swapon --show` 来查看 swap 文件位置。

```bash
sudo rm /swapfile
```

3. 配置新的 swapfile，其中 `bs=2G` 表示需要申请的 swapfile 大小为 2G.

```bash
sudo dd if=/dev/zero of=/swapfile bs=2G count=1
```

4. 启动 swap，并设置 swapfile 权限

```
sudo mkswap /swapfile
sudo chmod 0600 /swapfile
sudo swapon /swapfile
```

5. 使用 `swapon --show` 查看是否修改成功情况，而后检查 `/etc/fstab` 文件。

```
cat /etc/fstab
```

![image-20230618215236213](/assets/img/swap/image-20230618215236213.png)

检查 `/etc/fstab` 文件最后是否有一行 `swapfile none swap sw 0 0`。如果没有的话，添加上去

```bash
vim /etc/fstab
```

添加

```
swapfile none swap sw 0 0
```

