---
title: Kubernetes 详细教程(上)
date: 2022-12-25
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- Linux
mathjax: true
comments: 
---

来源笔记 [Kubernetes(K8S) 入门进阶实战完整教程，黑马程序员 K8S 全套教程](https://www.bilibili.com/video/BV1Qv41167ck/?p=39&spm_id_from=pageDriver&vd_source=4418d5cd5be787be7e3ff4138eeb9b0a)

## 1. Kubernetes 介绍

### 1.1 应用部署方式演变

在部署应用程序的方式上，主要经历了三个时代：

-  **传统部署** ：互联网早期，会直接将应用程序部署在物理机上

  > 优点：简单，不需要其它技术的参与
  >
  > 缺点：不能为应用程序定义资源使用边界，很难合理地分配计算资源，而且程序之间容易产生影响

-  **虚拟化部署** ：可以在一台物理机上运行多个虚拟机，每个虚拟机都是独立的一个环境

  > 优点：程序环境不会相互产生影响，提供了一定程度的安全性
  >
  > 缺点：增加了操作系统，浪费了部分资源

-  **容器化部署** ：与虚拟化类似，但是共享了操作系统

  > 优点：
  >
  > 可以保证每个容器拥有自己的文件系统、CPU、内存、进程空间等
  >
  > 运行应用程序所需要的资源都被容器包装，并和底层基础架构解耦
  >
  > 容器化的应用程序可以跨云服务商、跨 Linux 操作系统发行版进行部署

![image-20200505183738289](/assets/img/k8s/image-20200505183738289.png)

容器化部署方式给带来很多的便利，但是也会出现一些问题，比如说：

- 一个容器故障停机了，怎么样让另外一个容器立刻启动去替补停机的容器
- 当并发访问量变大的时候，怎么样做到横向扩展容器数量

这些容器管理的问题统称为 **容器编排** 问题，为了解决这些容器编排问题，就产生了一些容器编排的软件：

-  **Swarm** ：Docker 自己的容器编排工具
-  **Mesos** ：Apache 的一个资源统一管控的工具，需要和 Marathon 结合使用
-  **Kubernetes** ：Google 开源的的容器编排工具

![image-20200524150339551](/assets/img/k8s/image-20200524150339551.png)

### 1.2 kubernetes 简介

![image-20200406232838722](/assets/img/k8s/image-20200406232838722.png)

 

kubernetes，是一个全新的基于容器技术的分布式架构领先方案，是谷歌严格保密十几年的秘密武器----Borg 系统的一个开源版本，于 2014 年 9 月发布第一个版本，2015 年 7 月发布第一个正式版本。

kubernetes 的本质是 **一组服务器集群** ，它可以在集群的每个节点上运行特定的程序，来对节点中的容器进行管理。目的是实现资源管理的自动化，主要提供了如下的主要功能：

-  **自我修复** ：一旦某一个容器崩溃，能够在 1 秒中左右迅速启动新的容器
-  **弹性伸缩** ：可以根据需要，自动对集群中正在运行的容器数量进行调整
-  **服务发现** ：服务可以通过自动发现的形式找到它所依赖的服务
-  **负载均衡** ：如果一个服务起动了多个容器，能够自动实现请求的负载均衡
-  **版本回退** ：如果发现新发布的程序版本有问题，可以立即回退到原来的版本
-  **存储编排** ：可以根据容器自身的需求自动创建存储卷

### 1.3 kubernetes 组件

一个 kubernetes 集群主要是由 **控制节点(master)** 、 **工作节点(node)** 构成，每个节点上都会安装不同的组件。

 **master：集群的控制平面，负责集群的决策 ( 管理 )** 

>  **ApiServer**  : 资源操作的唯一入口，接收用户输入的命令，提供认证、授权、API 注册和发现等机制
>
>  **Scheduler**  : 负责集群资源调度，按照预定的调度策略将 Pod 调度到相应的 node 节点上
>
>  **ControllerManager**  : 负责维护集群的状态，比如程序部署安排、故障检测、自动扩展、滚动更新等
>
>  **Etcd**  ：负责存储集群中各种资源对象的信息

 **node：集群的数据平面，负责为容器提供运行环境 ( 干活 )** 

>  **Kubelet**  : 负责维护容器的生命周期，即通过控制 docker，来创建、更新、销毁容器
>
>  **KubeProxy**  : 负责提供集群内部的服务发现和负载均衡
>
>  **Docker**  : 负责节点上容器的各种操作

![相关图片](/assets/img/k8s/image-20200406184656917.png =x300)

下面，以部署一个 nginx 服务来说明 kubernetes 系统各个组件调用关系：

1. 首先要明确，一旦 kubernetes 环境启动之后，master 和 node 都会将自身的信息存储到 etcd 数据库中

2. 一个 nginx 服务的安装请求会首先被发送到 master 节点的 apiServer 组件

3. apiServer 组件会调用 scheduler 组件来决定到底应该把这个服务安装到哪个 node 节点上

   在此时，它会从 etcd 中读取各个 node 节点的信息，然后按照一定的算法进行选择，并将结果告知 apiServer

4. apiServer 调用 controller-manager 去调度 Node 节点安装 nginx 服务

5. kubelet 接收到指令后，会通知 docker，然后由 docker 来启动一个 nginx 的 pod

   pod 是 kubernetes 的最小操作单元，容器必须跑在 pod 中至此，

6. 一个 nginx 服务就运行了，如果需要访问 nginx，就需要通过 kube-proxy 来对 pod 产生访问的代理

这样，外界用户就可以访问集群中的 nginx 服务了

### 1.4 kubernetes 概念

 **Master** ：集群控制节点，每个集群需要至少一个 master 节点负责集群的管控

 **Node** ：工作负载节点，由 master 分配容器到这些 node 工作节点上，然后 node 节点上的 docker 负责容器的运行

 **Pod** ：kubernetes 的最小控制单元，容器都是运行在 pod 中的，一个 pod 中可以有 1 个或者多个容器

 **Controller** ：控制器，通过它来实现对 pod 的管理，比如启动 pod、停止 pod、伸缩 pod 的数量等等

 **Service** ：pod 对外服务的统一入口，下面可以维护者同一类的多个 pod

 **Label** ：标签，用于对 pod 进行分类，同一类 pod 会拥有相同的标签

 **NameSpace** ：命名空间，用来隔离 pod 的运行环境

## 2. kubernetes 集群环境搭建

### 2.1 前置知识点

目前生产部署 Kubernetes 集群主要有两种方式：

 **kubeadm**  （推荐）

Kubeadm 是一个 K8s 部署工具，提供 kubeadm init 和 kubeadm join，用于快速部署 Kubernetes 集群。

官方地址：https://kubernetes.io/docs/reference/setup-tools/kubeadm/kubeadm/

 **二进制包** 

从 github 下载发行版的二进制包，手动部署每个组件，组成 Kubernetes 集群。

Kubeadm 降低部署门槛，但屏蔽了很多细节，遇到问题很难排查。如果想更容易可控，推荐使用二进制包部署 Kubernetes 集群，虽然手动部署麻烦点，期间可以学习很多工作原理，也利于后期维护。

![相关图片](/assets/img/k8s/image-20200404094800622.png )

### 2.2 kubeadm 部署方式介绍

kubeadm 是官方社区推出的一个用于快速部署 kubernetes 集群的工具，这个工具能通过两条指令完成一个 kubernetes 集群的部署：

- 创建一个 Master 节点 kubeadm init
- 将 Node 节点加入到当前集群中$ kubeadm join <Master 节点的 IP 和端口>

### 2.3 安装要求

在开始之前，部署 Kubernetes 集群机器需要满足以下几个条件：

- 一台或多台机器，操作系统 CentOS7.x-86_x64
- 硬件配置：2GB 或更多 RAM，2 个 CPU 或更多 CPU，硬盘 30GB 或更多
- 集群中所有机器之间网络互通
- 可以访问外网，需要拉取镜像
- 禁止 swap 分区

### 2.4 最终目标

- 在所有节点上安装 Docker 和 kubeadm
- 部署 Kubernetes Master
- 部署容器网络插件
- 部署 Kubernetes Node，将节点加入 Kubernetes 集群中
- 部署 Dashboard Web 页面，可视化查看 Kubernetes 资源

### 2.5 准备环境

 

![相关图片](/assets/img/k8s/image-20210609000002940.png =x300)

| 角色     | IP 地址      | 组件                              |
| :------- | :---------- | :-------------------------------- |
| master01 | 192.168.5.3 | docker，kubectl，kubeadm，kubelet |
| node01   | 192.168.5.4 | docker，kubectl，kubeadm，kubelet |
| node02   | 192.168.5.5 | docker，kubectl，kubeadm，kubelet |

### 2.6 环境初始化

安装 K8S 集群，其实就是安装 K8S 各个组件，并配置。

#### 2.6.1 检查操作系统的版本

```shell
# 此方式下安装 kubernetes 集群要求 Centos 版本要在 7.5 或之上
[root@master ~]# cat /etc/redhat-release
Centos Linux 7.5.1804 (Core)
```

#### 2.6.2 主机名解析

为了方便集群节点间的直接调用，在这个配置一下主机名解析，企业中推荐使用内部 DNS 服务器

```shell
# 主机名成解析 编辑三台服务器的/etc/hosts 文件，添加下面内容
127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4
192.168.88.100 master
192.168.88.101 node1
192.168.88.102 node2
::1         localhost localhost.localdomain localhost6 localhost6.localdomain6
```

#### 2.6.3 时间同步

kubernetes 要求集群中的节点时间必须精确一直，这里使用 chronyd 服务从网络同步时间

企业中建议配置内部的会见同步服务器

```shell
# 启动 chronyd 服务
[root@master ~]# systemctl start chronyd
[root@master ~]# systemctl enable chronyd
[root@master ~]# date
```

#### 2.6.4  禁用 iptable 和 firewalld 服务

kubernetes 和 docker 在运行的中会产生大量的 iptables 规则，为了不让系统规则跟它们混淆，直接关闭系统的规则

```shell
# 1 关闭 firewalld 服务
[root@master ~]# systemctl stop firewalld
[root@master ~]# systemctl disable firewalld
# 2 关闭 iptables 服务
[root@master ~]# systemctl stop iptables
[root@master ~]# systemctl disable iptables
```

#### 2.6.5 禁用 selinux

selinux 是 linux 系统下的一个安全服务，如果不关闭它，在安装集群中会产生各种各样的奇葩问题 `setenforce 0`

```shell
# 编辑 /etc/selinux/config 文件，修改 SELINUX 的值为 disable
# 注意修改完毕之后需要重启 linux 服务
SELINUX=disabled
```

#### 2.6.6 禁用 swap 分区

swap 分区指的是虚拟内存分区，它的作用是物理内存使用完，之后将磁盘空间虚拟成内存来使用，启用 swap 设备会对系统的性能产生非常负面的影响，因此 kubernetes 要求每个节点都要禁用 swap 设备，但是如果因为某些原因确实不能关闭 swap 分区，就需要在集群安装过程中通过明确的参数进行配置说明

```shell
# 编辑分区配置文件/etc/fstab，注释掉 swap 分区一行
# 注意修改完毕之后需要重启 linux 服务
vim /etc/fstab
注释掉 /dev/mapper/centos-swap swap
# /dev/mapper/centos-swap swap
```

#### 2.6.7 修改 linux 的内核参数

```shell
# 修改 linux 的内核采纳数，添加网桥过滤和地址转发功能
# 编辑/etc/sysctl.d/kubernetes.conf 文件，添加如下配置：
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
net.ipv4.ip_forward = 1

# 重新加载配置
[root@master ~]# sysctl -p
# 加载网桥过滤模块
[root@master ~]# modprobe br_netfilter
# 查看网桥过滤模块是否加载成功
[root@master ~]# lsmod | grep br_netfilter
```

#### 2.6.8 配置 ipvs 功能

在 Kubernetes 中 Service 有两种带来模型，一种是基于 iptables 的，一种是基于 ipvs 的两者比较的话，ipvs 的性能明显要高一些，但是如果要使用它，需要手动载入 ipvs 模块

```shell
# 1.安装 ipset 和 ipvsadm
[root@master ~]# yum install ipset ipvsadm -y
# 2.添加需要加载的模块写入脚本文件
[root@master ~]# cat <<EOF> /etc/sysconfig/modules/ipvs.modules
#!/bin/bash
modprobe -- ip_vs
modprobe -- ip_vs_rr
modprobe -- ip_vs_wrr
modprobe -- ip_vs_sh
modprobe -- nf_conntrack_ipv4
EOF
# 3.为脚本添加执行权限
[root@master ~]# chmod +x /etc/sysconfig/modules/ipvs.modules
# 4.执行脚本文件
[root@master ~]# /bin/bash /etc/sysconfig/modules/ipvs.modules
# 5.查看对应的模块是否加载成功
[root@master ~]# lsmod | grep -e ip_vs -e nf_conntrack_ipv4
```

#### 2.6.9 安装 docker

```shell
# 1、切换镜像源
[root@master ~]# wget https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo -O /etc/yum.repos.d/docker-ce.repo

# 2、查看当前镜像源中支持的 docker 版本
[root@master ~]# yum list docker-ce --showduplicates

# 3、安装特定版本的 docker-ce
# 必须制定--setopt=obsoletes=0，否则 yum 会自动安装更高版本
[root@master ~]# yum install --setopt=obsoletes=0 docker-ce-18.06.3.ce-3.el7 -y

# 4、添加一个配置文件
#Docker 在默认情况下使用 Vgroup Driver 为 cgroupfs，而 Kubernetes 推荐使用 systemd 来替代 cgroupfs
[root@master ~]# mkdir /etc/docker
[root@master ~]# cat <<EOF> /etc/docker/daemon.json
{
	"exec-opts": ["native.cgroupdriver=systemd"],
	"registry-mirrors": ["https://kn0t2bca.mirror.aliyuncs.com"]
}
EOF

# 5、启动 dokcer
[root@master ~]# systemctl restart docker
[root@master ~]# systemctl enable docker
```

#### 2.6.10 安装 Kubernetes 组件

```shell
# 1、由于 kubernetes 的镜像在国外，速度比较慢，这里切换成国内的镜像源
# 2、编辑/etc/yum.repos.d/kubernetes.repo,添加下面的配置
[kubernetes]
name=Kubernetes
baseurl=http://mirrors.aliyun.com/kubernetes/yum/repos/kubernetes-el7-x86_64
enabled=1
gpgchech=0
repo_gpgcheck=0
gpgkey=http://mirrors.aliyun.com/kubernetes/yum/doc/yum-key.gpg
			http://mirrors.aliyun.com/kubernetes/yum/doc/rpm-package-key.gpg

# 3、安装 kubeadm、kubelet 和 kubectl
[root@master ~]# yum install --setopt=obsoletes=0 kubeadm-1.17.4-0 kubelet-1.17.4-0 kubectl-1.17.4-0 -y

# 4、配置 kubelet 的 cgroup
#编辑/etc/sysconfig/kubelet, 添加下面的配置
KUBELET_CGROUP_ARGS="--cgroup-driver=systemd"
KUBE_PROXY_MODE="ipvs"

# 5、设置 kubelet 开机自启
[root@master ~]# systemctl enable kubelet
```

#### 2.6.11 准备集群镜像

```shell
# 在安装 kubernetes 集群之前，必须要提前准备好集群需要的镜像，所需镜像可以通过下面命令查看
[root@master ~]# kubeadm config images list

# 下载镜像
# 此镜像 kubernetes 的仓库中，由于网络原因，无法连接，下面提供了一种替换方案
images=(
	kube-apiserver:v1.17.7
	kube-controller-manager:v1.17.4
	kube-scheduler:v1.17.4
	kube-proxy:v1.17.4
	pause:3.1
	etcd:3.4.3-0
	coredns:1.6.5
)

for imageName in ${images[@]};do
	docker pull registry.cn-hangzhou.aliyuncs.com/google_containers/$imageName
	docker tag registry.cn-hangzhou.aliyuncs.com/google_containers/$imageName k8s.gcr.io/$imageName
	docker rmi registry.cn-hangzhou.aliyuncs.com/google_containers/$imageName 
done

```

#### 2.6.11 集群初始化

>下面的操作只需要在 master 节点上执行即可

```shell
# 创建集群
[root@master ~]# kubeadm init \
	--apiserver-advertise-address=192.168.88.100 \
	--image-repository registry.aliyuncs.com/google_containers \
	--kubernetes-version=v1.17.4 \
	--service-cidr=10.96.0.0/12 \
	--pod-network-cidr=10.244.0.0/16
# 创建必要文件
[root@master ~]# mkdir -p $HOME/.kube
[root@master ~]# sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
[root@master ~]# sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

安装成功提示

```bash
Your Kubernetes control-plane has initialized successfully!

To start using your cluster, you need to run the following as a regular user:

  mkdir -p $HOME/.kube
  sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
  sudo chown $(id -u):$(id -g) $HOME/.kube/config

You should now deploy a pod network to the cluster.
Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
  https://kubernetes.io/docs/concepts/cluster-administration/addons/

Then you can join any number of worker nodes by running the following on each as root:

kubeadm join 192.168.88.100:6443 --token q4c7nz.tz79x3psc611pp3q \
    --discovery-token-ca-cert-hash sha256:2862978240fba6e4e135f9ba89c444e96cab1a8f6121d69b13ed437c80c5f051
```

> 根据说明，网 K8S 中添加节点。下面的操作只需要在 node 节点上执行即可

```shell
kubeadm join 192.168.88.100:6443 --token q4c7nz.tz79x3psc611pp3q \
    --discovery-token-ca-cert-hash sha256:2862978240fba6e4e135f9ba89c444e96cab1a8f6121d69b13ed437c80c5f051
```

在 master 上查看节点信息

```shell
[root@master ~]# kubectl get nodes
NAME    STATUS   ROLES     AGE   VERSION
master  NotReady  master   6m    v1.17.4
node1   NotReady   <none>  22s   v1.17.4
node2   NotReady   <none>  19s   v1.17.4
```

#### 2.6.13 安装网络插件，只在 master 节点操作即可

k8s 支持多种网络插件，如 flannel、calico、canal 等，以下选择 flannel。插件使用的是 DaemonSet 控制器，会在每个节点上运行

```shell
wget https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
```

由于外网不好访问，如果出现无法访问的情况，可以直接用下面的 记得文件名是 kube-flannel.yml，位置：/root/kube-flannel.yml 内容：

```shell
https://github.com/flannel-io/flannel/tree/master/Documentation/kube-flannel.yml
```

也可手动拉取指定版本
`docker pull quay.io/coreos/flannel:v0.14.0`              #拉取 flannel 网络，三台主机
`docker images`                  #查看仓库是否拉去下来

 **个人笔记** 
若是集群状态一直是 notready,用下面语句查看原因，
journalctl -f -u kubelet.service
若原因是： cni.go:237] Unable to update cni config: no networks found in /etc/cni/net.d
mkdir -p /etc/cni/net.d                    #创建目录给 flannel 做配置文件

#### 2.6.14 使用 kubeadm reset 重置集群

```
systemctl stop kubelet
rm -rf kubelet.conf 
rm -rf pki/ca.crt
```

```
swapoff -a
kubeadm reset
```

```
rm /etc/cni/net.d/* -f
systemctl daemon-reload
systemctl restart kubelet
systemctl restart docker
iptables -F && iptables -t nat -F && iptables -t mangle -F && iptables -X
```

#### 2.6.15 重启 kubelet 和 docker

master 重新配置 `kubeadm reset` 删除$home/.kube/下所有文件，然后重新根据上面的 init。

```shell
systemctl restart kubelet
systemctl restart docker
```

使用配置文件启动 fannel

```shell
kubectl apply -f kube-flannel.yml
```

等待它安装完毕 发现已经是 集群的状态已经是 Ready

![img](https:////wsl$/Ubuntu-20.04/home/kevin/learning/golang/k8s%E8%AF%A6%E7%BB%86%E6%95%99%E7%A8%8B-%E8%B0%83%E6%95%B4%E7%89%88/images/2232696-20210621233106024-1676033717.png)

#### 2.6.16 kubeadm 中的命令

```shell
# 生成 新的 token
[root@master ~]# kubeadm token create --print-join-command
```

### 2.7 集群测试

#### 2.7.1 创建一个 nginx 服务

```shell
kubectl create deployment nginx  --image=nginx:1.14-alpine
```

#### 2.7.2 暴露端口

```shell
kubectl expose deploy nginx  --port=80 --target-port=80  --type=NodePort
```

#### 2.7.3 查看服务

```shell
kubectl get pod,svc
```

#### 2.7.4 查看 pod

![相关图片](https:////wsl$/Ubuntu-20.04/home/kevin/learning/golang/k8s%E8%AF%A6%E7%BB%86%E6%95%99%E7%A8%8B-%E8%B0%83%E6%95%B4%E7%89%88/images/2232696-20210621233130477-111035427.png )

## 3. 资源管理

### 3.1 资源管理介绍

 **学习 K8S，就是学习如何使用其中的各类资源。** 

在 kubernetes 中，所有的内容都抽象为资源，用户需要通过操作资源来管理 kubernetes。

> kubernetes 的本质上就是一个集群系统，用户可以在集群中部署各种服务，所谓的部署服务，其实就是在 kubernetes 集群中运行一个个的容器，并将指定的程序跑在容器中。
>
> kubernetes 的最小管理单元是 pod 而不是容器，所以只能将容器放在`Pod`中，而 kubernetes 一般也不会直接管理 Pod，而是通过`Pod 控制器`来管理 Pod 的。
>
> Pod 可以提供服务之后，就要考虑如何访问 Pod 中服务，kubernetes 提供了`Service`资源实现这个功能。
>
> 当然，如果 Pod 中程序的数据需要持久化，kubernetes 还提供了各种`存储`系统。

![相关图片](/assets/img/k8s/image-20200406225334627.png =x300)

> 学习 kubernetes 的核心，就是学习如何对集群上的`Pod、Pod 控制器、Service、存储`等各种资源进行操作

### 3.2 YAML 语言介绍

YAML 是一个类似 XML、JSON 的标记性语言。它强调以 **数据** 为中心，并不是以标识语言为重点。因而 YAML 本身的定义比较简单，号称"一种人性化的数据格式语言"。

```xml
<heima>
    <age>15</age>
    <address>Beijing</address>
</heima>
```

```yaml
heima:
  age: 15
  address: Beijing
```

YAML 的语法比较简单，主要有下面几个：

- 大小写敏感
- 使用缩进表示层级关系
-  **缩进不允许使用 tab，只允许空格( 低版本限制 )** 
- 缩进的空格数不重要，只要相同层级的元素左对齐即可
- '#'表示注释

>  **小提示：** 
>
> + 书写 yaml 切记`:` 后面要加一个空格
> + 如果需要将多段 yaml 配置放在一个文件中，中间要使用`---`分隔
> + 下面是一个 yaml 转 json 的网站，可以通过它验证 yaml 是否书写正确
> + https://www.json2yaml.com/convert-yaml-to-json

YAML 支持以下几种数据类型：

- 纯量：单个的、不可再分的值
- 对象：键值对的集合，又称为映射（mapping）/ 哈希（hash） / 字典（dictionary）
- 数组：一组按次序排列的值，又称为序列（sequence） / 列表（list）

```yml
# 纯量, 就是指的一个简单的值，字符串、布尔值、整数、浮点数、Null、时间、日期
# 1 布尔类型
c1: true (或者 True)
# 2 整型
c2: 234
# 3 浮点型
c3: 3.14
# 4 null 类型 
c4: ~  # 使用~表示 null
# 5 日期类型
c5: 2018-02-17    # 日期必须使用 ISO 8601 格式，即 yyyy-MM-dd
# 6 时间类型
c6: 2018-02-17T15:02:31+08:00  # 时间使用 ISO 8601 格式，时间和日期之间使用 T 连接，最后使用+代表时区
# 7 字符串类型
c7: heima     # 简单写法，直接写值 , 如果字符串中间有特殊字符，必须使用双引号或者单引号包裹 
c8: line1
    line2     # 字符串过多的情况可以拆成多行，每一行会被转化成一个空格
```

```yaml
# 对象
# 形式一(推荐):
heima:
  age: 15
  address: Beijing
# 形式二(了解):
heima: {age: 15,address: Beijing}
```

```yaml
# 数组
# 形式一(推荐):
address:
  - 顺义
  - 昌平  
# 形式二(了解):
address: [顺义,昌平]
```

### 3.3 资源管理方式

- 命令式对象管理：直接使用命令去操作 kubernetes 资源

  ``` shell
  kubectl run nginx-pod --image=nginx:1.17.1 --port=80
  ```

- 命令式对象配置：通过命令配置和配置文件去操作 kubernetes 资源

  ```shell
  kubectl create/patch -f nginx-pod.yaml
  ```

- 声明式对象配置：通过 apply 命令和配置文件去 **创建和更新** kubernetes 资源

  ```shell
  kubectl apply -f nginx-pod.yaml
  ```


| 类型           | 操作对象 | 适用环境 | 优点           | 缺点                             |
| :------------- | :------- | :------- | :------------- | :------------------------------- |
| 命令式对象管理 | 对象     | 测试     | 简单           | 只能操作活动对象，无法审计、跟踪 |
| 命令式对象配置 | 文件     | 开发     | 可以审计、跟踪 | 项目大时，配置文件多，操作麻烦   |
| 声明式对象配置 | 目录     | 开发     | 支持目录操作   | 意外情况下难以调试               |

#### 3.3.1 命令式对象管理 [命令笔记整合]

 **kubectl 命令** 

kubectl 是 kubernetes 集群的命令行工具，通过它能够对集群本身进行管理，并能够在集群上进行容器化应用的安装部署。kubectl 命令的语法如下：

```shell
kubectl [command] [type] [name] [flags]
```

 **comand** ：指定要对资源执行的操作，例如 create、get、delete

 **type** ：指定资源类型，比如 deployment、pod、service

 **name** ：指定资源的名称，名称大小写敏感

 **flags** ：指定额外的可选参数

```shell
# 查看所有 pod
kubectl get pod 

# 查看某个 pod
kubectl get pod pod_name

# 查看某个 pod,以 yaml 格式展示结果
kubectl get pod pod_name -o yaml
```

 **资源类型** 

kubernetes 中所有的内容都抽象为资源，可以通过下面的命令进行查看:

```shell
kubectl api-resources
```

经常使用的资源有下面这些：

| 资源分类      | 资源名称                 | 缩写    | 资源作用        |
| :------------ | :----------------------- | :------ | :-------------- |
| 集群级别资源  | nodes                    | no      | 集群组成部分    |
| namespaces    | ns                       | 隔离 Pod |                 |
| pod 资源       | pods                     | po      | 装载容器        |
| pod 资源控制器 | replicationcontrollers   | rc      | 控制 pod 资源     |
|               | replicasets              | rs      | 控制 pod 资源     |
|               | deployments              | deploy  | 控制 pod 资源     |
|               | daemonsets               | ds      | 控制 pod 资源     |
|               | jobs                     |         | 控制 pod 资源     |
|               | cronjobs                 | cj      | 控制 pod 资源     |
|               | horizontalpodautoscalers | hpa     | 控制 pod 资源     |
|               | statefulsets             | sts     | 控制 pod 资源     |
| 服务发现资源  | services                 | svc     | 统一 pod 对外接口 |
|               | ingress                  | ing     | 统一 pod 对外接口 |
| 存储资源      | volumeattachments        |         | 存储            |
|               | persistentvolumes        | pv      | 存储            |
|               | persistentvolumeclaims   | pvc     | 存储            |
| 配置资源      | configmaps               | cm      | 配置            |
|               | secrets                  |         | 配置            |

 **操作** 

kubernetes 允许对资源进行多种操作，可以通过--help 查看详细的操作命令

```shell
kubectl --help
```

经常使用的操作有下面这些：

| 命令分类   | 命令         | 翻译                        | 命令作用                     |
| :--------- | :----------- | :-------------------------- | :--------------------------- |
| 基本命令   | create       | 创建                        | 创建一个资源                 |
|            | edit         | 编辑                        | 编辑一个资源                 |
|            | get          | 获取                        | 获取一个资源                 |
|            | patch        | 更新                        | 更新一个资源                 |
|            | delete       | 删除                        | 删除一个资源                 |
|            | explain      | 解释                        | 展示资源文档                 |
| 运行和调试 | run          | 运行                        | 在集群中运行一个指定的镜像   |
|            | expose       | 暴露                        | 暴露资源为 Service            |
|            | describe     | 描述                        | 显示资源内部信息             |
|            | logs         | 日志输出容器在 pod 中的日志 | 输出容器在 pod 中的日志      |
|            | attach       | 缠绕进入运行中的容器        | 进入运行中的容器             |
|            | exec         | 执行容器中的一个命令        | 执行容器中的一个命令         |
|            | cp           | 复制                        | 在 Pod 内外复制文件            |
|            | rollout      | 首次展示                    | 管理资源的发布               |
|            | scale        | 规模                        | 扩(缩)容 Pod 的数量            |
|            | autoscale    | 自动调整                    | 自动调整 Pod 的数量            |
| 高级命令   | apply        | rc                          | 通过文件对资源进行配置       |
|            | label        | 标签                        | 更新资源上的标签             |
| 其他命令   | cluster-info | 集群信息                    | 显示集群信息                 |
|            | version      | 版本                        | 显示当前 Server 和 Client 的版本 |

下面以一个 namespace / pod 的创建和删除简单演示下命令的使用：

```shell
# 创建一个 namespace
[root@master ~]# kubectl create namespace dev
namespace/dev created

# 获取 namespace
[root@master ~]# kubectl get ns
NAME              STATUS   AGE
default           Active   21h
dev               Active   21s
kube-node-lease   Active   21h
kube-public       Active   21h
kube-system       Active   21h

# 在此 namespace 下创建并运行一个 nginx 的 Pod
[root@master ~]# kubectl run pod --image=nginx:latest -n dev
kubectl run --generator=deployment/apps.v1 is DEPRECATED and will be removed in a future version. Use kubectl run --generator=run-pod/v1 or kubectl create instead.
deployment.apps/pod created

# 查看新创建的 pod
[root@master ~]# kubectl get pod -n dev
NAME  READY   STATUS    RESTARTS   AGE
pod   1/1     Running   0          21s

# 删除指定的 pod
[root@master ~]# kubectl delete pod pod-864f9875b9-pcw7x
pod "pod" deleted

# 删除指定的 namespace
[root@master ~]# kubectl delete ns dev
namespace "dev" deleted
```

#### 3.3.2 命令式对象配置

命令式对象配置就是使用命令配合配置文件一起来操作 kubernetes 资源。

1） 创建一个 nginxpod.yaml，内容如下：

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: dev

---

apiVersion: v1
kind: Pod
metadata:
  name: nginxpod
  namespace: dev
spec:
  containers:
  - name: nginx-containers
    image: nginx:latest
```

2）执行 create 命令，创建资源：

```shell
[root@master ~]# kubectl create -f nginxpod.yaml
namespace/dev created
pod/nginxpod created
```

此时发现创建了两个资源对象，分别是 namespace 和 pod

3）执行 get 命令，查看资源：

```shell
[root@master ~]#  kubectl get -f nginxpod.yaml
NAME            STATUS   AGE
namespace/dev   Active   18s

NAME            READY   STATUS    RESTARTS   AGE
pod/nginxpod    1/1     Running   0          17s
```

这样就显示了两个资源对象的信息

4）执行 delete 命令，删除资源：

```shell
[root@master ~]# kubectl delete -f nginxpod.yaml
namespace "dev" deleted
pod "nginxpod" deleted
```

此时发现两个资源对象被删除了

```
总结:
    命令式对象配置的方式操作资源，可以简单的认为：命令  +  yaml 配置文件（里面是命令需要的各种参数）
```

#### 3.3.3 声明式对象配置

声明式对象配置跟命令式对象配置很相似，但是它只有一个命令 apply。

```shell
# 首先执行一次 kubectl apply -f yaml 文件，发现创建了资源
[root@master ~]#  kubectl apply -f nginxpod.yaml
namespace/dev created
pod/nginxpod created

# 再次执行一次 kubectl apply -f yaml 文件，发现说资源没有变动
[root@master ~]#  kubectl apply -f nginxpod.yaml
namespace/dev unchanged
pod/nginxpod unchanged
```

```shell
总结:
    其实声明式对象配置就是使用 apply 描述一个资源最终的状态（在 yaml 中定义状态）
    使用 apply 操作资源：
        如果资源不存在，就创建，相当于 kubectl create
        如果资源已存在，就更新，相当于 kubectl patch
```

> 扩展：kubectl 可以在 node 节点上运行吗 ?

kubectl 的运行是需要进行配置的，它的配置文件是$HOME/.kube，如果想要在 node 节点运行此命令，需要将 master 上的.kube 文件复制到 node 节点上，即在 master 节点上执行下面操作：

```shell
scp  -r  HOME/.kube   node1: HOME/
```

> 使用推荐: 三种方式应该怎么用 ?

创建/更新资源 使用声明式对象配置 kubectl apply -f XXX.yaml

删除资源 使用命令式对象配置 kubectl delete -f XXX.yaml

查询资源 使用命令式对象管理 kubectl get(describe) 资源名称

## 4. 实战入门

本章节将介绍如何在 kubernetes 集群中部署一个 nginx 服务，并且能够对其进行访问。

### 4.1 Namespace

Namespace 是 kubernetes 系统中的一种非常重要资源，它的主要作用是用来实现 **多套环境的资源隔离** 或者 **多租户的资源隔离** 。

默认情况下，kubernetes 集群中的所有的 Pod 都是可以相互访问的。但是在实际中，可能不想让两个 Pod 之间进行互相的访问，那此时就可以将两个 Pod 划分到不同的 namespace 下。 **kubernetes 通过将集群内部的资源分配到不同的 Namespace 中，可以形成逻辑上的"组"，以方便不同的组的资源进行隔离使用和管理。** 

可以通过 kubernetes 的授权机制，将不同的 namespace 交给不同租户进行管理，这样就实现了多租户的资源隔离。此时还能结合 kubernetes 的资源配额机制，限定不同租户能占用的资源，例如 CPU 使用量、内存使用量等等，来实现租户可用资源的管理。

![image-20200407100850484](/assets/img/k8s/image-20200407100850484.png)

kubernetes 在集群启动之后，会默认创建几个 namespace

```shell
[root@master ~]# kubectl  get namespace
NAME              STATUS   AGE
default           Active   45h     #  所有未指定 Namespace 的对象都会被分配在 default 命名空间
kube-node-lease   Active   45h     #  集群节点之间的心跳维护，v1.13 开始引入
kube-public       Active   45h     #  此命名空间下的资源可以被所有人访问（包括未认证用户）
kube-system       Active   45h     #  所有由 Kubernetes 系统创建的资源都处于这个命名空间
```

下面来看 namespace 资源的具体操作：

#### 4.1.1  **查看** 

查看所有的 ns  命令：`kubectl get ns`

查看指定的 ns   命令：`kubectl get ns ns 名称`

指定输出格式  命令：`kubectl get ns ns 名称  -o 格式参数`

查看 ns 详情  命令：`kubectl describe ns ns 名称`

```shell
# 1 查看所有的 ns  命令：kubectl get ns
[root@master ~]# kubectl get ns
NAME              STATUS   AGE
default           Active   45h
kube-node-lease   Active   45h
kube-public       Active   45h     
kube-system       Active   45h     

# 2 查看指定的 ns   命令：kubectl get ns ns 名称
[root@master ~]# kubectl get ns default
NAME      STATUS   AGE
default   Active   45h

# 3 指定输出格式  命令：kubectl get ns ns 名称  -o 格式参数
# kubernetes 支持的格式有很多，比较常见的是 wide、json、yaml
[root@master ~]# kubectl get ns default -o yaml
apiVersion: v1
kind: Namespace
metadata:
  creationTimestamp: "2021-05-08T04:44:16Z"
  name: default
  resourceVersion: "151"
  selfLink: /api/v1/namespaces/default
  uid: 7405f73a-e486-43d4-9db6-145f1409f090
spec:
  finalizers:
  - kubernetes
status:
  phase: Active
  
# 4 查看 ns 详情  命令：kubectl describe ns ns 名称
[root@master ~]# kubectl describe ns default
Name:         default
Labels:       <none>
Annotations:  <none>
Status:       Active  # Active 命名空间正在使用中  Terminating 正在删除命名空间

# ResourceQuota 针对 namespace 做的资源限制
# LimitRange 针对 namespace 中的每个组件做的资源限制
No resource quota.
No LimitRange resource.
```

#### 4.1.2  **创建** 

```shell
# 创建 namespace
[root@master ~]# kubectl create ns dev
namespace/dev created
```

#### 4.1.3  **删除** 

```shell
# 删除 namespace
[root@master ~]# kubectl delete ns dev
namespace "dev" deleted
```

#### 4.1.4  **配置方式** 

首先准备一个 yaml 文件：ns-dev.yaml

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: dev
```

然后就可以执行对应的创建和删除命令了：

创建：kubectl create -f ns-dev.yaml

删除：kubectl delete -f ns-dev.yaml

### 4.2 Pod

Pod 是 kubernetes 集群进行管理的最小单元，程序要运行必须部署在容器中，而容器必须存在于 Pod 中。

Pod 可以认为是容器的封装，一个 Pod 中可以存在一个或者多个容器。

![相关图片](/assets/img/k8s/image-20200407121501907.png =x300)

kubernetes 在集群启动之后，集群中的各个组件也都是以 Pod 方式运行的。可以通过下面命令查看：

```shell
[root@master ~]# kubectl get pod -n kube-system
NAMESPACE     NAME                             READY   STATUS    RESTARTS   AGE
kube-system   coredns-6955765f44-68g6v         1/1     Running   0          2d1h
kube-system   coredns-6955765f44-cs5r8         1/1     Running   0          2d1h
kube-system   etcd-master                      1/1     Running   0          2d1h
kube-system   kube-apiserver-master            1/1     Running   0          2d1h
kube-system   kube-controller-manager-master   1/1     Running   0          2d1h
kube-system   kube-flannel-ds-amd64-47r25      1/1     Running   0          2d1h
kube-system   kube-flannel-ds-amd64-ls5lh      1/1     Running   0          2d1h
kube-system   kube-proxy-685tk                 1/1     Running   0          2d1h
kube-system   kube-proxy-87spt                 1/1     Running   0          2d1h
kube-system   kube-scheduler-master            1/1     Running   0          2d1h
```

#### 4.2.1 创建并运行

kubernetes 没有提供单独运行 Pod 的命令，都是通过 Pod 控制器来实现的

```shell
# 命令格式： kubectl run (pod 控制器名称) [参数] 
# --image  指定 Pod 的镜像
# --port   指定端口
# --namespace  指定 namespace
[root@master ~]# kubectl run nginx --image=nginx:latest --port=80 --namespace dev 
deployment.apps/nginx created
```

#### 4.2.2 查看 pod 信息

查看 Pod 基本信息 `kubectl get pods -n dev`

查看 Pod 的详细信息 `kubectl describe pod nginx -n dev`

```shell
# 查看 Pod 基本信息
[root@master ~]# kubectl get pods -n dev
NAME    READY   STATUS    RESTARTS   AGE
nginx   1/1     Running   0          43s

# 查看 Pod 的详细信息
[root@master ~]# kubectl describe pod nginx -n dev
```

#### 4.2.3 访问 Pod

```shell
# 获取 podIP
[root@master ~]# kubectl get pods -n dev -o wide
NAME    READY   STATUS    RESTARTS   AGE    IP             NODE    ... 
nginx   1/1     Running   0          190s   10.244.1.23   node1   ...

#访问 POD
[root@master ~]# curl http://10.244.1.23:80
```

#### 4.2.4 删除指定 Pod

> 这是因为当前 Pod 是由 Pod 控制器创建的，控制器会监控 Pod 状况，一旦发现 Pod 死亡，会立即重建。此时要想删除 Pod，必须删除 Pod 控制器

```shell
# 删除指定 Pod
[root@master ~]# kubectl delete pod nginx -n dev
pod "nginx" deleted

# 此时，显示删除 Pod 成功，但是再查询，发现又新产生了一个 
[root@master ~]# kubectl get pods -n dev
NAME    READY   STATUS    RESTARTS   AGE
nginx   1/1     Running   0          21s

# 先来查询一下当前 namespace 下的 Pod 控制器
[root@master ~]# kubectl get deploy -n  dev
NAME    READY   UP-TO-DATE   AVAILABLE   AGE
nginx   1/1     1            1           9m7s

# 接下来，删除此 PodPod 控制器
[root@master ~]# kubectl delete deploy nginx -n dev
deployment.apps "nginx" deleted

# 稍等片刻，再查询 Pod，发现 Pod 被删除了
[root@master ~]# kubectl get pods -n dev
No resources found in dev namespace.
```

#### 4.2.5 配置操作

创建一个 pod-nginx.yaml，内容如下：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  namespace: dev
spec:
  containers:
  - image: nginx:latest
    name: pod
    ports:
    - name: nginx-port
      containerPort: 80
      protocol: TCP
```

然后就可以执行对应的创建和删除命令了：

创建：kubectl create -f pod-nginx.yaml

删除：kubectl delete -f pod-nginx.yaml

### 4.3 Label

> 给 pod 添加标签，并可以根据标签进行选择。

Label 是 kubernetes 系统中的一个重要概念。它的作用就是在资源上添加标识，用来对它们进行区分和选择。

 **Label 的特点：** 

- 一个 Label 会以 key/value 键值对的形式附加到各种对象上，如 Node、Pod、Service 等等
- 一个资源对象可以定义任意数量的 Label ，同一个 Label 也可以被添加到任意数量的资源对象上去
- Label 通常在资源对象定义时确定，当然也可以在对象创建后动态添加或者删除

可以通过 Label 实现资源的多维度分组，以便灵活、方便地进行资源分配、调度、配置、部署等管理工作。

> 一些常用的 Label 示例如下：
>
> - 版本标签："version":"release", "version":"stable"......
> - 环境标签："environment":"dev"，"environment":"test"，"environment":"pro"
> - 架构标签："tier":"frontend"，"tier":"backend"

#####  **查询标签** 

标签定义完毕之后，还要考虑到标签的选择，这就要使用到 Label Selector，即：

Label 用于给某个资源对象定义标识

Label Selector 用于查询和筛选拥有某些标签的资源对象

当前有两种 Label Selector：

- 基于等式的 Label Selector

  `name = slave`: 选择所有包含 Label 中 key="name"且 value="slave"的对象

  `env != production`: 选择所有包括 Label 中的 key="env"且 value 不等于"production"的对象

- 基于集合的 Label Selector

  `name in (master, slave)`: 选择所有包含 Label 中的 key="name"且 value="master"或"slave"的对象

  `name not in (frontend)`: 选择所有包含 Label 中的 key="name"且 value 不等于"frontend"的对象

标签的选择条件可以使用多个，此时将多个 Label Selector 进行组合，使用逗号","进行分隔即可。例如：

name=slave，env!=production

name not in (frontend)，env!=production

#### 4.3.1 命令方式

```shell
# 为 pod 资源打标签
[root@master ~]# kubectl label pod nginx-pod version=1.0 -n dev
pod/nginx-pod labeled

# 为 pod 资源更新标签
[root@master ~]# kubectl label pod nginx-pod version=2.0 -n dev --overwrite
pod/nginx-pod labeled

# 查看标签
[root@master ~]# kubectl get pod nginx-pod  -n dev --show-labels
NAME        READY   STATUS    RESTARTS   AGE   LABELS
nginx-pod   1/1     Running   0          10m   version=2.0

# 筛选标签
[root@master ~]# kubectl get pod -n dev -l version=2.0  --show-labels
NAME        READY   STATUS    RESTARTS   AGE   LABELS
nginx-pod   1/1     Running   0          17m   version=2.0
[root@master ~]# kubectl get pod -n dev -l version!=2.0 --show-labels
No resources found in dev namespace.

#删除标签
[root@master ~]# kubectl label pod nginx-pod version- -n dev
pod/nginx-pod labeled
```

#### 4.3.2 配置方式

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  namespace: dev
  labels:
    version: "3.0" 
    env: "test"
spec:
  containers:
  - image: nginx:latest
    name: pod
    ports:
    - name: nginx-port
      containerPort: 80
      protocol: TCP
```

然后就可以执行对应的更新命令了：kubectl apply -f pod-nginx.yaml

### 4.4 Deployment

在 kubernetes 中，Pod 是最小的控制单元，但是 kubernetes 很少直接控制 Pod，一般都是通过 Pod 控制器来完成的。Pod 控制器用于 pod 的管理，确保 pod 资源符合预期的状态，当 pod 的资源出现故障时，会尝试进行重启或重建 pod。

在 kubernetes 中 Pod 控制器的种类有很多，本章节只介绍一种：Deployment。

![相关图片](/assets/img/k8s/image-20200408193950807.png =x300)

#### 4.4.1 操作命令

```shell
# 命令格式: kubectl create deployment 名称  [参数] 
# --image  指定 pod 的镜像
# --port   指定端口
# --replicas  指定创建 pod 数量
# --namespace  指定 namespace
[root@master ~]# kubectl run nginx --image=nginx:latest --port=80 --replicas=3 -n dev
deployment.apps/nginx created

# 查看创建的 Pod
[root@master ~]# kubectl get pods -n dev
NAME                     READY   STATUS    RESTARTS   AGE
nginx-5ff7956ff6-6k8cb   1/1     Running   0          19s
nginx-5ff7956ff6-jxfjt   1/1     Running   0          19s
nginx-5ff7956ff6-v6jqw   1/1     Running   0          19s

# 查看 deployment 的信息
[root@master ~]# kubectl get deploy -n dev
NAME    READY   UP-TO-DATE   AVAILABLE   AGE
nginx   3/3     3            3           2m42s

# UP-TO-DATE：成功升级的副本数量
# AVAILABLE：可用副本的数量
[root@master ~]# kubectl get deploy -n dev -o wide
NAME    READY UP-TO-DATE  AVAILABLE   AGE     CONTAINERS   IMAGES              SELECTOR
nginx   3/3     3         3           2m51s   nginx        nginx:latest        run=nginx

# 查看 deployment 的详细信息
[root@master ~]# kubectl describe deploy nginx -n dev

  
# 删除 
[root@master ~]# kubectl delete deploy nginx -n dev
deployment.apps "nginx" deleted
```

#### 4.4.2 配置操作

创建一个 deploy-nginx.yaml，内容如下：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  namespace: dev
spec:
  replicas: 3
  selector:
    matchLabels:
      run: nginx
  template:
    metadata:
      labels:
        run: nginx
    spec:
      containers:
      - image: nginx:latest
        name: nginx
        ports:
        - containerPort: 80
          protocol: TCP
```

然后就可以执行对应的创建和删除命令了：

创建：`kubectl create -f deploy-nginx.yaml`

删除：`kubectl delete -f deploy-nginx.yaml`

### 4.5 Service

虽然每个 Pod 都会分配一个单独的 Pod IP，然而却存在如下两问题：

- Pod IP 会随着 Pod 的重建产生变化
- Pod IP 仅仅是集群内可见的虚拟 IP，外部无法访问

> Service 可以看作是一组同类 Pod **对外的访问接口** 。借助 Service，应用可以方便地实现服务发现和负载均衡。

![相关图片](/assets/img/k8s/image-20200408194716912.png )

Service 通过 IP 选择器发现 Pod

#### 4.5.1 创建集群内部可访问的 Service

暴露 Service

```bash
kubectl expose deploy nginx --name=svc-nginx1 --type=ClusterIP --port=80 --target-port=80 -n dev
```

查看 service

```bash
kubectl get svc svc-nginx1 -n dev -o wide

kubectl get svc svc-nginx1 -n dev -o wide
NAME         TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)   AGE     SELECTOR
svc-nginx1   ClusterIP   10.109.179.231   <none>        80/TCP    3m51s   run=nginx
```

这里产生了一个 CLUSTER-IP，这就是 service 的 IP，在 Service 的生命周期中，这个地址是不会变动的，可以通过这个 IP 访问当前 service 对应的 POD

```shell
curl 10.109.179.231:80
```

#### 4.5.2 创建集群外部也可访问的 Service

上面创建的 Service 的 type 类型为 ClusterIP，这个 ip 地址只用集群内部可访问，如果需要创建外部也可以访问的 Service，需要修改 type 为 NodePort。

```shell
kubectl expose deploy nginx --name=svc-nginx2 --type=NodePort --port=80 --target-port=80 -n dev
```

此时查看，会发现出现了 NodePort 类型的 Service，而且有一对 Port（80:31928/TC）

```shell
kubectl get svc  svc-nginx2  -n dev -o wide

NAME          TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)        AGE    SELECTOR
svc-nginx2    NodePort    10.100.94.0      <none>        80:31928/TCP   9s     run=nginx
```

接下来就可以通过集群外的主机访问 节点 IP:31928 访问服务了

例如在的电脑主机上通过浏览器访问下面的地址

#### 4.5.3 删除 Service

```shell
kubectl delete svc svc-nginx1 -n dev 
```

#### 4.5.4 配置方式

创建一个 svc-nginx.yaml，内容如下：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: svc-nginx
  namespace: dev
spec:
  clusterIP: 10.109.179.231 #固定 svc 的内网 ip
  ports:
  - port: 80
    protocol: TCP
    targetPort: 80
  selector:
    run: nginx
  type: ClusterIP
```

然后就可以执行对应的创建和删除命令了：

创建：`kubectl create -f svc-nginx.yaml`

删除：`kubectl delete -f svc-nginx.yaml`

>  **小结** 
>
> 至此，已经掌握了 Namespace、Pod、Deployment、Service 资源的基本操作，有了这些操作，就可以在 kubernetes 集群中实现一个服务的简单部署和访问了，但是如果想要更好的使用 kubernetes，就需要深入学习这几种资源的细节和原理。

## 5. Pod 详解

### 5.1 Pod 介绍

#### 5.1.1 Pod 结构

![相关图片](/assets/img/k8s/image-20200407121501907-1626781151898.png =x300)

每个 Pod 中都可以包含一个或者多个容器，这些容器可以分为两类：

- 用户程序所在的容器，数量可多可少

- Pause 容器，这是每个 Pod 都会有的一个 **根容器** ，它的作用有两个：

  - 可以以它为依据，评估整个 Pod 的健康状态

  - 可以在根容器上设置 Ip 地址，其它容器都此 Ip（Pod IP），以实现 Pod 内部的网路通信

    ```
    这里是 Pod 内部的通讯，Pod 的之间的通讯采用虚拟二层网络技术来实现，我们当前环境用的是 Flannel
    ```

#### 5.1.2 Pod 定义(YAML 架构总览)

下面是 Pod 的资源清单：

```yaml
apiVersion: v1     #必选，版本号，例如 v1
kind: Pod       　 #必选，资源类型，例如 Pod
metadata:       　 #必选，元数据
  name: string     #必选，Pod 名称
  namespace: string  #Pod 所属的命名空间,默认为"default"
  labels:       　　  #自定义标签列表
    - name: string      　          
spec:  #必选，Pod 中容器的详细定义
  containers:  #必选，Pod 中容器列表
  - name: string   #必选，容器名称
    image: string  #必选，容器的镜像名称
    imagePullPolicy: [ Always|Never|IfNotPresent ]  #获取镜像的策略 
    command: [string]   #容器的启动命令列表，如不指定，使用打包时使用的启动命令
    args: [string]      #容器的启动命令参数列表
    workingDir: string  #容器的工作目录
    volumeMounts:       #挂载到容器内部的存储卷配置
    - name: string      #引用 pod 定义的共享存储卷的名称，需用 volumes[]部分定义的的卷名
      mountPath: string #存储卷在容器内 mount 的绝对路径，应少于 512 字符
      readOnly: boolean #是否为只读模式
    ports: #需要暴露的端口库号列表
    - name: string        #端口的名称
      containerPort: int  #容器需要监听的端口号
      hostPort: int       #容器所在主机需要监听的端口号，默认与 Container 相同
      protocol: string    #端口协议，支持 TCP 和 UDP，默认 TCP
    env:   #容器运行前需设置的环境变量列表
    - name: string  #环境变量名称
      value: string #环境变量的值
    resources: #资源限制和请求的设置
      limits:  #资源限制的设置
        cpu: string     #Cpu 的限制，单位为 core 数，将用于 docker run --cpu-shares 参数
        memory: string  #内存限制，单位可以为 Mib/Gib，将用于 docker run --memory 参数
      requests: #资源请求的设置
        cpu: string    #Cpu 请求，容器启动的初始可用数量
        memory: string #内存请求,容器启动的初始可用数量
    lifecycle: #生命周期钩子
        postStart: #容器启动后立即执行此钩子,如果执行失败,会根据重启策略进行重启
        preStop: #容器终止前执行此钩子,无论结果如何,容器都会终止
    livenessProbe:  #对 Pod 内各容器健康检查的设置，当探测无响应几次后将自动重启该容器
      exec:       　 #对 Pod 容器内检查方式设置为 exec 方式
        command: [string]  #exec 方式需要制定的命令或脚本
      httpGet:       #对 Pod 内个容器健康检查方法设置为 HttpGet，需要制定 Path、port
        path: string
        port: number
        host: string
        scheme: string
        HttpHeaders:
        - name: string
          value: string
      tcpSocket:     #对 Pod 内个容器健康检查方式设置为 tcpSocket 方式
         port: number
       initialDelaySeconds: 0       #容器启动完成后首次探测的时间，单位为秒
       timeoutSeconds: 0    　　    #对容器健康检查探测等待响应的超时时间，单位秒，默认 1 秒
       periodSeconds: 0     　　    #对容器监控检查的定期探测时间设置，单位秒，默认 10 秒一次
       successThreshold: 0
       failureThreshold: 0
       securityContext:
         privileged: false
  restartPolicy: [Always | Never | OnFailure]  #Pod 的重启策略
  nodeName: <string> #设置 NodeName 表示将该 Pod 调度到指定到名称的 node 节点上
  nodeSelector: obeject #设置 NodeSelector 表示将该 Pod 调度到包含这个 label 的 node 上
  imagePullSecrets: #Pull 镜像时使用的 secret 名称，以 key：secretkey 格式指定
  - name: string
  hostNetwork: false   #是否使用主机网络模式，默认为 false，如果设置为 true，表示使用宿主机网络
  volumes:   #在该 pod 上定义共享存储卷列表
  - name: string    #共享存储卷名称 （volumes 类型有很多种）
    emptyDir: {}       #类型为 emtyDir 的存储卷，与 Pod 同生命周期的一个临时目录。为空值
    hostPath: string   #类型为 hostPath 的存储卷，表示挂载 Pod 所在宿主机的目录
      path: string      　　        #Pod 所在宿主机的目录，将被用于同期中 mount 的目录
    secret:       　　　#类型为 secret 的存储卷，挂载集群与定义的 secret 对象到容器内部
      scretname: string  
      items:     
      - key: string
        path: string
    configMap:         #类型为 configMap 的存储卷，挂载预定义的 configMap 对象到容器内部
      name: string
      items:
      - key: string
        path: string
```

```yaml
#小提示：
#   在这里，可通过一个命令来查看每种资源的可配置项
#   kubectl explain 资源类型         查看某种资源可以配置的一级属性
#   kubectl explain 资源类型.属性     查看属性的子属性
[root@k8s-master01 ~]# kubectl explain pod
KIND:     Pod
VERSION:  v1
FIELDS:
   apiVersion   <string>
   kind <string>
   metadata     <Object>
   spec <Object>
   status       <Object>

[root@k8s-master01 ~]# kubectl explain pod.metadata
KIND:     Pod
VERSION:  v1
RESOURCE: metadata <Object>
FIELDS:
   annotations  <map[string]string>
   clusterName  <string>
   creationTimestamp    <string>
   deletionGracePeriodSeconds   <integer>
   deletionTimestamp    <string>
   finalizers   <[]string>
   generateName <string>
   generation   <integer>
   labels       <map[string]string>
   managedFields        <[]Object>
   name <string>
   namespace    <string>
   ownerReferences      <[]Object>
   resourceVersion      <string>
   selfLink     <string>
   uid  <string>
```

在 kubernetes 中基本所有资源的一级属性都是一样的，主要包含 5 部分：

- `apiVersion <string> 版本`，由 kubernetes 内部定义，版本号必须可以用 kubectl api-versions 查询到
- `kind <string> 类型`，由 kubernetes 内部定义，版本号必须可以用 kubectl api-resources 查询到
- `metadata <Object> 元数据`，主要是资源标识和说明，常用的有 name、namespace、labels 等
- `spec <Object> 描述`，这是配置中最重要的一部分，里面是对各种资源配置的详细描述
- `status <Object> 状态信息`，里面的内容不需要定义，由 kubernetes 自动生成

在上面的属性中，spec 是接下来研究的重点，继续看下它的常见子属性:

- `containers <[]Object>` 容器列表，用于定义容器的详细信息
- `nodeName <String> `根据 nodeName 的值将 pod 调度到指定的 Node 节点上
- `nodeSelector <map[]>` 根据 NodeSelector 中定义的信息选择将该 Pod 调度到包含这些 label 的 Node 上
- `hostNetwork <boolean>` 是否使用主机网络模式，默认为 false，如果设置为 true，表示使用宿主机网络
- `volumes <[]Object>` 存储卷，用于定义 Pod 上面挂在的存储信息
- `restartPolicy <string>` 重启策略，表示 Pod 在遇到故障的时候的处理策略

### 5.2 Pod 配置

本小节主要来研究`pod.spec.containers`属性，这也是 pod 配置中最为关键的一项配置。

```bash
[root@k8s-master01 ~]# kubectl explain pod.spec.containers
KIND:     Pod
VERSION:  v1
RESOURCE: containers <[]Object>   # 数组，代表可以有多个容器
FIELDS:
   name  <string>     # 容器名称
   image <string>     # 容器需要的镜像地址
   imagePullPolicy  <string> # 镜像拉取策略 
   command  <[]string> # 容器的启动命令列表，如不指定，使用打包时使用的启动命令
   args     <[]string> # 容器的启动命令需要的参数列表
   env      <[]Object> # 容器环境变量的配置
   ports    <[]Object>     # 容器需要暴露的端口号列表
   resources <Object>      # 资源限制和资源请求的设置
```

#### 5.2.1 基本配置

创建 pod-base.yaml 文件，内容如下：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-base
  namespace: dev
  labels:
    user: heima
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
  - name: busybox
    image: busybox:1.30
```

上面定义了一个比较简单 Pod 的配置，里面有两个容器：

- nginx：用 1.17.1 版本的 nginx 镜像创建，（nginx 是一个轻量级 web 容器）
- busybox：用 1.30 版本的 busybox 镜像创建，（busybox 是一个小巧的 linux 命令集合）



```yaml
# 创建 Pod
[root@k8s-master01 pod]# kubectl apply -f pod-base.yaml
pod/pod-base created

# 查看 Pod 状况
# READY 1/2 : 表示当前 Pod 中有 2 个容器，其中 1 个准备就绪，1 个未就绪
# RESTARTS  : 重启次数，因为有 1 个容器故障了，Pod 一直在重启试图恢复它
[root@k8s-master01 pod]# kubectl get pod -n dev
NAME       READY   STATUS    RESTARTS   AGE
pod-base   1/2     Running   4          95s

# 可以通过 describe 查看内部的详情
# 此时已经运行起来了一个基本的 Pod，虽然它暂时有问题
[root@k8s-master01 pod]# kubectl describe pod pod-base -n dev
```

#### 5.2.2 镜像拉取

创建 pod-imagepullpolicy.yaml 文件，内容如下：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-imagepullpolicy
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
    imagePullPolicy: Never # 用于设置镜像拉取策略
  - name: busybox
    image: busybox:1.30
```

imagePullPolicy，用于设置镜像拉取策略，kubernetes 支持配置三种拉取策略：

- Always：总是从远程仓库拉取镜像（一直远程下载）
- IfNotPresent：本地有则使用本地镜像，本地没有则从远程仓库拉取镜像（本地有就本地 本地没远程下载）
- Never：只使用本地镜像，从不去远程仓库拉取，本地没有就报错 （一直使用本地）

> 默认值说明：
>
> 如果镜像 tag 为具体版本号， 默认策略是：IfNotPresent
>
> 如果镜像 tag 为：latest（最终版本） ，默认策略是 always

```yaml
# 创建 Pod
[root@k8s-master01 pod]# kubectl create -f pod-imagepullpolicy.yaml
pod/pod-imagepullpolicy created

# 查看 Pod 详情
# 此时明显可以看到 nginx 镜像有一步 Pulling image "nginx:1.17.1"的过程
[root@k8s-master01 pod]# kubectl describe pod pod-imagepullpolicy -n dev
......
Events:
  Type     Reason     Age               From               Message
  ----     ------     ----              ----               -------
  Normal   Scheduled  <unknown>         default-scheduler  Successfully assigned dev/pod-imagePullPolicy to node1
  Normal   Pulling    32s               kubelet, node1     Pulling image "nginx:1.17.1"
  Normal   Pulled     26s               kubelet, node1     Successfully pulled image "nginx:1.17.1"
  Normal   Created    26s               kubelet, node1     Created container nginx
  Normal   Started    25s               kubelet, node1     Started container nginx
  Normal   Pulled     7s (x3 over 25s)  kubelet, node1     Container image "busybox:1.30" already present on machine
  Normal   Created    7s (x3 over 25s)  kubelet, node1     Created container busybox
  Normal   Started    7s (x3 over 25s)  kubelet, node1     Started container busybox
```

#### 5.2.3 启动命令

在前面的案例中，一直有一个问题没有解决，就是的 busybox 容器一直没有成功运行，那么到底是什么原因导致这个容器的故障呢？

原来 busybox 并不是一个程序，而是类似于一个工具类的集合，kubernetes 集群启动管理后，它会自动关闭。解决方法就是让其一直在运行，这就用到了 command 配置。

创建 pod-command.yaml 文件，内容如下：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-command
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
  - name: busybox
    image: busybox:1.30
    command: ["/bin/sh","-c","touch /tmp/hello.txt;while true;do /bin/echo $(date +%T) >> /tmp/hello.txt; sleep 30; done;"]
```

command，用于在 pod 中的容器初始化完毕之后运行一个命令。

> 稍微解释下上面命令的意思：
>
> "/bin/sh","-c", 使用 sh 执行命令
>
> touch /tmp/hello.txt; 创建一个/tmp/hello.txt 文件
>
> while true;do /bin/echo $(date +%T) >> /tmp/hello.txt; sleep 3; done; 每隔 3 秒向文件中写入当前时间

```bash
# 创建 Pod
[root@k8s-master01 pod]# kubectl create  -f pod-command.yaml
pod/pod-command created

# 查看 Pod 状态
# 此时发现两个 pod 都正常运行了
[root@k8s-master01 pod]# kubectl get pods pod-command -n dev
NAME          READY   STATUS   RESTARTS   AGE
pod-command   2/2     Runing   0          2s
```

进入 pod 中的 busybox 容器，查看文件内容

补充一个命令: kubectl exec  pod 名称 -n 命名空间 -it -c 容器名称 /bin/sh  在容器内部执行命令

使用这个命令就可以进入某个容器的内部，然后进行相关操作了

比如，可以查看 txt 文件的内容

```bash
[root@k8s-master01 pod]# kubectl exec pod-command -n dev -it -c busybox /bin/sh
/ # tail -f /tmp/hello.txt
14:44:19
14:44:22
14:44:25
```

 **特别说明：** 

通过上面发现 command 已经可以完成启动命令和传递参数的功能，为什么这里还要提供一个 args 选项，用于传递参数呢?这其实跟 docker 有点关系，kubernetes 中的 command、args 两项其实是实现覆盖 Dockerfile 中 ENTRYPOINT 的功能。

1. 如果 command 和 args 均没有写，那么用 Dockerfile 的配置。
2. 如果 command 写了，但 args 没有写，那么 Dockerfile 默认的配置会被忽略，执行输入的 command
3. 如果 command 没写，但 args 写了，那么 Dockerfile 中配置的 ENTRYPOINT 的命令会被执行，使用当前 args 的参数
4. 如果 command 和 args 都写了，那么 Dockerfile 的配置被忽略，执行 command 并追加上 args 参数

#### 5.2.4 环境变量

创建 pod-env.yaml 文件，内容如下：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-env
  namespace: dev
spec:
  containers:
  - name: busybox
    image: busybox:1.30
    command: ["/bin/sh","-c","while true;do /bin/echo $(date +%T);sleep 60; done;"]
    env: # 设置环境变量列表
    - name: "username"
      value: "admin"
    - name: "password"
      value: "123456"
```

env，环境变量，用于在 pod 中的容器设置环境变量。

```shell
# 创建 Pod
[root@k8s-master01 ~]# kubectl create -f pod-env.yaml
pod/pod-env created

# 进入容器，输出环境变量
[root@k8s-master01 ~]# kubectl exec pod-env -n dev -c busybox -it /bin/sh
/ # echo $username
admin
/ # echo $password
123456
```

这种方式不是很推荐，推荐将这些配置单独存储在配置文件中，这种方式将在后面介绍。

#### 5.2.5 端口设置

本小节来介绍容器的端口设置，也就是 containers 的 ports 选项。

首先看下 ports 支持的子选项：

```shell
[root@k8s-master01 ~]# kubectl explain pod.spec.containers.ports
KIND:     Pod
VERSION:  v1
RESOURCE: ports <[]Object>
FIELDS:
   name         <string>  # 端口名称，如果指定，必须保证 name 在 pod 中是唯一的		
   containerPort<integer> # 容器要监听的端口(0<x<65536)
   hostPort     <integer> # 容器要在主机上公开的端口，如果设置，主机上只能运行容器的一个副本(一般省略) 
   hostIP       <string>  # 要将外部端口绑定到的主机 IP(一般省略)
   protocol     <string>  # 端口协议。必须是 UDP、TCP 或 SCTP。默认为“TCP”。
```

接下来，编写一个测试案例，创建 pod-ports.yaml

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-ports
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
    ports: # 设置容器暴露的端口列表
    - name: nginx-port
      containerPort: 80
      protocol: TCP
```

访问容器中的程序需要使用的是`Podip:containerPort`

```shell
# 创建 Pod
[root@k8s-master01 ~]# kubectl create -f pod-ports.yaml
pod/pod-ports created

# 查看 pod
# 在下面可以明显看到配置信息
[root@k8s-master01 ~]# kubectl get pod pod-ports -n dev -o yaml
......
spec:
  containers:
  - image: nginx:1.17.1
    imagePullPolicy: IfNotPresent
    name: nginx
    ports:
    - containerPort: 80  # 访问容器中的程序需要使用的是`Podip:containerPort`
      name: nginx-port
      protocol: TCP
......
```



#### 5.2.6 资源配额

容器中的程序要运行，肯定是要占用一定资源的，比如 cpu 和内存等，如果不对某个容器的资源做限制，那么它就可能吃掉大量资源，导致其它容器无法运行。针对这种情况，kubernetes 提供了对内存和 cpu 的资源进行配额的机制，这种机制主要通过 resources 选项实现，他有两个子选项：

-  **limits：** 用于限制运行时容器的最大占用资源，当容器占用资源超过 limits 时会被终止，并进行重启
-  **requests ：** 用于设置容器需要的最小资源，如果环境资源不够，容器将无法启动

可以通过上面两个选项设置资源的上下限。

接下来，编写一个测试案例，创建 pod-resources.yaml

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-resources
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
    resources: # 资源配额
      limits:  # 限制资源（上限）
        cpu: "2" # CPU 限制，单位是 core 数
        memory: "10Gi" # 内存限制
      requests: # 请求资源（下限）
        cpu: "1"  # CPU 限制，单位是 core 数
        memory: "10Mi"  # 内存限制
```

在这对 cpu 和 memory 的单位做一个说明：

- cpu：core 数，可以为整数或小数
- memory： 内存大小，可以使用 Gi、Mi、G、M 等形式

```shell
# 运行 Pod
[root@k8s-master01 ~]# kubectl create  -f pod-resources.yaml
pod/pod-resources created

# 查看发现 pod 运行正常
[root@k8s-master01 ~]# kubectl get pod pod-resources -n dev
NAME            READY   STATUS    RESTARTS   AGE  
pod-resources   1/1     Running   0          39s   

# 接下来，停止 Pod
[root@k8s-master01 ~]# kubectl delete  -f pod-resources.yaml
pod "pod-resources" deleted

# 编辑 pod，修改 resources.requests.memory 的值为 10Gi
[root@k8s-master01 ~]# vim pod-resources.yaml

# 再次启动 pod
[root@k8s-master01 ~]# kubectl create  -f pod-resources.yaml
pod/pod-resources created

# 查看 Pod 状态，发现 Pod 启动失败
[root@k8s-master01 ~]# kubectl get pod pod-resources -n dev -o wide
NAME            READY   STATUS    RESTARTS   AGE          
pod-resources   0/1     Pending   0          20s    

# 查看 pod 详情会发现，如下提示
[root@k8s-master01 ~]# kubectl describe pod pod-resources -n dev
......
Warning  FailedScheduling  35s   default-scheduler  0/3 nodes are available: 1 node(s) had taint {node-role.kubernetes.io/master: }, that the pod didn't tolerate, 2 Insufficient memory.(内存不足)
```

### 5.3 Pod 生命周期

我们一般将 pod 对象从创建至终的这段时间范围称为 pod 的生命周期，它主要包含下面的过程：

- pod 创建过程
- 运行初始化容器（init container）过程
- 运行主容器（main container）
  - 容器启动后钩子（post start）、容器终止前钩子（pre stop）
  - 容器的存活性探测（liveness probe）、就绪性探测（readiness probe）
- pod 终止过程

在整个生命周期中，Pod 会出现 5 种 **状态** （ **相位** ），分别如下：

- 挂起（Pending）：apiserver 已经创建了 pod 资源对象，但它尚未被调度完成或者仍处于下载镜像的过程中
- 运行中（Running）：pod 已经被调度至某节点，并且所有容器都已经被 kubelet 创建完成
-  **成功（Succeeded）** ：pod 中的所有容器都已经成功终止并且不会被重启
- 失败（Failed）：所有容器都已经终止，但至少有一个容器终止失败，即容器返回了非 0 值的退出状态
- 未知（Unknown）：apiserver 无法正常获取到 pod 对象的状态信息，通常由网络通信失败所导致

#### 5.3.1 创建和终止

 **pod 的创建过程** 

1. 用户通过 kubectl 或其他 api 客户端提交需要创建的 pod 信息给 apiServer

2. apiServer 开始生成 pod 对象的信息，并将信息存入 etcd，然后返回确认信息至客户端

3. apiServer 开始反映 etcd 中的 pod 对象的变化，其它组件使用 watch 机制来跟踪检查 apiServer 上的变动

4. scheduler 发现有新的 pod 对象要创建，开始为 Pod 分配主机并将结果信息更新至 apiServer

5. node 节点上的 kubelet 发现有 pod 调度过来，尝试调用 docker 启动容器，并将结果回送至 apiServer

6. apiServer 将接收到的 pod 状态信息存入 etcd 中

   ![相关图片](/assets/img/k8s/image-20200406184656917-1626782168787.png =x300)

 **pod 的终止过程** 

1. 用户向 apiServer 发送删除 pod 对象的命令
2. apiServcer 中的 pod 对象信息会随着时间的推移而更新，在宽限期内（默认 30s），pod 被视为 dead
3. 将 pod 标记为 terminating 状态
4. kubelet 在监控到 pod 对象转为 terminating 状态的同时启动 pod 关闭过程
5. 端点控制器监控到 pod 对象的关闭行为时将其从所有匹配到此端点的 service 资源的端点列表中移除
6. 如果当前 pod 对象定义了 preStop 钩子处理器，则在其标记为 terminating 后即会以同步的方式启动执行
7. pod 对象中的容器进程收到停止信号
8. 宽限期结束后，若 pod 中还存在仍在运行的进程，那么 pod 对象会收到立即终止的信号
9. kubelet 请求 apiServer 将此 pod 资源的宽限期设置为 0 从而完成删除操作，此时 pod 对于用户已不可见

#### 5.3.2 初始化容器

初始化容器是在 pod 的主容器启动之前要运行的容器，主要是做一些主容器的前置工作，它具有两大特征：

1. 初始化容器必须运行完成直至结束，若某初始化容器运行失败，那么 kubernetes 需要重启它直到成功完成
2. 初始化容器必须按照定义的顺序执行，当且仅当前一个成功之后，后面的一个才能运行

初始化容器有很多的应用场景，下面列出的是最常见的几个：

- 提供主容器镜像中不具备的工具程序或自定义代码
- 初始化容器要先于应用容器串行启动并运行完成，因此可用于延后应用容器的启动直至其依赖的条件得到满足

接下来做一个案例，模拟下面这个需求：

假设要以主容器来运行 nginx，但是要求在运行 nginx 之前先要能够连接上 mysql 和 redis 所在服务器

为了简化测试，事先规定好 mysql`(192.168.90.14)`和 redis`(192.168.90.15)`服务器的地址

创建 pod-initcontainer.yaml，内容如下：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-initcontainer
  namespace: dev
spec:
  containers:
  - name: main-container
    image: nginx:1.17.1
    ports: 
    - name: nginx-port
      containerPort: 80
  initContainers:
  - name: test-mysql
    image: busybox:1.30
    command: ['sh', '-c', 'until ping 192.168.88.14 -c 1 ; do echo waiting for mysql...; sleep 2; done;']
  - name: test-redis
    image: busybox:1.30
    command: ['sh', '-c', 'until ping 192.168.88.15 -c 1 ; do echo waiting for reids...; sleep 2; done;']
```

```yaml
# 创建 pod
[root@k8s-master01 ~]# kubectl create -f pod-initcontainer.yaml
pod/pod-initcontainer created

# 查看 pod 状态
# 发现 pod 卡在启动第一个初始化容器过程中，后面的容器不会运行
root@k8s-master01 ~]# kubectl describe pod  pod-initcontainer -n dev
........
Events:
  Type    Reason     Age   From               Message
  ----    ------     ----  ----               -------
  Normal  Scheduled  49s   default-scheduler  Successfully assigned dev/pod-initcontainer to node1
  Normal  Pulled     48s   kubelet, node1     Container image "busybox:1.30" already present on machine
  Normal  Created    48s   kubelet, node1     Created container test-mysql
  Normal  Started    48s   kubelet, node1     Started container test-mysql

# 动态查看 pod
[root@k8s-master01 ~]# kubectl get pods pod-initcontainer -n dev -w
NAME                             READY   STATUS     RESTARTS   AGE
pod-initcontainer                0/1     Init:0/2   0          15s
pod-initcontainer                0/1     Init:1/2   0          52s
pod-initcontainer                0/1     Init:1/2   0          53s
pod-initcontainer                0/1     PodInitializing   0          89s
pod-initcontainer                1/1     Running           0          90s

# 接下来新开一个 shell，为当前服务器新增两个 ip，观察 pod 的变化
[root@k8s-master01 ~]# ifconfig ens33:1 192.168.88.14 netmask 255.255.255.0 up
[root@k8s-master01 ~]# ifconfig ens33:2 192.168.88.15 netmask 255.255.255.0 up
```

#### 5.3.3 钩子函数

钩子函数能够感知自身生命周期中的事件，并在相应的时刻到来时运行用户指定的程序代码。

kubernetes 在主容器的启动之后和停止之前提供了两个钩子函数：

- post start：容器创建之后执行，如果失败了会重启容器
- pre stop ：容器终止之前执行，执行完成之后容器将成功终止，在其完成之前会阻塞删除容器的操作

钩子处理器支持使用下面三种方式定义动作：

- Exec 命令：在容器内执行一次命令

  ```yaml
  ……
    lifecycle:
      postStart: 
        exec:
          command:
          - cat
          - /tmp/healthy
  ……
  ```

- TCPSocket：在当前容器尝试访问指定的 socket

  ```yaml
  ……      
    lifecycle:
      postStart:
        tcpSocket:
          port: 8080
  ……
  ```

- HTTPGet：在当前容器中向某 url 发起 http 请求

  ```yaml
  ……
    lifecycle:
      postStart:
        httpGet:
          path: / #URI 地址
          port: 80 #端口号
          host: 192.168.5.3 #主机地址
          scheme: HTTP #支持的协议，http 或者 https
  ……
  ```

接下来，以 exec 方式为例，演示下钩子函数的使用，创建 pod-hook-exec.yaml 文件，内容如下：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-hook-exec
  namespace: dev
spec:
  containers:
  - name: main-container
    image: nginx:1.17.1
    ports:
    - name: nginx-port
      containerPort: 80
    lifecycle:
      postStart: 
        exec: # 在容器启动的时候执行一个命令，修改掉 nginx 的默认首页内容
          command: ["/bin/sh", "-c", "echo postStart... > /usr/share/nginx/html/index.html"]
      preStop:
        exec: # 在容器停止之前停止 nginx 服务
          command: ["/usr/sbin/nginx","-s","quit"]
```

```yaml
# 创建 pod
[root@k8s-master01 ~]# kubectl create -f pod-hook-exec.yaml
pod/pod-hook-exec created

# 查看 pod
[root@k8s-master01 ~]# kubectl get pods  pod-hook-exec -n dev -o wide
NAME           READY   STATUS     RESTARTS   AGE    IP            NODE    
pod-hook-exec  1/1     Running    0          29s    10.244.2.48   node2   

# 访问 pod
[root@k8s-master01 ~]# curl 10.244.2.48
postStart...
```

#### 5.3.4 容器探测

容器探测用于检测容器中的应用实例是否正常工作，是保障业务可用性的一种传统机制。如果经过探测，实例的状态不符合预期，那么 kubernetes 就会把该问题实例" 摘除 "，不承担业务流量。

 **kubernetes 提供了两种探针来实现容器探测，分别是：** 

- liveness probes：存活性探针，用于检测应用实例当前是否处于正常运行状态，如果不是，k8s 会重启容器
- readiness probes：就绪性探针，用于检测应用实例当前是否可以接收请求，如果不能，k8s 不会转发流量

>  **livenessProbe 决定是否重启容器，readinessProbe 决定是否将请求转发给容器** 。

上面两种探针目前均支持三种探测方式：

- Exec 命令：在容器内执行一次命令，如果命令执行的退出码为 0，则认为程序正常，否则不正常

  ```yaml
  ……
    livenessProbe:
      exec:
        command:
        - cat
        - /tmp/healthy
  ……
  ```

- TCPSocket：将会尝试访问一个用户容器的端口，如果能够建立这条连接，则认为程序正常，否则不正常

  ```yaml
  ……      
    livenessProbe:
      tcpSocket:
        port: 8080
  ……
  ```

- HTTPGet：调用容器内 Web 应用的 URL，如果返回的状态码在 200 和 399 之间，则认为程序正常，否则不正常

  ```yaml
  ……
    livenessProbe:
      httpGet:
        path: / #URI 地址
        port: 80 #端口号
        host: 127.0.0.1 #主机地址
        scheme: HTTP #支持的协议，http 或者 https
  ……
  ```

下面以 liveness probes 为例，做几个演示：

 **方式一：Exec** 

创建 pod-liveness-exec.yaml

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-liveness-exec
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
    ports: 
    - name: nginx-port
      containerPort: 80
    livenessProbe:
      exec:
        command: ["/bin/cat","/tmp/hello.txt"] # 执行一个查看文件的命令
```

创建 pod，观察效果

```yaml
# 创建 Pod
[root@k8s-master01 ~]# kubectl create -f pod-liveness-exec.yaml
pod/pod-liveness-exec created

# 查看 Pod 详情
[root@k8s-master01 ~]# kubectl describe pods pod-liveness-exec -n dev
......
  Normal   Created    20s (x2 over 50s)  kubelet, node1     Created container nginx
  Normal   Started    20s (x2 over 50s)  kubelet, node1     Started container nginx
  Normal   Killing    20s                kubelet, node1     Container nginx failed liveness probe, will be restarted
  Warning  Unhealthy  0s (x5 over 40s)   kubelet, node1     Liveness probe failed: cat: can't open '/tmp/hello11.txt': No such file or directory
  
# 观察上面的信息就会发现 nginx 容器启动之后就进行了健康检查
# 检查失败之后，容器被 kill 掉，然后尝试进行重启（这是重启策略的作用，后面讲解）
# 稍等一会之后，再观察 pod 信息，就可以看到 RESTARTS 不再是 0，而是一直增长
[root@k8s-master01 ~]# kubectl get pods pod-liveness-exec -n dev
NAME                READY   STATUS             RESTARTS   AGE
pod-liveness-exec   0/1     CrashLoopBackOff   2          3m19s

# 当然接下来，可以修改成一个存在的文件，比如/tmp/hello.txt，再试，结果就正常了......
```

 **方式二：TCPSocket** 

创建 pod-liveness-tcpsocket.yaml

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-liveness-tcpsocket
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
    ports: 
    - name: nginx-port
      containerPort: 80
    livenessProbe:
      tcpSocket:
        port: 8080 # 尝试访问 8080 端口
```

创建 pod，观察效果

```yaml
# 创建 Pod
[root@k8s-master01 ~]# kubectl create -f pod-liveness-tcpsocket.yaml
pod/pod-liveness-tcpsocket created

# 查看 Pod 详情
[root@k8s-master01 ~]# kubectl describe pods pod-liveness-tcpsocket -n dev
......
  Normal   Scheduled  31s                            default-scheduler  Successfully assigned dev/pod-liveness-tcpsocket to node2
  Normal   Pulled     <invalid>                      kubelet, node2     Container image "nginx:1.17.1" already present on machine
  Normal   Created    <invalid>                      kubelet, node2     Created container nginx
  Normal   Started    <invalid>                      kubelet, node2     Started container nginx
  Warning  Unhealthy  <invalid> (x2 over <invalid>)  kubelet, node2     Liveness probe failed: dial tcp 10.244.2.44:8080: connect: connection refused
  
# 观察上面的信息，发现尝试访问 8080 端口,但是失败了
# 稍等一会之后，再观察 pod 信息，就可以看到 RESTARTS 不再是 0，而是一直增长
[root@k8s-master01 ~]# kubectl get pods pod-liveness-tcpsocket  -n dev
NAME                     READY   STATUS             RESTARTS   AGE
pod-liveness-tcpsocket   0/1     CrashLoopBackOff   2          3m19s

# 当然接下来，可以修改成一个可以访问的端口，比如 80，再试，结果就正常了......
```

 **方式三：HTTPGet** 

创建 pod-liveness-httpget.yaml

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-liveness-httpget
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
    ports:
    - name: nginx-port
      containerPort: 80
    livenessProbe:
      httpGet:  # 其实就是访问 http://127.0.0.1:80/hello  
        scheme: HTTP #支持的协议，http 或者 https
        port: 80 #端口号
        path: /hello #URI 地址
```

创建 pod，观察效果

```yaml
# 创建 Pod
[root@k8s-master01 ~]# kubectl create -f pod-liveness-httpget.yaml
pod/pod-liveness-httpget created

# 查看 Pod 详情
[root@k8s-master01 ~]# kubectl describe pod pod-liveness-httpget -n dev
.......
  Normal   Pulled     6s (x3 over 64s)  kubelet, node1     Container image "nginx:1.17.1" already present on machine
  Normal   Created    6s (x3 over 64s)  kubelet, node1     Created container nginx
  Normal   Started    6s (x3 over 63s)  kubelet, node1     Started container nginx
  Warning  Unhealthy  6s (x6 over 56s)  kubelet, node1     Liveness probe failed: HTTP probe failed with statuscode: 404
  Normal   Killing    6s (x2 over 36s)  kubelet, node1     Container nginx failed liveness probe, will be restarted
  
# 观察上面信息，尝试访问路径，但是未找到,出现 404 错误
# 稍等一会之后，再观察 pod 信息，就可以看到 RESTARTS 不再是 0，而是一直增长
[root@k8s-master01 ~]# kubectl get pod pod-liveness-httpget -n dev
NAME                   READY   STATUS    RESTARTS   AGE
pod-liveness-httpget   1/1     Running   5          3m17s

# 当然接下来，可以修改成一个可以访问的路径 path，比如/，再试，结果就正常了......
```

至此，已经使用 liveness Probe 演示了三种探测方式，但是查看 livenessProbe 的子属性，会发现除了这三种方式，还有一些其他的配置，在这里一并解释下：

```yaml
[root@k8s-master01 ~]# kubectl explain pod.spec.containers.livenessProbe
FIELDS:
   exec <Object>  
   tcpSocket    <Object>
   httpGet      <Object>
   initialDelaySeconds  <integer>  # 容器启动后等待多少秒执行第一次探测
   timeoutSeconds       <integer>  # 探测超时时间。默认 1 秒，最小 1 秒
   periodSeconds        <integer>  # 执行探测的频率。默认是 10 秒，最小 1 秒
   failureThreshold     <integer>  # 连续探测失败多少次才被认定为失败。默认是 3。最小值是 1
   successThreshold     <integer>  # 连续探测成功多少次才被认定为成功。默认是 1
```

下面稍微配置两个，演示下效果即可：

```yaml
[root@k8s-master01 ~]# more pod-liveness-httpget.yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-liveness-httpget
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
    ports:
    - name: nginx-port
      containerPort: 80
    livenessProbe:
      httpGet:
        scheme: HTTP
        port: 80 
        path: /
      initialDelaySeconds: 30 # 容器启动后 30s 开始探测
      timeoutSeconds: 5 # 探测超时时间为 5s
```

#### 5.3.5 重启策略

在上一节中，一旦容器探测出现了问题，kubernetes 就会对容器所在的 Pod 进行重启，其实这是由 pod 的重启策略决定的，pod 的重启策略有 3 种，分别如下：

- Always ：容器失效时，自动重启该容器，这也是默认值。
- OnFailure ： 容器终止运行且退出码不为 0 时重启
- Never ： 不论状态为何，都不重启该容器

重启策略适用于 pod 对象中的所有容器，首次需要重启的容器，将在其需要时立即进行重启，随后再次需要重启的操作将由 kubelet 延迟一段时间后进行，且反复的重启操作的延迟时长以此为 10s、20s、40s、80s、160s 和 300s，300s 是最大延迟时长。

创建 pod-restartpolicy.yaml：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-restartpolicy
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
    ports:
    - name: nginx-port
      containerPort: 80
    livenessProbe:
      httpGet:
        scheme: HTTP
        port: 80
        path: /hello
  restartPolicy: Never # 设置重启策略为 Never
```

运行 Pod 测试

```yaml
# 创建 Pod
[root@k8s-master01 ~]# kubectl create -f pod-restartpolicy.yaml
pod/pod-restartpolicy created

# 查看 Pod 详情，发现 nginx 容器失败
[root@k8s-master01 ~]# kubectl  describe pods pod-restartpolicy  -n dev
......
  Warning  Unhealthy  15s (x3 over 35s)  kubelet, node1     Liveness probe failed: HTTP probe failed with statuscode: 404
  Normal   Killing    15s                kubelet, node1     Container nginx failed liveness probe
  
# 多等一会，再观察 pod 的重启次数，发现一直是 0，并未重启   
[root@k8s-master01 ~]# kubectl  get pods pod-restartpolicy -n dev
NAME                   READY   STATUS    RESTARTS   AGE
pod-restartpolicy      0/1     Running   0          5min42s
```

### 5.4 Pod 调度

在默认情况下，一个 Pod 在哪个 Node 节点上运行，是由 Scheduler 组件采用相应的算法计算出来的，这个过程是不受人工控制的。但是在实际使用中，这并不满足的需求，因为很多情况下，我们想控制某些 Pod 到达某些节点上，那么应该怎么做呢？这就要求了解 kubernetes 对 Pod 的调度规则，kubernetes 提供了四大类调度方式：

-  **自动调度：** 运行在哪个节点上完全由 Scheduler 经过一系列的算法计算得出
-  **定向调度：** NodeName、NodeSelector
-  **亲和性调度：** NodeAffinity、PodAffinity、PodAntiAffinity
-  **污点（容忍）调度：** Taints、Toleration

#### 5.4.1 定向调度

定向调度，指的是利用在 pod 上声明 nodeName 或者 nodeSelector，以此将 Pod 调度到期望的 node 节点上。注意，这里的调度是强制的，这就意味着即使要调度的目标 Node 不存在，也会向上面进行调度，只不过 pod 运行失败而已。

 **NodeName** 

NodeName 用于强制约束将 Pod 调度到指定的 Name 的 Node 节点上。这种方式，其实是直接跳过 Scheduler 的调度逻辑，直接将 Pod 调度到指定名称的节点。

接下来，实验一下：创建一个 pod-nodename.yaml 文件

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-nodename
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
  nodeName: node1 # 指定调度到 node1 节点上
```

```shell
#创建 Pod
[root@k8s-master01 ~]# kubectl create -f pod-nodename.yaml
pod/pod-nodename created

#查看 Pod 调度到 NODE 属性，确实是调度到了 node1 节点上
[root@k8s-master01 ~]# kubectl get pods pod-nodename -n dev -o wide
NAME           READY   STATUS    RESTARTS   AGE   IP            NODE      ......
pod-nodename   1/1     Running   0          56s   10.244.1.87   node1     ......   

# 接下来，删除 pod，修改 nodeName 的值为 node3（并没有 node3 节点）
[root@k8s-master01 ~]# kubectl delete -f pod-nodename.yaml
pod "pod-nodename" deleted
[root@k8s-master01 ~]# vim pod-nodename.yaml
[root@k8s-master01 ~]# kubectl create -f pod-nodename.yaml
pod/pod-nodename created

#再次查看，发现已经向 Node3 节点调度，但是由于不存在 node3 节点，所以 pod 无法正常运行
[root@k8s-master01 ~]# kubectl get pods pod-nodename -n dev -o wide
NAME           READY   STATUS    RESTARTS   AGE   IP       NODE    ......
pod-nodename   0/1     Pending   0          6s    <none>   node3   ......           
```

 **NodeSelector** 

NodeSelector 用于将 pod 调度到添加了指定标签的 node 节点上。它是通过 kubernetes 的 label-selector 机制实现的，也就是说，在 pod 创建之前，会由 scheduler 使用 MatchNodeSelector 调度策略进行 label 匹配，找出目标 node，然后将 pod 调度到目标节点，该匹配规则是强制约束。

接下来，实验一下：

1 首先分别为 node 节点添加标签

```shell
[root@k8s-master01 ~]# kubectl label nodes node1 nodeenv=pro
node/node2 labeled
[root@k8s-master01 ~]# kubectl label nodes node2 nodeenv=test
node/node2 labeled
```

2 创建一个 pod-nodeselector.yaml 文件，并使用它创建 Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-nodeselector
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
  nodeSelector: 
    nodeenv: pro # 指定调度到具有 nodeenv=pro 标签的节点上
```

```shell
#创建 Pod
[root@k8s-master01 ~]# kubectl create -f pod-nodeselector.yaml
pod/pod-nodeselector created

#查看 Pod 调度到 NODE 属性，确实是调度到了 node1 节点上
[root@k8s-master01 ~]# kubectl get pods pod-nodeselector -n dev -o wide
NAME               READY   STATUS    RESTARTS   AGE     IP          NODE    ......
pod-nodeselector   1/1     Running   0          47s   10.244.1.87   node1   ......

# 接下来，删除 pod，修改 nodeSelector 的值为 nodeenv: xxxx（不存在打有此标签的节点）
[root@k8s-master01 ~]# kubectl delete -f pod-nodeselector.yaml
pod "pod-nodeselector" deleted
[root@k8s-master01 ~]# vim pod-nodeselector.yaml
[root@k8s-master01 ~]# kubectl create -f pod-nodeselector.yaml
pod/pod-nodeselector created

#再次查看，发现 pod 无法正常运行,Node 的值为 none
[root@k8s-master01 ~]# kubectl get pods -n dev -o wide
NAME               READY   STATUS    RESTARTS   AGE     IP       NODE    
pod-nodeselector   0/1     Pending   0          2m20s   <none>   <none>

# 查看详情,发现 node selector 匹配失败的提示
[root@k8s-master01 ~]# kubectl describe pods pod-nodeselector -n dev
.......
Events:
  Type     Reason            Age        From               Message
  ----     ------            ----       ----               -------
  Warning  FailedScheduling  <unknown>  default-scheduler  0/3 nodes are available: 3 node(s) didn't match node selector.
```

#### 5.4.2 亲和性调度

上一节，介绍了两种定向调度的方式，使用起来非常方便，但是也有一定的问题，那就是如果没有满足条件的 Node，那么 Pod 将不会被运行，即使在集群中还有可用 Node 列表也不行，这就限制了它的使用场景。

基于上面的问题，kubernetes 还提供了一种亲和性调度（Affinity）。它在 NodeSelector 的基础之上的进行了扩展，可以通过配置的形式，实现优先选择满足条件的 Node 进行调度，如果没有，也可以调度到不满足条件的节点上，使调度更加灵活。

Affinity 主要分为三类：

- nodeAffinity(node 亲和性）: 以 node 为目标，解决 pod 可以调度到哪些 node 的问题
- podAffinity(pod 亲和性) : 以 pod 为目标，解决 pod 可以和哪些已存在的 pod 部署在同一个拓扑域中的问题
- podAntiAffinity(pod 反亲和性) : 以 pod 为目标，解决 pod 不能和哪些已存在 pod 部署在同一个拓扑域中的问题

> 关于亲和性(反亲和性)使用场景的说明：
>
>  **亲和性** ：如果两个应用频繁交互，那就有必要利用亲和性让两个应用的尽可能的靠近，这样可以减少因网络通信而带来的性能损耗。
>
>  **反亲和性** ：当应用的采用多副本部署时，有必要采用反亲和性让各个应用实例打散分布在各个 node 上，这样可以提高服务的高可用性。

 **NodeAffinity** 

首先来看一下`NodeAffinity`的可配置项：

```markdown
pod.spec.affinity.nodeAffinity
  requiredDuringSchedulingIgnoredDuringExecution  Node 节点必须满足指定的所有规则才可以，相当于硬限制
    nodeSelectorTerms  节点选择列表
      matchFields   按节点字段列出的节点选择器要求列表
      matchExpressions   按节点标签列出的节点选择器要求列表(推荐)
        key    键
        values 值
        operat or 关系符 支持 Exists, DoesNotExist, In, NotIn, Gt, Lt
  preferredDuringSchedulingIgnoredDuringExecution 优先调度到满足指定的规则的 Node，相当于软限制 (倾向)
    preference   一个节点选择器项，与相应的权重相关联
      matchFields   按节点字段列出的节点选择器要求列表
      matchExpressions   按节点标签列出的节点选择器要求列表(推荐)
        key    键
        values 值
        operator 关系符 支持 In, NotIn, Exists, DoesNotExist, Gt, Lt
	weight 倾向权重，在范围 1-100。
```

```yaml
关系符的使用说明:

- matchExpressions:
  - key: nodeenv              # 匹配存在标签的 key 为 nodeenv 的节点
    operator: Exists
  - key: nodeenv              # 匹配标签的 key 为 nodeenv,且 value 是"xxx"或"yyy"的节点
    operator: In
    values: ["xxx","yyy"]
  - key: nodeenv              # 匹配标签的 key 为 nodeenv,且 value 大于"xxx"的节点
    operator: Gt
    values: "xxx"
```

接下来首先演示一下`requiredDuringSchedulingIgnoredDuringExecution` ,

创建 pod-nodeaffinity-required.yaml

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-nodeaffinity-required
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
  affinity:  #亲和性设置
    nodeAffinity: #设置 node 亲和性
      requiredDuringSchedulingIgnoredDuringExecution: # 硬限制
        nodeSelectorTerms:
        - matchExpressions: # 匹配 env 的值在["xxx","yyy"]中的标签
          - key: nodeenv
            operator: In
            values: ["xxx","yyy"]
```

```yaml
# 创建 pod
[root@k8s-master01 ~]# kubectl create -f pod-nodeaffinity-required.yaml
pod/pod-nodeaffinity-required created

# 查看 pod 状态 （运行失败）
[root@k8s-master01 ~]# kubectl get pods pod-nodeaffinity-required -n dev -o wide
NAME                        READY   STATUS    RESTARTS   AGE   IP       NODE    ...... 
pod-nodeaffinity-required   0/1     Pending   0          16s   <none>   <none>  ......

# 查看 Pod 的详情
# 发现调度失败，提示 node 选择失败
[root@k8s-master01 ~]# kubectl describe pod pod-nodeaffinity-required -n dev
......
  Warning  FailedScheduling  <unknown>  default-scheduler  0/3 nodes are available: 3 node(s) didn't match node selector.
  Warning  FailedScheduling  <unknown>  default-scheduler  0/3 nodes are available: 3 node(s) didn't match node selector.

#接下来，停止 pod
[root@k8s-master01 ~]# kubectl delete -f pod-nodeaffinity-required.yaml
pod "pod-nodeaffinity-required" deleted

# 修改文件，将 values: ["xxx","yyy"]------> ["pro","yyy"]
[root@k8s-master01 ~]# vim pod-nodeaffinity-required.yaml

# 再次启动
[root@k8s-master01 ~]# kubectl create -f pod-nodeaffinity-required.yaml
pod/pod-nodeaffinity-required created

# 此时查看，发现调度成功，已经将 pod 调度到了 node1 上
[root@k8s-master01 ~]# kubectl get pods pod-nodeaffinity-required -n dev -o wide
NAME                        READY   STATUS    RESTARTS   AGE   IP            NODE  ...... 
pod-nodeaffinity-required   1/1     Running   0          11s   10.244.1.89   node1 ......
```

接下来再演示一下`requiredDuringSchedulingIgnoredDuringExecution` ,

创建 pod-nodeaffinity-preferred.yaml

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-nodeaffinity-preferred
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
  affinity:  #亲和性设置
    nodeAffinity: #设置 node 亲和性
      preferredDuringSchedulingIgnoredDuringExecution: # 软限制
      - weight: 1
        preference:
          matchExpressions: # 匹配 env 的值在["xxx","yyy"]中的标签(当前环境没有)
          - key: nodeenv
            operator: In
            values: ["xxx","yyy"]
```

```yaml
# 创建 pod
[root@k8s-master01 ~]# kubectl create -f pod-nodeaffinity-preferred.yaml
pod/pod-nodeaffinity-preferred created

# 查看 pod 状态 （运行成功）
[root@k8s-master01 ~]# kubectl get pod pod-nodeaffinity-preferred -n dev
NAME                         READY   STATUS    RESTARTS   AGE
pod-nodeaffinity-preferred   1/1     Running   0          40s
```

```
NodeAffinity 规则设置的注意事项：
    1 如果同时定义了 nodeSelector 和 nodeAffinity，那么必须两个条件都得到满足，Pod 才能运行在指定的 Node 上
    2 如果 nodeAffinity 指定了多个 nodeSelectorTerms，那么只需要其中一个能够匹配成功即可
    3 如果一个 nodeSelectorTerms 中有多个 matchExpressions ，则一个节点必须满足所有的才能匹配成功
    4 如果一个 pod 所在的 Node 在 Pod 运行期间其标签发生了改变，不再符合该 Pod 的节点亲和性需求，则系统将忽略此变化
```

 **PodAffinity** 

PodAffinity 主要实现以运行的 Pod 为参照，实现让新创建的 Pod 跟参照 pod 在一个区域的功能。

首先来看一下`PodAffinity`的可配置项：

```markdown
pod.spec.affinity.podAffinity
  requiredDuringSchedulingIgnoredDuringExecution  硬限制
    namespaces       指定参照 pod 的 namespace
    topologyKey      指定调度作用域
    labelSelector    标签选择器
      matchExpressions  按节点标签列出的节点选择器要求列表(推荐)
        key    键
        values 值
        operator 关系符 支持 In, NotIn, Exists, DoesNotExist.
      matchLabels    指多个 matchExpressions 映射的内容
  preferredDuringSchedulingIgnoredDuringExecution 软限制
    podAffinityTerm  选项
      namespaces      
      topologyKey
      labelSelector
        matchExpressions  
          key    键
          values 值
          operator
        matchLabels 
    weight 倾向权重，在范围 1-100
```

```markdown
topologyKey 用于指定调度时作用域,例如:
    如果指定为 kubernetes.io/hostname，那就是以 Node 节点为区分范围
	如果指定为 beta.kubernetes.io/os,则以 Node 节点的操作系统类型来区分
```

接下来，演示下`requiredDuringSchedulingIgnoredDuringExecution`,

1）首先创建一个参照 Pod，pod-podaffinity-target.yaml：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-podaffinity-target
  namespace: dev
  labels:
    podenv: pro #设置标签
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
  nodeName: node1 # 将目标 pod 名确指定到 node1 上
```

```shell
# 启动目标 pod
[root@k8s-master01 ~]# kubectl create -f pod-podaffinity-target.yaml
pod/pod-podaffinity-target created

# 查看 pod 状况
[root@k8s-master01 ~]# kubectl get pods  pod-podaffinity-target -n dev
NAME                     READY   STATUS    RESTARTS   AGE
pod-podaffinity-target   1/1     Running   0          4s
```

2）创建 pod-podaffinity-required.yaml，内容如下：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-podaffinity-required
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
  affinity:  #亲和性设置
    podAffinity: #设置 pod 亲和性
      requiredDuringSchedulingIgnoredDuringExecution: # 硬限制
      - labelSelector:
          matchExpressions: # 匹配 env 的值在["xxx","yyy"]中的标签
          - key: podenv
            operator: In
            values: ["xxx","yyy"]
        topologyKey: kubernetes.io/hostname
```

上面配置表达的意思是：新 Pod 必须要与拥有标签 nodeenv=xxx 或者 nodeenv=yyy 的 pod 在同一 Node 上，显然现在没有这样 pod，接下来，运行测试一下。

```yaml
# 启动 pod
[root@k8s-master01 ~]# kubectl create -f pod-podaffinity-required.yaml
pod/pod-podaffinity-required created

# 查看 pod 状态，发现未运行
[root@k8s-master01 ~]# kubectl get pods pod-podaffinity-required -n dev
NAME                       READY   STATUS    RESTARTS   AGE
pod-podaffinity-required   0/1     Pending   0          9s

# 查看详细信息
[root@k8s-master01 ~]# kubectl describe pods pod-podaffinity-required  -n dev
......
Events:
  Type     Reason            Age        From               Message
  ----     ------            ----       ----               -------
  Warning  FailedScheduling  <unknown>  default-scheduler  0/3 nodes are available: 2 node(s) didn't match pod affinity rules, 1 node(s) had taints that the pod didn't tolerate.

# 接下来修改  values: ["xxx","yyy"]----->values:["pro","yyy"]
# 意思是：新 Pod 必须要与拥有标签 nodeenv=xxx 或者 nodeenv=yyy 的 pod 在同一 Node 上
[root@k8s-master01 ~]# vim pod-podaffinity-required.yaml

# 然后重新创建 pod，查看效果
[root@k8s-master01 ~]# kubectl delete -f  pod-podaffinity-required.yaml
pod "pod-podaffinity-required" de leted
[root@k8s-master01 ~]# kubectl create -f pod-podaffinity-required.yaml
pod/pod-podaffinity-required created

# 发现此时 Pod 运行正常
[root@k8s-master01 ~]# kubectl get pods pod-podaffinity-required -n dev
NAME                       READY   STATUS    RESTARTS   AGE   LABELS
pod-podaffinity-required   1/1     Running   0          6s    <none>
```

关于`PodAffinity`的 `preferredDuringSchedulingIgnoredDuringExecution`，这里不再演示。

 **PodAntiAffinity** 

PodAntiAffinity 主要实现以运行的 Pod 为参照，让新创建的 Pod 跟参照 pod 不在一个区域中的功能。

它的配置方式和选项跟 PodAffinty 是一样的，这里不再做详细解释，直接做一个测试案例。

1）继续使用上个案例中目标 pod

```shell
[root@k8s-master01 ~]# kubectl get pods -n dev -o wide --show-labels
NAME                     READY   STATUS    RESTARTS   AGE     IP            NODE    LABELS
pod-podaffinity-required 1/1     Running   0          3m29s   10.244.1.38   node1   <none>     
pod-podaffinity-target   1/1     Running   0          9m25s   10.244.1.37   node1   podenv=pro
```

2）创建 pod-podantiaffinity-required.yaml，内容如下：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-podantiaffinity-required
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
  affinity:  #亲和性设置
    podAntiAffinity: #设置 pod 亲和性
      requiredDuringSchedulingIgnoredDuringExecution: # 硬限制
      - labelSelector:
          matchExpressions: # 匹配 podenv 的值在["pro"]中的标签
          - key: podenv
            operator: In
            values: ["pro"]
        topologyKey: kubernetes.io/hostname
```

上面配置表达的意思是：新 Pod 必须要与拥有标签 nodeenv=pro 的 pod 不在同一 Node 上，运行测试一下。

```yaml
# 创建 pod
[root@k8s-master01 ~]# kubectl create -f pod-podantiaffinity-required.yaml
pod/pod-podantiaffinity-required created

# 查看 pod
# 发现调度到了 node2 上
[root@k8s-master01 ~]# kubectl get pods pod-podantiaffinity-required -n dev -o wide
NAME                           READY   STATUS    RESTARTS   AGE   IP            NODE   .. 
pod-podantiaffinity-required   1/1     Running   0          30s   10.244.1.96   node2  ..
```

#### 5.4.3 污点和容忍

 **污点（Taints）** 

前面的调度方式都是站在 Pod 的角度上，通过在 Pod 上添加属性，来确定 Pod 是否要调度到指定的 Node 上，其实我们也可以站在 Node 的角度上，通过在 Node 上添加 **污点** 属性，来决定是否允许 Pod 调度过来。

Node 被设置上污点之后就和 Pod 之间存在了一种相斥的关系，进而拒绝 Pod 调度进来，甚至可以将已经存在的 Pod 驱逐出去。

污点的格式为：`key=value:effect`, key 和 value 是污点的标签，effect 描述污点的作用，支持如下三个选项：

- PreferNoSchedule：kubernetes 将尽量避免把 Pod 调度到具有该污点的 Node 上，除非没有其他节点可调度
- NoSchedule：kubernetes 将不会把 Pod 调度到具有该污点的 Node 上，但不会影响当前 Node 上已存在的 Pod
- NoExecute：kubernetes 将不会把 Pod 调度到具有该污点的 Node 上，同时也会将 Node 上已存在的 Pod 驱离

![image-20200605021606508](/assets/img/k8s/image-20200605021831545.png)

使用 kubectl 设置和去除污点的命令示例如下：

```shell
# 设置污点
kubectl taint nodes node1 key=value:effect

# 去除污点
kubectl taint nodes node1 key:effect-

# 去除所有污点
kubectl taint nodes node1 key-
```

接下来，演示下污点的效果：

1. 准备节点 node1（为了演示效果更加明显，暂时停止 node2 节点）
2. 为 node1 节点设置一个污点: `tag=heima:PreferNoSchedule`；然后创建 pod1( pod1 可以 )
3. 修改为 node1 节点设置一个污点: `tag=heima:NoSchedule`；然后创建 pod2( pod1 正常 pod2 失败 )
4. 修改为 node1 节点设置一个污点: `tag=heima:NoExecute`；然后创建 pod3 ( 3 个 pod 都失败 )

```yaml
# 为 node1 设置污点(PreferNoSchedule)
[root@k8s-master01 ~]# kubectl taint nodes node1 tag=heima:PreferNoSchedule

# 创建 pod1
[root@k8s-master01 ~]# kubectl run taint1 --image=nginx:1.17.1 -n dev
[root@k8s-master01 ~]# kubectl get pods -n dev -o wide
NAME                      READY   STATUS    RESTARTS   AGE     IP           NODE   
taint1-7665f7fd85-574h4   1/1     Running   0          2m24s   10.244.1.59   node1    

# 为 node1 设置污点(取消 PreferNoSchedule，设置 NoSchedule)
[root@k8s-master01 ~]# kubectl taint nodes node1 tag:PreferNoSchedule-
[root@k8s-master01 ~]# kubectl taint nodes node1 tag=heima:NoSchedule

# 创建 pod2
[root@k8s-master01 ~]# kubectl run taint2 --image=nginx:1.17.1 -n dev
[root@k8s-master01 ~]# kubectl get pods taint2 -n dev -o wide
NAME                      READY   STATUS    RESTARTS   AGE     IP            NODE
taint1-7665f7fd85-574h4   1/1     Running   0          2m24s   10.244.1.59   node1 
taint2-544694789-6zmlf    0/1     Pending   0          21s     <none>        <none>   

# 为 node1 设置污点(取消 NoSchedule，设置 NoExecute)
[root@k8s-master01 ~]# kubectl taint nodes node1 tag:NoSchedule-
[root@k8s-master01 ~]# kubectl taint nodes node1 tag=heima:NoExecute

# 创建 pod3
[root@k8s-master01 ~]# kubectl run taint3 --image=nginx:1.17.1 -n dev
[root@k8s-master01 ~]# kubectl get pods -n dev -o wide
NAME                      READY   STATUS    RESTARTS   AGE   IP       NODE     NOMINATED 
taint1-7665f7fd85-htkmp   0/1     Pending   0          35s   <none>   <none>   <none>    
taint2-544694789-bn7wb    0/1     Pending   0          35s   <none>   <none>   <none>     
taint3-6d78dbd749-tktkq   0/1     Pending   0          6s    <none>   <none>   <none>     
```

```
小提示：
    使用 kubeadm 搭建的集群，默认就会给 master 节点添加一个污点标记,所以 pod 就不会调度到 master 节点上.
```

 **容忍（Toleration）** 

上面介绍了污点的作用，我们可以在 node 上添加污点用于拒绝 pod 调度上来，但是如果就是想将一个 pod 调度到一个有污点的 node 上去，这时候应该怎么做呢？这就要使用到 **容忍** 。

![image-20200514095913741](/assets/img/k8s/image-20200514095913741.png)

> 污点就是拒绝，容忍就是忽略，Node 通过污点拒绝 pod 调度上去，Pod 通过容忍忽略拒绝

下面先通过一个案例看下效果：

1. 上一小节，已经在 node1 节点上打上了`NoExecute`的污点，此时 pod 是调度不上去的
2. 本小节，可以通过给 pod 添加容忍，然后将其调度上去

创建 pod-toleration.yaml,内容如下

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-toleration
  namespace: dev
spec:
  containers:
  - name: nginx
    image: nginx:1.17.1
  tolerations:      # 添加容忍
  - key: "tag"        # 要容忍的污点的 key
    operator: "Equal" # 操作符
    value: "heima"    # 容忍的污点的 value
    effect: "NoExecute"   # 添加容忍的规则，这里必须和标记的污点规则相同
```

```yaml
# 添加容忍之前的 pod
[root@k8s-master01 ~]# kubectl get pods -n dev -o wide
NAME             READY   STATUS    RESTARTS   AGE   IP       NODE     NOMINATED 
pod-toleration   0/1     Pending   0          3s    <none>   <none>   <none>           

# 添加容忍之后的 pod
[root@k8s-master01 ~]# kubectl get pods -n dev -o wide
NAME             READY   STATUS    RESTARTS   AGE   IP            NODE    NOMINATED
pod-toleration   1/1     Running   0          3s    10.244.1.62   node1   <none>        
```

下面看一下容忍的详细配置:

```yaml
[root@k8s-master01 ~]# kubectl explain pod.spec.tolerations
......
FIELDS:
   key       # 对应着要容忍的污点的键，空意味着匹配所有的键
   value     # 对应着要容忍的污点的值
   operator  # key-value 的运算符，支持 Equal 和 Exists（默认）
   effect    # 对应污点的 effect，空意味着匹配所有影响
   tolerationSeconds   # 容忍时间, 当 effect 为 NoExecute 时生效，表示 pod 在 Node 上的停留时间
```

