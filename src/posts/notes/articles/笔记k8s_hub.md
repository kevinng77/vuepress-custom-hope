---
title: 小规模实战 - K8S 部署 JupyterHub 集群
date: 2022-09-13
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- DevOps
---

## JupyterHub K8S on Azure K8S services

使用 Azure K8S 服务，搭建多用户 JupyterHub 环境。了解 K8S 基本概念与操作。关于 K8S 基础理论，可以参考 [K8S 视频教程笔记](https://drive.google.com/drive/folders/10ENodosk4XDR10FYTYZFkH24aykAlP8_?usp=share_link) 

### K8S 基础概念

![相关图片](https://d33wubrfki0l68.cloudfront.net/26a177ede4d7b032362289c6fccd448fc4a91174/eb693/images/docs/container_evolution.svg )

容器化部署方式给带来很多的便利，但是也会出现一些问题，比如说：

- 一个容器故障停机了，怎么样让另外一个容器立刻启动去替补停机的容器

- 当并发访问量变大的时候，怎么样做到横向扩展容器数量

这些容器管理的问题统称为容器编排问题，为了解决这些容器编排问题，就产生了一些容器编排的软件：

- Swarm：Docker 自己的容器编排工具
- Mesos：Apache 的一个资源统一管控的工具，需要和 Marathon 结合使
- Kubernetes：Google 开源的的容器编排工具

### K8S 特点

kubernetes 的本质是一组服务器集群，它可以在集群的每个节点上运行特定的程序，来对节点中的容器进行管理。目的是实现资源管理的自动化，主要提供了如下的主要功能：

+ 自我修复：一旦某一个容器崩溃，能够在 1 秒中左右迅速启动新的容器
+ 弹性伸缩：可以根据需要，自动对集群中正在运行的容器数量进行调整
+ 服务发现：服务可以通过自动发现的形式找到它所依赖的服务
+ 负载均衡：如果一个服务起动了多个容器，能够自动实现请求的负载均衡
+ 版本回退：如果发现新发布的程序版本有问题，可以立即回退到原来的版本
+ 存储编排：可以根据容器自身的需求自动创建存储卷下文中，

我们会通过 JupyterHub 环境，对以上提到的这些特点进行实践与探索。K8S 组件参考[官网](https://kubernetes.io/docs/concepts/overview/components/)

![相关图片](https://d33wubrfki0l68.cloudfront.net/2475489eaf20163ec0f54ddc1d92aa8d4c87c96b/e7c81/images/docs/components-of-kubernetes.svg )

一个 kubernetes 集群主要是由控制节点(master)、 **工作节点(node)** 构成，每个节点上都会安装不同的组件。

 **master：集群的控制平面，负责集群的决策 ( 管理 )** 

> ApiServer : 资源操作的唯一入口，接收用户输入的命令，提供认证、授权、API 注册和发现等机制
>
> Scheduler : 负责集群资源调度，按照预定的调度策略将 Pod 调度到相应的 node 节点上
>
> ControllerManager : 负责维护集群的状态，比如程序部署安排、故障检测、自动扩展、滚动更新等
>
> Etcd ：负责存储集群中各种资源对象的信息

 **node：集群的数据平面，负责为容器提供运行环境 ( 干活 )** 

> Kubelet : 负责维护容器的生命周期，即通过控制 docker，来创建、更新、销毁容器
>
> KubeProxy : 负责提供集群内部的服务发现和负载均衡
>
> Docker : 负责节点上容器的各种操作

### K8S 基础概念

[Kubectl](https://kubernetes.io/docs/reference/kubectl/cheatsheet/) 

Kubectl 是 K8S 的命令管理工具，可以支持：

- 查询资源 `kubectl get pods`, `kubectl describe pod_name`

- 删除资源 `kubectl delete namespace dev`

- 创建资源 `kubectl apply -f nginx-pod.yaml` 等操作

[K8S Object](https://kubernetes.io/docs/concepts/overview/working-with-objects/kubernetes-objects/#describing-a-kubernetes-object) 

- 基本上所有的 K8S 资源都可以通过 YAML 文件来进行创建与配置。

[POD](https://kubernetes.io/docs/concepts/workloads/pods/#working-with-pods)

- POD 是 K8S 的最小管理单位。一个 POD 可以看成是由多个容器封装而成的应用。

[Label](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/)

- Label 能够在资源上添加标识，用来对它们进行区分和选择。因此在部署应用时，我们就无需关注 POD 或者其他服务具体被运行在哪个区域的哪台机子上了。

[NAMESPACE](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/) 

- Namespace 主要用于资源、环境隔离。可以视为为资源分组；默认情况下，kubernetes 集群中的所有的 Pod 都是可以相互访问的。但是在实际中，可能不想让两个 Pod 之间进行互相的访问，那此时就可以将两个 Pod 划分到不同的 namespace 下。kubernetes 通过将集群内部的资源分配到不同的 Namespace 中，可以形成逻辑上的"组"，以方便不同的组的资源进行隔离使用和管理。

PODController 或 [workload resource](https://kubernetes.io/docs/concepts/workloads/controllers/) 

- Pod 控制器用于 pod 的管理，确保 pod 资源符合预期的状态，当 pod 的资源出现故障时，会尝试进行重启或重建 pod。

[Service](https://kubernetes.io/docs/concepts/services-networking/service/)

- 主要用于提供对外访问服务，通过 Service 能够轻松实现负载均衡、服务发现等功能。

[storage](https://kubernetes.io/docs/concepts/storage/)

### K8S 安装

安装 K8S 有许多[方法](https://kubernetes.io/docs/tasks/tools/)，目前绝大多数云服务平台都有提供 K8S 集群搭建服务，只需要输入一些你想要的配置信息既可以一步搭建 K8S 集群，就是要收点云平台费用；想要免费的话，可以参考 [网友整](https://gitee.com/yooome/golang)理笔记 、[bilibili 黑马程序](https://www.bilibili.com/video/BV1Qv41167ck/?spm_id_from=333.337.search-card.all.click)员教程 通过  **kubeadm**  手动搭建，或者使用 kuboard 等操作工具搭建。
以下会通过 Azure 平台的 AKS 服务进行 K8S 集群服务搭建。

### 在 K8S 上搭建应用

最基础的建立 K8S 应用方法，是为各个 K8S 组件编写对应的 YAML 配置文件，然后一一启动。HELM 提供了便捷的 YAML 模板，并通过 HELM chart 对模板进行统一管理与包装。

关于 HELM，他是一个 K8S 集群应用管理工具，你可以把他类比成 python 的 pip，helm 通常通过 values.yaml 文件来控制你想要安装的集群应用，包括版本、参数等信息，好比 pip 中的 requirments.txt 文件；如果你想详细了解 Helm，可以参考 [Helm 官网](https://helm.sh/) 。

### 搭建 JupyterHub K8S 操作说明

[JupyterHub K8S 官网](https://z2jh.jupyter.org/en/stable/jupyterhub/customizing/user-environment.html) 有详细的搭建过程，但资料过于零散，以下列举一些关键步骤。

1. 在 Azure K8s 服务上创建一个 K8S 集群服务，由于 Azure 官方配置不断更新，建议参考 K8S[ 官网文档](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-portal?tabs=azure-cli) 完成该步骤。

2. 编写 config.yaml 文件，即 Helm 需要的 values.yaml 文件，该文档所在文件夹下，我们准备好了一份  config.yaml 文件，大家可以直接使用他即可，也可以根据 [JupyterHub K8S 官网](https://z2jh.jupyter.org/en/stable/jupyterhub/customizing/user-environment.html) 自定义内容。配置文件注意两点：

   1. 配置用户端使用的镜像：我们基于 `jupyter/minimal-notebook` 镜像，构建了 `kevinng77/coe_jupyter` 镜像，而后上传到 dockerhub 上。并在 `/tmp/` 中放入每个用户都会使用到的教学文件。当新用户注册时，k8s 会创建 `kevinng77/coe_jupyter` 容器，而后触发钩子，自动将文件转移到用户工作路径下。

      ```yaml
      singleuser:
        image:
          name: kevinng77/coe_jupyter
          tag: latest
        cmd: null
        lifecycleHooks:
          postStart:
            exec:
              command:
                - "sh"
                - "-c"
                - >
                  rm -r /home/jovyan/*;
                  cp -r /tmp/* /home/jovyan/
      ```

   2. 配置 JupyterHUB 用户名，请参考官网说明。

      ```yaml
      hub:
        config:
          Authenticator:
            admin_users:
            - kevinng
            allowed_users:  # 不配置的话，允许用户注册
            - user01
            - user02
          DummyAuthenticator:
            password: userpasswd                      #配置通用密码（基本没什么安全性）
          JupyterHub:
            admin_access: true                      #配置是否允许管理者账户存在
            authenticator_class: dummy              #指定所有账户授权类型（默认是傻瓜式）
      ```

::: detail 完整 config.yaml 文件

```yaml
# 这个文件仅用于通过 HELM 配置 JupyterHub K8S 集群
singleuser:
  image:
    # 配置你需要使用的 docker 镜像
    # 详细可以参考：https://z2jh.jupyter.org/en/stable/jupyterhub/customizing/user-environment.html
    # 配置用户端使用的镜像，我们基于 `jupyter/minimal-notebook` 镜像，构建了 `kevinng77/coe_jupyter` 镜像，
    # 包括 安装了 paddle 等 python 依赖等，而后上传镜像到 dockerhub 上。
    name: kevinng77/coe_jupyter  
    tag: latest
    # 部署应用时相当于运行 docker pull kevinng77/coe_jupyter:latest

  # `cmd: null` allows the custom CMD of the Jupyter docker-stacks to be used
  # which performs further customization on startup.  

  cmd: null
  lifecycleHooks:
    # 当新用户注册时，k8s 会创建 kevinng77/coe_jupyter 容器，而后触发钩子，自动执行以下 linux 命令，将文件转移到用户工作路径下。
    postStart:
      exec:
        command:
          - "sh"
          - "-c"
          - >
            rm -r /home/jovyan/*;
            cp -r /tmp/* /home/jovyan/
  # 每个用户分配 1G 的单独储存空间
  storage:
    capacity: 1Gi
    
hub:
  config:
    Authenticator:
      admin_users:                          # jupyterhub 的管理员用户
      - kevinng
      allowed_users:                        # 不配置的话，允许用户注册
      - xxx01
      - xxx02
      - xx19
    DummyAuthenticator:
      password: xx                      #配置通用密码（基本没什么安全性）
    JupyterHub:
      admin_access: true                      #配置是否允许管理者账户存在
      authenticator_class: dummy              #指定所有账户授权类型（默认是傻瓜式）

proxy:
  # 这个 secretToken 似乎不需要
  # secretToken: "1928cb5e4d54464c94bc2874d63ebc3c3e59b6d2c3b35d8eb4ef603d872a5b0a"
  service:
    type: LoadBalancer
    annotations: 
      service.beta.kubernetes.io/azure-dns-label-name: kevinng   # 注意：修改成你想养的 DNS 名称
      # 如果上面设置的是 kevinng，那么在浏览器中访问: http://kevinng.eastus.cloudapp.azure.com/  (仅对 AKS 生效)
```

:::

1. 打开你的 K8S resource，界面如下：![image-20230903111634269](https://pic2.zhimg.com/80/v2-2dd69265b0b0c99d9f8e5c0367067095_1440w.webp)

2. 在你的 K8S resource 中点击 connect -> open cloud shell，会出现下面的操作终端。

   ![image-20230903111654054](/assets/img/k8s_hub/image-20230903111654054.png)

3. 点击上传文件按钮，上传 config.yaml 文件到终端上。然后执行通过 helm 部署 k8s jupyterhub 服务。首先添加仓库源（仓库源类似 pip 安装时候的仓库源）：

   ```shell
   helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/
   helm repo update
   ```

   ![image-20230903111730520](/assets/img/k8s_hub/image-20230903111730520.png)

4. 然后 helm 一键部署 jupyterhub 应用，在你的终端输入：

   ```shell
   helm upgrade --cleanup-on-fail \ 
   	--install cv jupyterhub/jupyterhub \ 
   	--namespace coe \ 
   	--create-namespace \ 
   	--values config.yaml
   ```

   执行后开始构建集群，构建成功后等待几分钟，会有这种提示：

   ![image-20230903111749193](/assets/img/k8s_hub/image-20230903111749193.png)

5. 访问[ http://kevinng.eastasia.cloudapp.azure.com/](http://kevinng.eastasia.cloudapp.azure.com/) 即可登录网站，注意需要将 eastus 换成你的 K8S 服务器所在地址。免费的 Azure 服务只提供这种类型的域名： `your_dns_label.<location>.cloudapp.azure.com`；部分 Azure 虚拟机，或者 Azure 服务都可以免费设置 DNS label，在 Azure K8S 的 jupyterhub 服务中，你可以提供 service 对应的 annotations 规则，来修改你的 dns label，如：

   ```yaml
   proxy:
     secretToken: "1928cb5e4d54464c94bc2874d63ebc3c3e59b6d2c3b35d8eb4ef603d872a5b0a"
     service:
       type: LoadBalancer
       annotations: 
         service.beta.kubernetes.io/azure-dns-label-name: aclearning  
         # DN: http://kevinng.eastus.cloudapp.azure.com/  (仅对 AKS 生效)
   ```

6. 修改后，登录网站的 URL 就变成了：`aclearning.<location>.cloudapp.azure.com`登录后你可以看到 jupyterlab 的登陆界面。

   其中登录用户和密码在 config 中配置，如下登录管理员账号为 kevinng，所有用户统一密码为 xxx

   ```yaml
   hub:
     config:
       Authenticator:
         admin_users:
         - kevinng
         # allowed_users:                      # 不配置的话，允许用户注册
         # - xxx01
         # - xxx02
       DummyAuthenticator:
         password: xxx                     #配置通用密码（基本没什么安全性）
       JupyterHub:
         admin_access: true                      #配置是否允许管理者账户存在
         authenticator_class: dummy   
   
   ```

   

7. 自定义 notebook 的文件和 python 环境 K8S 集群式是通过管理 docker container 来实现自动化的规模控制的，因此你需要将自定义的文件和 python 环境打包成 docker image，而后在 K8S 中使用。具体请参考[JupyterHub K8S 官网](https://z2jh.jupyter.org/en/stable/jupyterhub/customizing/user-environment.html) 

### K8S 集群监控

以下对 COE 举办过程中，K8S 集群的部分运行指标进行展示。

#### 弹性伸缩

观察节点数量（Number Nodes），[Nodes](https://kubernetes.io/docs/concepts/architecture/nodes/) 为集群中计算机节点数量，本次实验中配置了自动服务规划（[HPA](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)），K8S 会根据应用的负载程度，根据 CPU 或内存使用率，自动对集群节点进行规划。如下，我们于下午 1 点开启所有 server 服务。在 1 点、5-6 点之间，出现了负载高峰期，节点数量自动调整为 5 个。在 COE Lab 结束后，节点数量自动调整为 1 个。

![image-20230903111944195](/assets/img/k8s_hub/image-20230903111944195.png)

#### 负载均衡

本次搭建的 K8S 集群采用了 [ **LoadBalancer** ](https://kubernetes.io/docs/concepts/services-networking/service/#loadbalancer) 进行轮询服务转发。观察集群 CPU、内存使用率以及每个集群上的用户数量。我们采用 4 核 16G 内存的计算机作为节点，每个节点平均会分配约 7-8 个用户。COE lab 中存在大小为 32 MB 的 CV DeepLab 模型需要预测。在整个 COE 过程中， CPU 以及内存使用率都很低，可以说使用起来搓搓有余。

![image-20230903112021050](/assets/img/k8s_hub/image-20230903112021050.png)

当然 K8S 支持其他更多的服务转发，可以参考[ ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)版本回退
在 K8S 中，只需要在 YAML 中更新好想要的版本，然后重新应用以下即可。如修改 HELM config.yaml 中的配置内容，我们修改 JUPYTERHUB 的登录密码，更新后在命令行重新应用以下配置文件即可。因此每一套 YAML 文件，可以以视为一个 K8S 应用版本。

```shell
helm upgrade --cleanup-on-fail \ --install cv jupyterhub/jupyterhub \ --namespace coe \ --create-namespace \ --values config.yaml
```

#### 版本回退

在 K8S 中，只需要在 YAML 中更新好想要的版本，然后重新应用以下即可。如修改 HELM config.yaml 中的配置内容，我们修改 JUPYTERHUB 的登录密码，更新后在命令行重新应用以下配置文件即可。因此每一套 YAML 文件，可以以视为一个 K8S 应用版本。

```shell
helm upgrade --cleanup-on-fail \
 --install cv jupyterhub/jupyterhub \
 --namespace coe \
 --create-namespace \
 --values config.yaml   # 指定版本的 config 文件
```

#### 储存编排

应用会根据具体的需求，申请/创建储存卷。我们观察 AKS 中的 PVC 数量。本次实验我们为每一个用户配置了 1GB 的储存空间。当 JupyterHub 用户注册并开启服务器是，我们能够看到 PVC 的数量与对应 POD 的数量同时增长，每个用户对应一个 POD，每个 POD 拥有一个 PVC 以储存数据。

## 更多

[更多参考官网](https://kubernetes.io/docs/home/) 