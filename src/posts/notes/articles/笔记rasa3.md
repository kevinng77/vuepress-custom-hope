---
title: RASA 回忆录（三） - RASA K8S 部署
date: 2022-09-25
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
---

本文围绕 RASA + K8S 部署方案展开，对 RASA 集群性能等进行分享。

RASA 采用 sanic 搭建 API 服务，NLU 和 Dialog 模块分别由两个 API 服务组成（Kore.ai, Nuance 等平台也是采用该方案）。其中 NLU 模块涉及到 Transformer 模型的推理，属于 CPU Bound。Dialog 中大部分需要基于 NLU 模块进行逻辑推理，属于 I/O Bound。

因此实际部署过程中，我们可能会希望采用 1 个 RASA ACTION 进程，配合多个 RASA NLU 模块来进行服务。

## RASA K8S

https://github.com/kevinng77/rasa_example/tree/master/examples/9_k8s_rasa

基于示例目的，我们采用 helm 来快速构建 RASA 集群。

### 1. 创建 K8S 环境

```
kubectl create namespace rasa
```

需要添加对应的 helm chart 源

```
helm repo add rasa https://helm.rasa.com
```

### 2. 制作对应的 Docker Image

将 actionss 文件放在 `actions/actions.py` 中，执行：

```bash
docker build ./action_docker -t <account_username>/<repository_name>:<custom_image_tag>
```

保存后，上传到 docker 的某个 registry。本案例使用 dockerhub

```
docker login
docker push kevinng77/myaction
```

### 3. 配置 Action helm 文件

主要修改 `action_values.yml` 其中的 image 路径:

```yaml
image:
  # -- Action Server image name to use (relative to `registry`)
  name: kevinng77/myaction

  # -- Action Server image tag to use
  tag: "latest"
```

把 `kevinng77/myaction` 改成你 dockerhub 对应的镜像 ID 就行。

其他配置可以参考 [rasa chart](https://github.com/RasaHQ/helm-charts/tree/main/charts) ，其中你可能会考虑：

- 调整 auto scaling 方案。
- Service 方案采用 ClusterIP，因为我们会将 RASA server 和 Action server 部署在同一个集群中。如果要分开部署，可以设置其他服务方式。

### 4. 安装 Action server service

参考官方提供的 helm，一键安装即可，我们将 action server 部署 release_name 为 `rasa-action-server` ：

```
helm install --namespace rasa \
  --values action_values.yml rasa-action-server rasa/rasa-action-server
```

更新的话使用：

```
helm upgrade --namespace rasa  --reuse-values  \
  --values action_values.yml rasa-action-server rasa/rasa-action-server
```

### 5. 制作 RASA Image

默认的 RASA image 当中，时没有 spacy 等包的，如果你的 rasa 架构使用了 torch，paddle，spacy 等依赖，可以自行打包：

```dockerfile
# rasa_docker/Dockerfile
FROM rasa/rasa:3.4.4
WORKDIR /app
USER root

RUN pip install spacy
RUN python -m spacy download zh_core_web_trf

# By best practices, don't run the code with root user
USER 1001
```

构建镜像并推送到 dockerhub：

```
docker build ./rasa_docker -t kevinng77/rasa:3.4.4
docker push kevinng77/rasa:3.4.4
```

将本地上用 `rasa train` 训练出来的模型推到 github 上（也可以是其他你可以通过 wget 下载到的地方），比如该案例中，将模型推到了： `https://github.com/kevinng77/rasa_model/zh_model.tar.gz`

### 6. 配置 RASA helm 文件

修改 rasa_values.yml, 完整文件可以参考 rasa_values.yml 文件。比较值得注意的是：

- rasa server 和 action server 的通信，通过 helm 配置方式为：

```bash
## Settings for Rasa Action Server
## See: https://github.com/RasaHQ/helm-charts/tree/main/charts/rasa-action-server
rasa-action-server:
  # -- Install Rasa Action Server
  install: false

  external:
    # -- Determine if external URL is used
    enabled: true
    # -- External URL to Rasa Action Server
    url: "http://rasa-action-server/webhook"
```

其中 URL 用的 `http://rasa-action-server/webhook` 表示 action server 在同 K8S 集群上的 resource name: `rasa-action-server` 运行。因此通过 ClusterIP 的方式就能访问到。

- 我们设置让 rasa server（同名的 pod） 不要分布在同一个 label 中，设置 pod label 为 `app: rasa-server`，而后配置：

```
podLabels:
  app: rasa-server
affinity: 
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchExpressions:
        - key: app
          operator: In
          values:
          - rasa-server
      topologyKey: "kubernetes.io/hostname"
```

其中的 topologyKey 可以通过 `kubectl get node --show-labels` 查看。

1. 配置模型路径和 credentials，支持 restAPI 以及 socketio 通信。

```
applicationSettings:
	# 该路径应该是一个下载路径，对应 https://github.com/kevinng77/rasa_model/zh_model.tar.gz
	# 上的内容
  initialModel: "https://github.com/kevinng77/rasa_model/blob/master/zh_model.tar.gz?raw=true"
  credentials:
    # 
    enabled: true
    additionalChannelCredentials:
      rest: {}
      socketio:
        user_message_evt: user_uttered
        bot_message_evt: bot_uttered
        session_persistence: true
        # 其它 credentials 配置
```

1. 修改 Image source 为你推送到 dockerhub 的镜像

```
image:
  name: rasa
  tag: "3.4.4"
  # -- Override default registry + image.name for Rasa Open Source
  repository: "kevinng77/rasa"
```

1. 针对 AZURE 进行特殊服务配置

```
service:
  type: LoadBalancer
  port: 5005
  # Azure 专用 LoadBalancer 申请域名方法
  annotations: {
    service.beta.kubernetes.io/azure-dns-label-name: acrasa
  }
```

### 7. 启动 RASA 服务

```
helm install \
    --namespace rasa \
    --values rasa_values.yml \
    myrasa \
    rasa/rasa
```

更新 helm 用

```
helm upgrade -f rasa_values.yml --reuse-values  \
    --namespace rasa \
        myrasa rasa/rasa
```

### 8. 访问 RASA

可以通过 LoadBalancer 对应的 IP 地址进行方案。其中我们基于 Azure 配置，可以直接访问 IP:

```
http://acrasa.eastasia.cloudapp.azure.com/
```

前端通过 restAPI 发送请求，或者通过 RASA 提供的 chat Widget，具体查看 RASA 官网:

```
<div id="rasa-chat-widget" data-websocket-url="http://your_ip:5005/socket.io"></div>
<script src="https://unpkg.com/@rasahq/rasa-chat" type="application/javascript"></script>
```

通过 gevent 和 requests 在 python 上模拟了一下高峰访问：

node 配置：2 核 8G 内存。NLU 模型：SPACE ZH(400M)

### 9. 性能测试

列名（1,10,50,100）表示 1 秒内，连续给 RASA 服务器发送请求的数量。1/2/4/10 node，表示 K8S 集群的 node 数量。数据表示每个请求从发送，到接受到第一次回复的耗时范围。 

| 1 秒内访问数量/耗时（秒） | 1       | 10      | 50        | 100     |
| ------------------------ | ------- | ------- | --------- | ------- |
| 1 node                   | 0.5-0.7 | 2.5-3.4 | 12.5-14.2 | 27-29   |
| 2 node                   | 0.5-0.7 | 0.5-1.4 | 3.9-6     | 8-11.2  |
| 4 node                   | 0.5-0.7 | 0.5-0.8 | 1.1-3.5   | 2.3-5.8 |
| 10 node                  | 0.5-0.7 | 0.5-0.7 | 0.5-3     | 0.5-3   |

单台机器配置不能太低，否则轮询策略对耗时影响大。建议 4 核 16+GB 内存节点。NLU 部分对模型进行推理优化后，2 - 3 台 4 核 16+GB 内存节点就能应付好每秒钟 100 次的请求了。