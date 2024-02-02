---
title: FRP 简易部署内网穿透
date: 2022-05-07
author: Kevin 吴嘉文
keywords: 
category:
- Hobbies 业余爱好
tag:
- Linux
---

## FRP 简易部署内网穿透

部署 FRP，需要在一台拥有公网 IP 的服务器上安装 FRP server，然后在本地需要进行内网穿透的机器上安装 FRP client。安装成功后，能够实现在访问 FRP server 某个端口时候，转接到 FRP client 对应的端口上。

### 服务器端

在拥有公网 IP 的服务器上操作

```shell
vim ~/frp/frps.ini
```

而后在  `frps.ini` 中写入：

```ini
[common]
bind_port = 7000
dashboard_port = 7500
dashboard_user = admin
dashboard_pwd = dashbord_passwd

vhost_http_port = 7080
vhost_https_port = 7081

token = your_passwd
```

而后在服务器上安装 frp docker 即可：

```shell
docker run --restart=always --network host -d -v ~/frp/frps.ini:/etc/frp/frps.ini snowdreamtech/frps:0.48.0
```

### 客户端操作

创建 `frpc.ini`

```shell
vim /etc/frp/frpc.ini
```

写入：

```ini
[common]
server_addr = your_srp_server_ip
server_port = 7000
token = your_passwd

[ssh]
type = tcp
local_ip = 127.0.0.1
local_port = 22
remote_port = 2222

[ssh]
type = tcp
local_ip = 192.168.1.229
local_port = 80
remote_port = 18022
```

如上配置，访问 FRP server 公网 ip:2222 时，会被跳转到 `127.0.0.1:22` 端口上，访问 `18022` 端口则跳转到路由器网络内的另一台机子上 `192.168.1.229:80`

可以配置映射的 IP 范围：

```ini
[range:FTP]
type = tcp
local_port = 5000-5050
remote_port = 5000-5050
local_ip = 127.0.0.1
```

同样的启动 frp client：

```shell
docker run --restart=always --net=host -d -v /etc/frp/:/etc/frp/ snowdreamtech/frpc:0.48.0 
```

### 监控界面

通过 `公网 ip:7500` 可以看到 frp webui