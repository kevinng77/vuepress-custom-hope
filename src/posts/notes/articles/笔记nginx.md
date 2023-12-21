---
title: Nginx 简陋资源笔记
date: 2022-08-27
author: Kevin 吴嘉文
category:
- 知识笔记
---

## Nginx 目的

在客户端和服务器之间加入管理层，实现用户访问同一地址，但能在不同服务器地址上操作。session 共享的话还是需要使用 redis 等来实现。

## Nginx

官方数据测试表明能够支持高达 50,000 个并发连接数的响应。平时一个 Nginx 可能就能满足小规模的业务需求。

Nginx 可以用来做很多事情：

- Http 代理，反向代理
- 负载均衡：Nginx 提供的负载均衡策略有 2 种：内置策略和扩展策略。内置策略为轮询，加权轮询，Ip hash。扩展策略，就天马行空，只有你想不到的没有他做不到的啦，你可以参照所有的负载均衡算法，给他一一找出来做下实现。

[![img](https://camo.githubusercontent.com/d13d6330de88bf9f549b8086be9a91cea00a0fba6093cb690afcb4b49b8834d0/68747470733a2f2f7777772e72756e6f6f622e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031382f30382f313533353732353037382d383330332d32303136303230323133333735333338322d313836333635373234322e6a7067)](https://camo.githubusercontent.com/d13d6330de88bf9f549b8086be9a91cea00a0fba6093cb690afcb4b49b8834d0/68747470733a2f2f7777772e72756e6f6f622e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031382f30382f313533353732353037382d383330332d32303136303230323133333735333338322d313836333635373234322e6a7067)

- https 证书：nginx 配置更简单（[SSL 证书配置](https://www.liaoxuefeng.com/article/990311924891552)）；
- 一组 host：nginx 配置更简单；
- 限流：nginx 配置更简单；
- 限 ip：nginx 配置更简单；
- 静态文件：nginx 可缓存；
- http2：nginx 支持，内部转 http1.x 到 tomcat；
- http3：nginx 支持，内部转 http1.x 到 tomcat；
- 临时重定向 url：nginx 改配置 reload 不重启；
- 遇到 500 错误：nginx 可重试；
- 很多 cors、自定义 header 配置、[http://www.example.com 转 http://example.com 放 nginx 不用改 java 应用。](http://www.example.xn--comhttp-c72v//example.com 放 nginx 不用改 java 应用。)

核心思想是利用 nginx 强大的配置能力，避免改配置反复部署 ja 应用。

## Nginx 的安装

可以直接用 docker 进行配置：

```shell
docker run --rm -d --name nginx \
    --net host \
    -v ./log:/var/log/nginx \
    -v ./nginx.conf:/etc/nginx/nginx.conf:ro \
    nginx
```

## Nginx 配置示例

```nginx
########### 每个指令必须有分号结束。#################
#user administrator administrators;  #配置用户或者组，默认为 nobody nobody。
#worker_processes 2;  #允许生成的进程数，默认为 1
#pid /nginx/pid/nginx.pid;   #指定 nginx 进程运行文件存放地址
error_log log/error.log debug;  #制定日志路径，级别。这个设置可以放入全局块，http 块，server 块，级别以此为：debug|info|notice|warn|error|crit|alert|emerg
events {
    accept_mutex on;   #设置网路连接序列化，防止惊群现象发生，默认为 on
    multi_accept on;  #设置一个进程是否同时接受多个网络连接，默认为 off
    #use epoll;      #事件驱动模型，select|poll|kqueue|epoll|resig|/dev/poll|eventport
    worker_connections  1024;    #最大连接数，默认为 512
}
http {
    include       mime.types;   #文件扩展名与文件类型映射表
    default_type  application/octet-stream; #默认文件类型，默认为 text/plain
    #access_log off; #取消服务日志    
    log_format myFormat '$remote_addr–$remote_user [$time_local] $request $status $body_bytes_sent $http_referer $http_user_agent $http_x_forwarded_for'; #自定义格式
    access_log log/access.log myFormat;  #combined 为日志格式的默认值
    sendfile on;   #允许 sendfile 方式传输文件，默认为 off，可以在 http 块，server 块，location 块。
    sendfile_max_chunk 100k;  #每个进程每次调用传输数量不能大于设定的值，默认为 0，即不设上限。
    keepalive_timeout 65;  #连接超时时间，默认为 75s，可以在 http，server，location 块。

    upstream mysvr {   
      server 127.0.0.1:7878;
      server 192.168.10.121:3333 backup;  #热备
    }
    error_page 404 https://www.baidu.com; #错误页
    server {
        keepalive_requests 120; #单连接请求上限次数。
        listen       4545;   #监听端口
        server_name  127.0.0.1;   #监听地址       
        location  ~*^.+$ {       #请求的 url 过滤，正则匹配，~为区分大小写，~*为不区分大小写。
           #root path;  #根目录
           #index vv.txt;  #设置默认页
           proxy_pass  http://mysvr;  #请求转向 mysvr 定义的服务器列表
           deny 127.0.0.1;  #拒绝的 ip
           allow 172.18.5.54; #允许的 ip           
        } 
    }
}
```

::: details 示例配置 2

```nginx
# 全局参数
user nginx;              # Nginx 进程运行用户
worker_processes auto;   # Nginx 工作进程数，通常设置为 CPU 核数
error_log /var/log/nginx/error.log warn;    # 错误日志路径和日志级别
pid /run/nginx.pid;      # 进程 PID 保存路径

# 定义事件模块
events {
    worker_connections 1024;    # 每个工作进程最大并发连接数
    use epoll;                  # 使用 epoll 网络模型，提高性能
    multi_accept on;            # 开启支持多个连接同时建立
}

# 定义 HTTP 服务器模块
http {
    # 缓存文件目录
    client_body_temp_path /var/cache/nginx/client_temp;
    proxy_temp_path /var/cache/nginx/proxy_temp;
    fastcgi_temp_path /var/cache/nginx/fastcgi_temp;

    # 定义日志格式，main 是默认的日志格式
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
        '$status $body_bytes_sent "$http_referer" '
        '"$http_user_agent" "$http_x_forwarded_for"';

    # 默认访问日志保存路径和格式
    access_log /var/log/nginx/access.log main;

    # 定义 MIME 类型
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # 代理参数
    proxy_connect_timeout 6s;       # 连接超时时间
    proxy_send_timeout 10s;         # 发送超时时间
    proxy_read_timeout 10s;         # 接收超时时间
    proxy_buffer_size 16k;          # 缓冲区大小
    proxy_buffers 4 32k;            # 缓冲区个数和大小
    proxy_busy_buffers_size 64k;    # 忙碌缓冲区大小
    proxy_temp_file_write_size 64k; # 代理临时文件写入大小

    # 启用压缩，可以提高网站访问速度
    gzip on;
    gzip_min_length 1k;                    # 最小压缩文件大小
    gzip_types text/plain text/css application/json application/javascript application/xml;

    # 定义 HTTP 服务器
    server {
        listen 80;              # 监听端口

        server_name example.com;    # 域名

        # 重定向到 HTTPS，强制使用 HTTPS 访问
        if ($scheme != "https") {
            return 301 https://$server_name$request_uri;
        }

        # HTTPS 服务器配置
        ssl_certificate      /etc/nginx/ssl/server.crt;    # SSL 证书路径
        ssl_certificate_key  /etc/nginx/ssl/server.key;    # SSL 私钥路径

        # SSL 会话缓存参数
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;
        ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
        ssl_prefer_server_ciphers on;
        ssl_ciphers ECDH+AESGCM:ECDH+AES256:ECDH+AES128:DH+3DES:!ADH:!AECDH:!MD5;

        # 配置代理路径
        location / {
            proxy_pass http://localhost:8080;        # 转发请求的目标地址
            proxy_set_header Host $host;             # 设置请求头中的 Host 字段
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                            # 设置 HTTP 头中的 X-Forwarded-For 字段，表示客户端真实 IP，多个 IP 用逗号隔开
            proxy_set_header X-Real-IP $remote_addr; # 设置请求头中的 X-Real-IP 字段，表示客户端真实 IP
        }

        # 配置静态文件访问路径
        location /static/ {
            alias /path/to/static/files/;   # 静态文件的目录
            expires 7d;                     # 静态文件缓存时间
            add_header Pragma public;       # 添加 HTTP 响应头
            add_header Cache-Control "public, must-revalidate, proxy-revalidate";
        }

        # 配置错误页面
        error_page 404 /404.html;           # 404 错误页
        location = /404.html {
            internal;                       # 不接受外部访问
            root /usr/share/nginx/html;     # 404 错误页文件所在目录
        }

        # 配置重定向
        location /old/ {
            rewrite ^/old/([^/]+) /new/$1 permanent;   # 将/old/xxx 路径重定向为/new/xxx，返回 301 状态码
        }
    }

    # 其他服务配置
    # server {
    #     ...
    # }

    # 配置 TCP 负载均衡
    upstream backends {
        server backend1.example.com:8080 weight=5;  # 后端服务器地址和权重
        server backend2.example.com:8080;
        server backend3.example.com:8080 backup;   # 备用服务器
        keepalive 16;                               # 连接池大小
    }

    server {
        listen 80;
        server_name example.com;

        location / {
            proxy_pass http://backends;             # 负载均衡转发请求的目标地址
            proxy_set_header Host $host;            # 设置请求头中的 Host 字段
            proxy_set_header X-Real-IP $remote_addr; # 设置请求头中的 X-Real-IP 字段，表示客户端真实 IP
        }
    }
}
```



:::

### Nginx HTTPS 配置

### Let's Encrypt 免费 SSL

Let's Encrypt 是一个由权威机构设置的计划，为网站用户提供免费的 SSL 证书。Let's Encrypt 证书是由非营利组织 Electronic Frontier Foundation（EFF）以及 Mozilla、Akamai、Cisco、IETF 等企业和组织提供支持，同时它也依托了 Linux 社区的合作。用户可以使用任何采用 Let's Encrypt 证书的网站而无需支付任何费用。Let's Encrypt 发行的数字证书有“DV SSL”和“OV SSL”两种类型。

如果可以通过命令行方式，以 root 权限进入你的服务器时，推荐使用功能 [certbot](https://certbot.eff.org/instructions?ws=nginx&os=centosrhel8)。根据官方指南操作即可。

或者可以使用  https://github.com/kevinng77/nginx-certbot/tree/master 一键部署。

配置 `app.conf` 文件：

```nginx
server {
    listen 80;
    server_name wujiawen.xyz;
    server_tokens off;

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name wujiawen.xyz;
    server_tokens off;

    ssl_certificate /etc/letsencrypt/live/wujiawen.xyz/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/wujiawen.xyz/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    location / {
        root /home/blog;
        index  index.html index.htm;
    }
}
```

然后启动服务：

```bash
docker compose up -d
```

[参考文章](https://pentacent.medium.com/nginx-and-lets-encrypt-with-docker-in-less-than-5-minutes-b4b8a60d3a71)

### 其他免费 ssl 证书

Cloudflare

Cloudflare 是另一个可提供免费 SSL 证书的公司，他们是一家全球性的提供缓存和 DNS 技术的公司。Cloudflare 提供的 SSL 证书包括“Flexible SSL”、“Full SSL”、“Full SSL Strict”三种类型。CloudFlare 还提供“Always Use HTTPS”选项，将网站访问升级到 HTTPS。

SSLForFree

SSLForFree 是一个提供免费 SSL 证书的网站。用户可以在这里获取有效的数字证书。“SSLForFree”支持 RapidSSL, GeoTrust 和 Comodo 域名验证证书，用户只需要验证域名所有权，即可获得 SSL 证书的使用权。



## 资料

 [nginx 中文指南](https://www.nginx-cn.net/) 