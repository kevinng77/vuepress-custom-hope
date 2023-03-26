---
title: HEXO 搭建与相关问题解决方案
date: 2020-10-22
author: Kevin 吴嘉文
keywords: 
language: cn
category:
- Project|项目
tag:
- Hexo
mathjax: true
toc: true
comments: 博客总结
---

# 博客指南

> 本文整理了 再 Linux 下阿里云服务器搭建 Hexo 个人博客的方案。同时也总结了一些再 Hexo 搭建 yilia 主题时存在问题的解决方案，如 Latex 显示错误，主题元素自定义，文章中视频大小自适应，评论系统等。

<!--more-->

## 动态博客

动态博客可以尝试 docker + wordpress，这个博客方案提供文章管理平台和用户管理平台。个人觉得，相对 Hexo 来说操作起来更容易。你只需要在网页端就可以实现编辑发布文章，处理网站数据等。教程可以直接看阿里云上的课程。

## 静态博客 HEXO

### 云服务器部署

Hexo 提供将博客免费部署到 github.io 上。但这样存在两个问题，一是访问 404 问题，二是网址不够优雅。因此在国内搭建服务器，然后申请一个域名进行解析是比较推荐的一种方案。当然，除了自己搭建服务器，也可以考虑云服务器这种方便快捷的服务。

+ 购买阿里云 ECS ，我买了一年 89，双 11 新人价。

#### 设置实例密码与启动

购买云服务器后进行基本的配置

+ 设置左上角定位信息（华南 1）

+ 进入实例，选择实例，重置实例密码，重启实例

![image-20201108110300611](/assets/img/blog/image-20201108110300611.png)

#### 设置安全组和安全规则

虚拟防火墙

+ 网络与安全-安全组-点击实例名进入实例管理

+ 入方向：

![image-20201108111354302](/assets/img/blog/image-20201108111354302.png)

+ [官方新手教程](https://help.aliyun.com/learn/getting-started.html?spm=5176.19720258.J_2937333540.5.61e82c4a7KiJ9O) 

如果你对安全要求比较高，建议额外设置防火墙，官方也有提供更高级的防火墙服务。

#### 登录服务器

打开系统自带的终端工具。

- Windows：CMD 或 Powershell。
- MAC：Terminal。

Windows 用户请检查系统中是否安装有 ssh 工具。检查方法：

+ 在终端中输入命令 ssh -V。

```
ssh -V
```

+ 出现如下结果说明已安装。否则请下载安装[OpenSSH](https://www.mls-software.com/files/setupssh-8.2p1-1.exe)。

![查看 ssh 版本](http://static-aliyun-doc.oss-cn-hangzhou.aliyuncs.com/assets/assets/img/zh-CN/2223009851/p102980.png)

在终端中输入连接命令 ssh [username]@[ipaddress]。将其中的 username 和 ipaddress 替换为第 1 小节中创建的 ECS 服务器的登录名和公网地址。例如：

```
ssh root@47.115.75.179
```

初次外，也可以通过阿里云服务器上提供的远程操控来访问服务器

### Hexo 配置到阿里云服务器

这部分内容主要参考[ObjectSpace](https://blog.csdn.net/NoCortY)发布的云服务器搭建教程 [链接在这](https://blog.csdn.net/NoCortY/article/details/99631249?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.add_param_isCf&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.add_param_isCf)，针对其中一些问题也做出了修改。

登录服务器后，我们在服务器上操作。

#### NGINX 安装与配置


```bash
yum install -y nginx

systemctl start nginx
systemctl enable nginx
```

![image-20201108121336029](/assets/img/blog/image-20201108121336029.png)

启动 nginx 后，在浏览器输入公网 IP 可以登录到 nginx 页面，表示操作成功

![image-20201108121542784](/assets/img/blog/image-20201108121542784.png)

原作者的方案是新建一个 vhost 文件夹，然后将这个文件夹 include 到 nginx.conf 中去。操作如下

 ```
  cd /etc/nginx
  mkdir vhost
  cd vhost
  vim blog.conf
 ```

编辑 blog.conf 文件内容

设置有 www 到无 www 的跳转

```
server{
	listen    80;
	root /home/www/website;(服务器博客目录存放地址，本文将博客的存放地址设置为/home/www/website )
	server_name 这里填域名如(www.baidu.com) 如果暂时没有域名就填阿里云的公网 ip，以后有了再改回来;
	location /{
	}
}

server {
        listen *:80;
        listen [::]:80;
        server_name www.example.com;
        return 301 http://example.com$request_uri;
}
```

在 nginx.conf 中 include vhost
`vi /etc/nginx/nginx.conf`

找到 `include /etc/nginx/conf.d/*.conf` 这行代码，

在下面添加 `include /etc/nginx/vhost/*.conf`

创建服务器博客目录 /home/www/website

```
cd /home
mkdir www
cd www
mkdir website
```

设置 website 权限

`cd /home`

`chmod -R 777 ./www`

#### nodejs 与 git 安装

 **安装 nodejs** 

`curl -sL https://rpm.nodesource.com/setup_14.x | bash - `

`yum install -y nodejs`

 **安装 git:** 
`yum install git`
配置 git 用户
`adduser git`
修改用户权限:

```
chmod 740 /etc/sudoers
vi /etc/sudoers
```

找到 `root ALL=(ALL) ALL` 下面添加:
`git	ALL=(ALL)	ALL`

保存退出后 将 sudoers 文件权限改回原样
`chmod 400 /etc/sudoers`
设置 git 用户的密码
`sudo passwd git`
切换到 git 用户，然后在~目录下创建.ssh 文件夹

```
su git
cd ~
mkdir .ssh
cd .ssh
```

然后为 git 配置密钥，生成公钥密钥文件
`ssh-keygen`
文件夹中生成 RSA 私钥和公钥如下：
`id_rsa 和 id_rsa.pub` 
复制一份公钥文件 id_rsa.pub 
`cp id_rsa.pub authorized_keys`
修改 authorized_keys 的权限

```
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

#### 个人电脑端链接服务器

 **然后在自己的电脑上链接云服务器** 

`ssh -v git@你的公网 IP`

登录成功提示

 **Welcome to Alibaba Cloud Elastic Compute Service !** 

登陆后创建一个 git 的仓库，并且新建一个 post-receive 文件，操作如下:

```
cd ~
git init --bare blog.git
vi ~/blog.git/hooks/post-receive
```

输入以下内容：
`git --work-tree=/home/www/website --git-dir=/home/git/blog.git checkout -f`
保存退出并授予该文件可执行权限
`chmod +x ~/blog.git/hooks/post-receive`

在 **服务器端** 重新启动服务

`nginx -s reload`

### 本地主机部署

#### 安装 Node.js 和 git

`curl -sL https://rpm.nodesource.com/setup_14.x | bash - `

`yum install -y nodejs`
`yum install git`

#### 安装 Hexo

通过淘宝镜像安装 cnpm

```
npm install -g cnpm --registry=https://registry.npm.taobao.org
#设置环境变量
sudo ln -s /opt/node/bin/cnpm /usr/local/bin/cnpm
```

全局安装 hexo-cli

```
cnpm install -g hexo-cli

sudo ln -s /opt/node/bin/hexo /usr/local/bin/hexo  
--配置环境变量
cnpm #验证安装成功
```

新建 blog 文件夹, 然后进入文件夹，使用 hexo 初始化

```
cd blog
hexo init
```

安装发布和本地展示的插件

```
npm install hexo-deployer-git --save
npm install hexo-server
```

#### 配置_config.yml 完成服务器的部署

```
deploy:
type: git
repo: git@这里改为服务器公网 IP:/home/git/blog.git       
branch: master                           
message: 
```



到这里就完成了 hexo 的搭建，在本地主机创建 hexo 文章并生成后，使用 hexo d 将会把博客内容通过 git 同步到服务器上。关于 hexo 的一些基本操作，如写文章，发布等，可以参考 hexo 官网的教程。

### Yilia 主题个性化配置

#### 初始化 yilia

初始化下载与配置

```
sudo su
git clone https://github.com/litten/hexo-theme-yilia.git themes/yilia
vi _config.yml
```

[yilia github 网页](https://github.com/litten/hexo-theme-yilia)

在 /blog/_config.yml 文件中，把 theme 改为添加的主题，author 名称改成你自己的名字或者网站的名称

yilia 主题框架修改路径为`themes/yilia/layout/_partial`

网页的布局大部分都在该文件夹中，可以通过修改其中的文件来满足部分个性化需求。

#### 左侧栏内容添加

主题文件中 _config.yml 文件添加

```
author: #添加网站名称
subtitle: # 添加网站标语
```

#### 修改 Latex 支持方案

```
npm uninstall hexo-renderer-marked --save
npm install hexo-renderer-kramed --save
```

打开`/node_modules/hexo-renderer-kramed/lib/renderer.js`
将：

```csharp
// Change inline math rule
function formatText(text) {
    // Fit kramed's rule: $$ + \1 + $$
    return text.replace(/`\$(.*?)\$`/g, '$$$$$1$$$$');
}

```

修改为：

```csharp
// Change inline math rule
function formatText(text) {
    return text;
}
```

卸载 hexo-math

```csharp
npm uninstall hexo-math --save
```

安装 hexo-renderer-mathjax

```csharp
npm install hexo-renderer-mathjax --save
```

更新 Mathjax 配置文件

打开`/node_modules/hexo-renderer-mathjax/mathjax.html`
将最后一行的`<script>`改为：

```csharp
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>
```

更改默认转义规则

因为 LaTeX 与 markdown 语法有语义冲突，所以 hexo 默认的转义规则会将一些字符进行转义，所以我们需要对默认的规则进行修改.
打开`/node_modules\kramed\lib\rules\inline.js`
更改后为：

```csharp
escape: /^\\([`*\[\]()#$+\-.!_>])/,

em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
```

开启 mathjax

打开主题目录下的`config.yml`文件（注意这里是所使用主题的 yml 文件，不是 hexo 的 yml 文件）
加入下面这句：

```csharp
mathjax: true
```

并且写博客的时候需要开启 Latex 就需要加上头开启 Mathjax：
例如：

```csharp
title: Python 向信号中添加不同强度 dB 的噪声
date: 2019-10-23 14:05:37
tags: [噪声, ECG]
categories: [Python]
mathjax: true
```

#### 采用本地文件来源

（包括文中的图片文件或者网页显示的其他本地文件）

展示图片内容时，通过网络图片链接来发布会有较大延迟，建议使用本地图片链接

系统的默认文件源为/blog/source，因此将文件添加至/source 目录下，然后进行引用就行。

如在 yilia _config.yml 中配置头像: 应先将 touxiang.jpg 保存在/blog/source 下，然后再_config.yml 中修改头像链接为： /touxiang.jpg

博客文章中添加图片也可以直接使用本地链接如 在/blog/source 下创建 img 文件夹，图片放置于 img 中。再博文中使用 img1(/assets/img/filename/img1.jpg)

#### 视频大小自适应浏览器

Hexo 的视频直接采用嵌入链接的化无法实现高度自适应。解决方案：先创建一个 div 框架，然后将视频以 HTML 形式放在 div 中来实现高度自适应。在 markdown 文件中可能看不到视频的效果，但是网页上可以。代码如下

```
<div style="height: 0;padding-bottom:65%;position: relative;"><br><iframe width="760" height="510" src="你的视频源" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen style="position: absolute;height: 105%;width: 100%;"> </iframe><br></div>
```

#### 网站的图标修改

制作 favicon.ico 放在 source 下

在_config.yml 中配置 favicon 参数为图片路径

#### 备份

可以使用 git 上传至 gitee 或者 github。

不会使用 git 的伙伴也可以使用下面链接的快捷方法。该链接就是把 git add/git commit 等一系列操作编辑成了 hexo b 。

https://github.com/coneycode/hexo-git-backup

#### Yilia 添加评论系统

评论系统中，个人觉得 valine 的白色风格最适合 yilia。具体的配置步骤在下面的链接中有具体介绍。

https://valine.js.org/

https://blog.csdn.net/qq_43827595/article/details/101450966

1. config.yml

yilia 主题根目录下的_config.yml，不是博客根目录下的_config.yml，打开后添加到什么位置都可以，但是建议根据已经写好的几个评论系统就对应的添加到#5、Gitment 的下方

```yml
#6、Valine https://valine.js.org
valine: 
 appid: #Leancloud 应用的 appId
 appkey: #Leancloud 应用的 appKey
 verify: false #验证码，verify 和 notify 这两个最好就别动了
 notify: false #评论回复提醒
 avatar: mm #评论列表头像样式：''/mm/identicon/monsterid/wavatar/retro/hide
 placeholder: Just go go #评论框占位符
12345678
```

2. yilia\layout_partial\article.ejs

这段代码改过一点点，添加了一个判断条件如果在博客首页就不执行下面的代码（具体添加了下面代码的第一行和最后一行），也就是只在阅读全文的时候才在文章底部显示评论框，原 Issue 下的代码不含这个判断条件，也就是首页每篇博客下方都会有一个评论框，这应该不是我们想要的。 代码直接插入在最后面就可以

```js
<% if (!index){ %>
   <% if (theme.valine && theme.valine.appid && theme.valine.appkey){ %>
       <section id="comments" class="comments">
         <style>
           .comments{margin:30px;padding:10px;background:#fff}
           @media screen and (max-width:800px){.comments{margin:auto;padding:10px;background:#fff}}
         </style>
         <%- partial('post/valine', {
           key: post.slug,
           title: post.title,
           url: config.url+url_for(post.path)
           }) %>
     </section>
   <% } %>
<% } %>
123456789101112131415
```

3.valine.ejs

这个文件是没有的，按照路径 yilia\layout_partial\post\valine.ejs 新建一个 valine.ejs，再把下面的代码添加添加进去，保存，注意和 Issue 给出的代码可能不太一样，说过了不想踩坑请直接使用我给出的。

```js
<div id="vcomment" class="comment"></div>
<script src="//cdn1.lncld.net/static/js/3.0.4/av-min.js"></script>
<script src='//unpkg.com/valine/dist/Valine.min.js'></script>
<script src="https://cdnjs.loli.net/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
<script>
  var notify = '<%= theme.valine.notify %>' == true ? true : false;
  var verify = '<%= theme.valine.verify %>' == true ? true : false;
  new Valine({
    av: AV,
    el: '#vcomment',
    notify: notify,
    app_id: "<%= theme.valine.appid %>",
    app_key: "<%= theme.valine.appkey %>",
    placeholder: "<%= theme.valine.placeholder %>",
    avatar: "<%= theme.valine.avatar %>",
  });
</script>
1234567891011121314151617
```

安装评论 【二】

接下来就要使用到 Leancloud 了，大概就是作为我们 Valine 评论系统的服务器，因为 Valine 首页就介绍了 Valine 是“一款快速、简洁且高效的无后端评论系统”，自行注册一个账号并登录。

创建一个应用，应用名看个人喜好。
![20190831181719.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pLmxvbGkubmV0LzIwMTkvMDgvMzEvREJYdWtjZXNwaTNuZFRXLnBuZw?x-oss-process=image/format,png)

选择刚刚创建的应用>设置>选择应用 Key，然后你就能看到你的 App ID 和 App Key 了，参考下图：
![20190831181744.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pLmxvbGkubmV0LzIwMTkvMDgvMzEvRDJhdjVyZjZlWklNS09HLnBuZw?x-oss-process=image/format,png)

分别复制 App ID 和 App Key 粘贴到前面设置的主题根目录下的_config.yml 里对应位置，注意“:”后面必须要有一个空格，如图：
![20190831181902.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pLmxvbGkubmV0LzIwMTkvMDgvMzEvbExmalFJM2NlcW9aRUt5LnBuZw?x-oss-process=image/format,png)
为了数据安全，再填写一下安全域名，应用>设置>安全设置中的 Web 安全域名，如果是 Hexo 一般填写自己博客主页的地址和 http://localhost:4000/就可以了。

#### 翻页 bug

在 yilia/layout_artial/archive.ejs

修改 prev_text 与 next_text 的值为‘上一页’，‘下一页’

![image-20210120114735200](/assets/img/blog/image-20210120114735200.png)

然后再 yilia\layout_partial\script.ejs 中修改 `rel="prev">`后的&laquo，Prev 为上一页。修改`rel="next">`后的 Next &laquo，为下一页。

![](/assets/img/blog/image-20210120115025316.png)

修改每页的显示博文数量

再 blog/_config.yml 文件中找到

```
index_generator:
  path: ''
  per_page: 4
  order_by: -date
```

可以自己根据博文的风格修改 per_page，即每页显示的博文数量

### 提交百度搜索

https://ziyuan.baidu.com/linksubmit/index 先登录百度账号，验证你的网站。接着跟着连接上的提示操作就行了。

## 参考：

[nodejs](https://blog.csdn.net/mrzhouxiaofei/article/details/66974644?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.add_param_isCf&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.add_param_isCf) 

