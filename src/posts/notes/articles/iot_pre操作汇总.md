---
title: 基于 Arduino MKR 1010 WIFI|IoT 药盒
date: 2021-04-26
author: Kevin 吴嘉文
category:
- Project|项目
tag:
- IoT
mathjax: true
toc: true
comments: 
---

> 一个很马虎的基于 Arduino MKR 1010 WIFI 的项目，主要用户为忙碌的上班组，或者需要定时了解患者用药情况的监护人。药盒可以在他们忘记吃药的时候发送消息提示（邮件或者 whatsapp），同时他们的用药情况也将会被告知他们的家人/监护人。
>
> 文章为产品功能介绍，技术实现与环境搭载经验分享。服务器搭建的步骤建议在 虚拟机上尝试。本项目服务器采用 ubuntu20.04 server，web 服务器使用 BOA。设备端，药盒是基于 arduino 开发的，通过 MQTT 协议与服务器通信。
>
> 本文中代码均为截取，具体源码请移步 [我的 github](https://github.com/kevinng77/iot_pill_box)

<!--more-->

## 产品使用与功能

### 药盒初始化

初次使用药盒时候需要进行初始化配置：1.设置药盒 WIFI 密钥; 2.设置提醒用药时间。

#### 将药盒连接至 WIFI

+ 用户链接药盒的 WIFI pill_box_setting，通过浏览器登录 IP 地址： 192.168.4.1 进行配置。

![相关图片](/assets/img/iot_pre/c05740d9d7f86a18b706357469e9cef.png =x300)

+ 用户设置 username，家庭网络的 WIFI SSID 和 KEY 并提交。药盒实现联网。

#### 用户设置提醒用药时间

+ 第一次使用产品时，用户需要登陆官网 http://ngkaman.nat300.top/ 进行注册，填写药盒的 username，密码，监护人邮箱，个人邮箱，三次用药的提示时间，以及晚上填充药时间。

![image-20210419224627230](/assets/img/iot_pre/image-20210419224627230.png)

*（图：产品官网截图）*

+ 系统自动初始化用药时间为 8 点，12 点，18 点，用户可以在产品官网登陆账号进行用药时间的重置。
+ 用可以通过官网修改邮箱的联系方式

### 用户使用产品期间：

+ 药盒发出三次用药提醒：

  + 用药时间点，药盒进行第一次判断，若对应的药槽还有残留药物，则药盒判定为用户没有用药，用户将会接收到用药提醒邮件。

    ![相关图片](/assets/img/iot_pre/image-20210423125225426.png =x300)

    *（图：未吃药邮件通知）*

  + 若第一次判断用户没有吃药，药盒会再 10 分钟后进行第二次判断，若对应的药槽还有残留药物，则该用药用户将会接收到用药第二次提醒邮件。

  + 若第二次判断用户没有吃药，药盒会再 10 分钟后进行第三次判断，若对应的药槽还有残留药物，则 **患者监护人** 将会接收到用药提醒邮件。后续药盒将不再通知患者进行用药。

+ 每晚填药提醒：夜晚填药时间，若存在药槽未填充药物，用户联系人将会收到填药提醒。

+ 每月用户会收到文字版月度用药记录

## 信息传输细节

所有的消息都是通过 MQTT 协议进行发布与订阅，本项目中对传输的消息并没有加密，容易造成他人的恶意数据改造。

![image-20210424181229871](/assets/img/iot_pre/image-20210424181229871.png)

## 设计与考虑

监护人/亲属必须使用网络获取用药者用药记录，因此 IoT 是产品实现的必须选择。

 **为什么不用手机 APP？** 

通过 Androd APP 编写 MQTT 实现消息的接收同样可以实现绝大部分的产品功能，而且相对的通过手机 APP 可以更好的控制各种消息的接法，无需考虑由于使用邮件传输或者 Whatsapp 等其他主流通讯传输手段的成本，设计开发起来也更加容易，然而 Androd APP 存在部分考虑：

1. 后台运行或消息发送被屏蔽。相对于通过专属 APP 进行消息推送，用户更经常检查 whatsapp 推送的消息，因此避免了因手机系统屏蔽问题而产生的消息遗漏。
2. 安装软件会是用户考虑使用该产品的因素，用户需要时间去熟悉 app 界面与操作，接受 APP 的隐私协议，这些都是用户流失的潜在隐患，对于本产品而言，并没有太多使用 app 的必要。
3. 项目开始使用的是 Whatsapp 通知，Whatsapp 的成本会比 Androd App 高，起初的尝试是使用 Selenium 控制 google drive 进行消息推送。如果需要使用稳定的 whatsapp 推送，需要申请使用 Whatsapp Business，申请的成本或许会比写一个 APP 要高很多。但是由于网页的源代码的变化 Selenium 控制的 Whatsapp 并不稳定，所以最后使用 SMTP 邮箱通知代替。如果可以使用 Whatsapp Business API，那么效果肯定是最好的。

 **Arduino MRK 1010 的替代品** ：

理论上 ESP8266 或者 ESP32 实现的 WIFI 设备都可以实现这些功能。

 **为什么使用 IR sensor（红外避障传感器）对药物进行检测？** 

不同于大多数的智能药盒，通过药盒槽门的开关来判断用户是否吃药，使用 IR 传感器对药物直接进行判断可以避免了用户开了盒子却又忘了吃药的情况。IR 传感器的成本相对较低，对近距离的物体检测较为准确。为了避免误报，部分药盒面也被设计处理成反射红外量少的黑色。

![相关图片](/assets/img/iot_pre/a076f85fd0c7dfcf28250040a995b8d.jpg )*（药盒半成品，为了演示并没有放上盖子）*	

传感器部分使用了 IR sensor（红外避障传感器）对药物进行简单的检测。然而传感器的准确度也影响了药盒的准确度。为了减小因为传感器的失误而发送的 fakenews，程序只有再进入了用户设定的用药时间段才会进行药物检测。

# 环境配置

服务器系统为 Ubuntu 20.04

### 配置 MQTT 

MQTT 由 IBM 公司开发的，90 年代的产物，IBM 为了解决是由公司管道检测问题，当时卫星数量不够，卫星通信比基站通信费用更高，管道几百公里，每隔几百米就会安插采集点。

解决问题：
1、服务器必须要实现成千上万客户端的接入
2、单次数据数据量小，但不能出错 
3、必须能够适应高延迟、偶尔断网等通信不可靠的风险
4、根据数据的重要程度和特性，设置不同等级的服务质量(session)

以下使用 mosquitto，用其他的也行。

### mosquitto 在 ubuntu20.04 环境安装：

[mosquitto 官网](https://mosquitto.org/)

先尝试是否可以直接动过 `apt-get install`安装

```sh
sudo apt-add-repository ppa:mosquitto-dev/mosquitto-ppa
sudo apt-get update
sudo apt-get install mosquitto -y
sudo apt-get install mosquitto-clients
```

若遇到`apt-add command not found`

```sh
sudo apt-get install python-software-properties
sudo apt-get update
sudo apt install software-properties-common 
sudo apt-get update
```

 **开放防火墙 1883/tcp** 

检查防火墙情况 `sudo ufw status`

```shell
sudo ufw enable
sudo ufw allow 80
sudo ufw allow 1883
# 或使用 sudo ufw allow 1883/tcp
ufw reload
```

```shell
sudo iptables -I INPUT -p tcp --dport 1883 -j ACCEPT
sudo iptables-save
sudo apt-get install iptables-persistent
sudo netfilter-persistent save
sudo netfilter-persistent reload
```

然后把`/etc/mosquitto/mosquitto.conf`中注释掉下面这两行

```shell
# persistence_location /var/lib/mosquitto/
# log_dest file /var/log/mosquitto/mosquitto.log
```

![相关图片](/assets/img/iot_pre/image-20210414170432612.png )

配置工作到这里就完成了

### mosquitto 基本操作：

 **启动服务器， 指定配置文件位置**  

```shell
mosquitto -c /etc/mosquitto/mosquitto.conf -p 1883 -d
```

在本机中测试更多参数选择操作:

```shell
mosquitto_sub -d -v -t temp -h 192.168.235.130 -p 1883 -q 2 //-h 指定主机 -p 指定端口 -q 指定通讯质量
mosquitto_pub -d -t temp -h 192.168.235.130 -p 1883 -m hello -q 2 //对于 public 也一样可以指定主机和端口
```

完成本机测试后，在一个终端下订阅是可以接收到另一个终端发布的内容的，要实现不同 ip 间的通讯，需要设置非匿名登录。Mosquitto 最新版本默认不允许匿名外网访问。

### mosquitto 非匿名登录配置：

修改配置 `/etc/mosquitto/mosquitto.conf`

添加：

`allow_anonymous false`

`password_file /etc/mosquitto/passwd.conf`

`listener 1883`

 **服务端创建用户**  

+ 隐藏密码创建

`sudo mosquitto_passwd -c
/etc/mosquitto/passwd.conf username` 

+ 明文创建

`sudo mosquitto_passwd -b /etc/mosquitto/passwd.conf username pwd`  

 **配置好后测试一下，先启动服务器** 

`mosquitto -c /etc/mosquitto/mosquitto.conf`

在计算机 A 上订阅：`mosquitto_sub -t "temp" -u username -P 111111` 

在计算机 B 上发布：`mosquitto_pub -t "temp" -m "hello" -u username -P 111111`

A 收到消息表示成功

以上的方式中，传输的信息是明文，并不安全，真实使用中应该在进行 TLS 加密，具体请查看[我的博客], 若无法通过 apt-get 安装，那么就需要手动移植。

### API

[MQTT C API](https://mosquitto.org/api/files/mosquitto-h.html)

注意编译时使用：

```shell
gcc mosquito.c -o mosquito -lmosquitto
```

 **Python API** 

```shell
sudo apt update
sudo apt full-upgrade -y
sudo apt install python3.9 python3.9-venv
```

使用搭建安全的虚拟环境:

```shell
mkdir ~/apps
cd ~/apps
rm -rf ~/apps/env

python3.9 -m venv ~/apps/env
. ~/apps/env/bin/activate
pip3.9 install wheel
pip3.9 install pyserial paho-mqtt requests dweepy
```

安装好之后就可以用了，本次项目的主要 mqtt 通信也是通过 python API 完成 [案例链接](https://github.com/kevinng77/iot_pill_box/tree/main/mqtt_py)，使用 python 主要因为本人对于 python 更了解，C 与 python 的孰优孰劣并不晓得。python API 还是很方便的查看 paho-mqtt 的 documentation 了解到更多的操作。

## 配置 natapp 内网穿透

[natapp 网站](https://natapp.cn/tunnel/buy/free)

具体的配置过程[这篇博客](https://blog.csdn.net/hyh17808770899/article/details/108936090)有了详细的介绍，因为免费的 natapp 只提供一个端口的内网穿透，因此本项目中申请了两个隧道，分别对应 1883/tcp 与 80，获得两个 authtoken 后配置 sh 文件分别执行两次 natapp 就行了

实现后台运行可以运行命令`nohup ./natapp -authtoken=xxxx -log=stdout &`实现，但是这样的话我们就无法看到运行后随机域名是多少，所以需要进行如下配置：

编写脚本 natapp.sh

```shell
vi natapp.sh
```

添加下面的语句，然后保存退出

```shell
#!/bin/bash
rm /home/kevin/Desktop/iot/natapp/nohup.out
touch nohup.out
service mosquitto stop
mosquitto -c /etc/mosquitto/mosquitto.conf -d
/home/kevin/share/myboa/boa/boa
cd /home/kevin/Desktop/iot/natapp
nohup /usr/local/natapp/natapp &
nohup /usr/local/natappweb/natapp &
```

对该脚本进行授权

```java
chmod 777 natapp.sh
```

在 `/usr/local/natapp/natapp` 和 `/usr/local/natappweb/natapp`目录下添加 config.ini 文件，当然你也可以使用 `natapp -authtoken=xxx`来执行 natapp，在运行时传入配置参数，那么就不需要配置这边的文件。

```ini
#将本文件放置于 natapp 同级目录 程序将读取 [default] 段
#在命令行参数模式如 natapp -authtoken=xxx 等相同参数将会覆盖掉此配置
#命令行参数 -config= 可以指定任意 config.ini 文件
[default]
authtoken=77fa4653298558f9                      #对应一条隧道的 authtoken
clienttoken=         #对应客户端的 clienttoken,将会忽略 authtoken,若无
请留空,
log=stdout             #log 日志文件,可指定本地文件, none=不做记录,stdout=直接屏幕输出 ,默认为 none
loglevel=INFO                  #日志等级 DEBUG, INFO, WARNING, ERROR 默认为 DEBUG
http_proxy=                     #代理设置 如 http://10.123.10.10:3128 非代理上>网用户请务必留空
```

 **如果多次执行 natapp.sh 文件的话，请关掉多余的进程**  `ps -ef|grep natapp`  `kill -9 2777`

配置完成 antapp 后，我们可以通过 mosquitto 订阅，其中 ip 地址设置如下：

`mosquitto_sub -t record -h server.natappfree.cc -u user22 -P 111111 -p 35444`

端口也需要指定到 natapp 的端口

为免费的内网穿透服务搞个 ip 与 web 名称抓取，抓取后就可以自动开启 mqtt 服务了。如果是付费用户就不用了：

```python
def run_main(ip,port):
    client = mqtt.Client()
    client.reinitialise()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(ip, port, 60)
    client.loop_forever()

def get_server_ip():
    with open('../natapp/nohup.out', 'r') as fp:
        a = fp.readlines()
        ip = "192.168.235.131" 
        port = "1883"
        web_add = ""
        for i in a:
            web = re.findall('http://.+\.natappfree\.cc', i)
            if web:
                web_add = web[0]
            ip_port = re.findall('tcp://.+\.natappfree\.cc:[\d]+', i)
            if ip_port:
                ip = ip_port[0][6:-6]
                port = ip_port[0][-5:]
    return ip, int(port), web_add

if __name__ == '__main__':
    ip,port,web_add = get_server_ip()
    print(f"ip:{ip},port:{port},web:{web_add}")
    run_main(ip, port)
```

## Whatsapp 控制

在许久的查阅后，通过服务器实现发送 whatsapp 通知的方法有两种，一是通过 whatsapp Business API，使用需要有公司进行注册，好吧那只好为了我的小 project 去注册一个公司了 doge.jpg。

另一种方式也是本项目中采用的，使用 selenium 来控制浏览器实现消息发送。网上有人发布了一个 python whatsapp-web 的库，但是并不实用，也是通过 selenium 来实现，然鹅 whatsapp 的 xpath 随着时间和地区改变，那个包自然也会出现很多 bug。读者可以尝试一下，不行的话不必太执着，自己写效率也很高。[whatsapp-web 链接](https://pypi.org/project/whatsapp-web/)

本项目的 Selenium driver 使用了 Chrome，首先需要安装 Chrome，网上经验丰富这边不在阐述，可以参考这篇[Ubuntu16.04 安装 chromedriver、chrome 及 运行 selenium](https://long97.blog.csdn.net/article/details/103619926?utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control)

安装后在 python 中导入

```python
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
```

建立链接

```python
driver = webdriver.Chrome('/usr/local/share/chromedriver')
driver.get("https://web.whatsapp.com/")
wait = WebDriverWait(driver, 5)
```

发送消息

```python
whatsapp_account = "+65 1234 1234"
message = f"nihao 666"
user_xpath = f'//span[@title="{whatsapp_account}"]'
group_title = wait.until(EC.presence_of_element_located((By.XPATH,user_xpath)))
group_title.click()
inp_xpath = '//div[@class="_2A8P4"]'
input_box = wait.until(EC.presence_of_element_located((By.XPATH, inp_xpath)))
input_box.send_keys(message + Keys.ENTER)
time.sleep(2)
```

其中`user_xpath` 与 `inp_xpath` 分别为用户名称和输入框的 Xpath 地址，建议使用 Chrome Xpath helper 拓展插件查看。



## Web 服务器搭建

本项目使用的是 BOA 服务器，具体的搭建教程网上很丰富，这边不展开讨论。

下载源码 http://www.boa.org/

创建相关文件路径

```sh
mkdir /home/kevin/share/myboa
cd myboa
mkdir boa
mkdir log
mkdir www
mkdir www/cgi-bin
```

安装 `sudo apt-get install bison sudo apt-get install flex`

解压 `tar -xzf boa-0.94.13.tar.gz`

进入 `cd src ./configure` 配置文件

修改 defines.h 30 行左右的 服务器根路径为`boa.conf`存放的路径

`#define SERVER_ROOT "/home/linux/share/myboa/boa"`

注释掉 boa.c 226 行左右的 

`if (setuid(0) != -1) {DIE("icky Linux kernel bug!");}`

修改 compat.h  120 行左右的 `#define TIMEZONE_OFFSET(foo) foo##->tm_gmtoff` 为：

`#define TIMEZONE_OFFSET(foo) (foo)->tm_gmtoff`

编译 make

将 boa-0.94.13/src 目录下生成的两个二进制文件复制到指定的 boa 目录下
`cp boa /home/kevin/share/myboa/boa`
`cp boa_indexer/home/kevin/share/myboa/boa`
(2)将 boa-0.94.13 目录下的 boa.conf 复制到指定的 boa 目录下
`cp boa.conf /home/kevin/share/myboa/boa`

创建 log 文件

`touch /home/kevin/share/myboa/log/error_log`
`touch /home/kevin/share/myboa/log/access_log`

配置 boa.conf 里面的内容 `cd /home/kevin/share/myboa/boa vi boa.conf`

```c
User 0 Group 0
ErrorLog /home/kevin/share/myboa/log/error_log
AccessLog /home/kevin/share/myboa/log/access_log
DocumentRoot /home/kevin/share/myboa/www
DirectoryMaker /home/kevin/share/myboa/boa/boa_indexer
ScriptAlias /cgi-bin/ /home/kevin/share/myboa/www/cgi-bin/
```

### web 设计

web 主要为用户提供 注册账户，修改 whatsapp，修改药盒提示时间功能。本次项目的 HTML 设计主要改动与 [Day 001 login form](https://codepen.io/khadkamhn/pen/ZGvPLo) 仅学习，非商用。源码也放在了 [我的 github](https://github.com/kevinng77/iot_pill_box/tree/main/www) 上。

### 实现原理：

配置 `update.js`文件，通过 CGI 运行 C 程序。从表单中提取对应的信息，通过 sqlite3 API 与 mosquitto API，将用户修改的时间发送到 主题 setting/+ 下。设备端订阅该主题，接收到后充值时间变量，从而实现通过网页端改变设备上的时间。



