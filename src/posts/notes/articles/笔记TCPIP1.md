---
title: 嵌入式学习之（八）| 计算机网络 1
date: 2021-04-02
author: Kevin 吴嘉文
keywords: 
language: cn
category:
- 知识笔记
tag:
- TCP/IP
- 嵌入式学习
mathjax: true
toc: true
comments: 
---

> 嵌入式自学开始几个月啦？ 
>
> 断断续续的忙碌，但总能提醒自己在忙也要做自己想做的。
>
> 为了自己的贾维斯
>
> 
>
> 笔记总结 课程链接：[千峰嵌入式教程](https://www.bilibili.com/video/BV1FA411v7YW?p=530&spm_id_from=pageDriver)

<!-- more-->

## 分组交换

![image-20210310153610191](/assets/img/信号/image-20210310153610191.png)

首部包含计数器，发送方等。 接受者通过将某一个发送方的数据包通过计数器排列后得到报文。

 **交换方式—存储转发** 
节点收到分组，先暂时存储下来，再检查其首部，按照首部中的目的地址，找到合适的节点转发出去

 **特点：**  
1、以分组作为传输单位
2、独立的选择转发路由
3、逐段占用，动态分配传输带宽

## TCP/IP 协议族

为了能够实现不同类型的计算机和不同类型的操作系统之间进行通信，引入了分层的概念

最早的分层体系结构是 OSI 开放系统互联模型，是由国际化标准组织（ISO）指定的，由于 OSI 过于复杂，所以到现在为止也没有适用，而使用的是 TCP/IP 协议族
OSI 一共分为 7 层，TCP/IP 协议族一共四层，虽然 TCP/IP 协议族层次少，但是却干了 OSI7 层所有任务

![image-20210310152818968](/assets/img/信号/image-20210310152818968.png)

 **应用层：** 应用程序间沟通的层
例如：FTP、Telnet、HTTP 等

 **传输层：** 提供进程间的数据传送服务
负责传送数据，提供应用程序端到端的逻辑通信
例如：TCP、UDP

 **网络层：** 提供基本的数据封包传送功能
最大可能的让每个数据包都能够到达目的主机
例如：IP、ICMP 等

 **链路层：** 负责数据帧的发送和接收

![image-20210310152843120](/assets/img/信号/image-20210310152843120.png)

### IP 协议/网际协议

特指为实现在一个相互连接的网络系统上从源地址到目的地传输数据包（互联网数据包）所提供必要功能的协议

 **特点：** 
不可靠：

+ 不能保证 IP 数据包能成功地到达它的目的地，仅 **提供尽力而为的传输服务** 

无连接：

+ IP 并不维护任何关于后续数据包的状态信息。每个数据包的处理是相互独立的。
+ IP 数据包可以不按发送顺序接收

 **IP 数据包中含有发送它主机的 IP 地址（源地址）和接收它主机的 IP 地址（目的地址）**  

### TCP - 传输控制协议

TCP 是一种面向连接的,可靠的传输层通信协议

 **功能：**  提供不同主机上的进程间通信

 **特点** 
1、建立链接->使用链接->释放链接（虚电路）
2、TCP 数据包中包含序号和确认序号  **（保证数据可靠）** 
3、对包进行排序并检错，而损坏的包可以被重传

 **服务对象** 
需要高度可靠性且面向连接的服务,如 HTTP、FTP、SMTP 等

### UDP 协议 - 用户数据报协议

UDP 是一种面向无连接的传输层通信协议

 **功能：** 
提供不同主机上的进程间通信

 **特点** 
1、发送数据之前不需要建立链接
2、不对数据包的顺序进行检查
3、没有错误检测和重传机制

 **服务对象** 
主要用于“查询—应答”的服务

如：NFS、NTP、DNS 等 微信语音，视频通话等，传输速度快，卡了就卡了。不会像 TCP 一样一直传输失败内容

## MAC

网卡 - 网络接口卡 NIC

### MAC 地址

每一个网卡在出厂时，都会给分配一个编号，这个编号就称之为 mac 地址
MAC 地址,用于标识网络设备,类似于身份证号，且理论上全球唯一

 **组成：** 以太网内的 MAC 地址是一个 48bit 的值，通常人为识别时是通过 16 进制数来识别的，以两个十六进制数为一组，一共分为 6 组，每组之间通过:隔开，前三组称之为厂商 ID，后三组称之为设备 ID

如何查询 ubuntu 的 mac 地址：`ifconfig`

如何查询 windows 的 mac 地址

​	鼠标右键点击计算机右下角电脑图标，选择网络和共享中心，左边选择更改适配器设置，找
到自己联网的图标，双击打开，点击详细信息，即可找到对应的 mac 地址

![image-20210310155802080](/assets/img/信号/image-20210310155802080.png)

### IP 地址

 **IP 地址的分类** 
ipv4，占 32 位 （目前主要用到）
ipv6，占 128 位
 **IPV4 地址的组成** 
ipv4 一般使用点分十进制字符串来标识，比如 192.168.3.103

使用 32bit,由{网络 ID，主机 ID}两部分组成
子网 ID:IP 地址中由子网掩码中 1 覆盖的连续位
主机 ID:IP 地址中由子网掩码中 0 覆盖的连续位

![image-20210310160043387](/assets/img/信号/image-20210310160043387.png)

 **ip 地址特点** 
子网 ID 不同的网络不能直接通信，如果要通信则需要路由器转发
主机 ID 全为 0 的 IP 地址表示网段地址
主机 ID 全为 1 的 IP 地址表示该网段的广播地址
例如：
192.168.3.10 和 192.168.3.111 可以直接通信
如果 192.168.3.x 网段而言，192.168.3.0 表示网段，192.168.3.255 表示广播地址

 **ipv4 地址的分类（依据前八位来进行区分）** 
A 类地址：默认 8bit 子网 ID,第一位为 0，前八位 00000000 - 01111111,范围 0.x.x.x -
127.x.x.x
B 类地址：默认 16bit 子网 ID,前两位为 10，前八位 10000000 - 10111111,范围
128.x.x.x-191.x.x.x
C 类地址：默认 24bit 子网 ID,前三位为 110,前八位 11000000 - 11011111,范围
192.x.x.x-223.x.x.x
D 类地址：前四位为 1110,组播地址，前八位 11100000-11101111，范围 224.x.x.x-
239.x.x.x
E 类地址: 前五位为 11110,保留为今后使用，前八位 11110000-11111111，范围
240.x.x.x-255.x.x.x
A,B,C 三类地址是最常用的

 **私有 ip 地址** 
公有 IP（可直接连接 Internet）
经由 InterNIC 所统一规划的 IP
私有 IP（不可直接连接 Internet ）
主要用于局域网络内的主机联机规划

![image-20210310161130759](/assets/img/信号/image-20210310161130759.png)

 **回环 ip 地址** 
通常 127.0.0.1 称为回环地址
 **功能** 
主要是测试本机的网络配置，能 ping 通 127.0.0.1 说
明本机的网卡和 IP 协议安装都没有问题
 **注意** 
127.0.0.1~127.255.255.254 中的任何地址都将回环到本地主机中
不属于任何一个有类别地址类,它代表设备的本地虚拟接口

 **查询 ip 一直的命令** 
在 ubuntu 中 `ifconfig`

lo - 本地回环

![image-20210310164245947](/assets/img/信号/image-20210310164245947.png)

在 windows 中 `ipconfig`

 **如何判断主机是否可以连通通信** 
`ping ip 地址`

如果现实 0% packet loss，就表示可以正常通信

### 子网掩码 subnet mask

子网掩码（subnet mask）又叫网络掩码、地址掩码是一个 32bit 由 1 和 0 组成的数值，并且 1 和 0 分别连续

 **作用** 
指明 IP 地址中哪些位标识的是主机所在的子网以及哪些位标识的是主机号
 **特点** 
必须结合 IP 地址一起使用，不能单独存在 

IP 地址中由子网掩码中 1 覆盖的连续位为子网 ID,其余为主机 ID

 **子网掩码的表现形式** 
192.168.220.0/255.255.255.0
192.168.220.0/24 (有 24 个连续的 1)

手动进行配置如下(linux)

![image-20210310164825710](/assets/img/信号/image-20210310164825710.png)

 **默认的子网掩码** 
A 类 ip 地址的默认子网掩码为 255.0.0.0
B 类 ip 地址的默认子网掩码为 255.255.0.0
C 类 ip 地址的默认子网掩码为 255.255.255.0



## 端口
TCP/IP 协议采用端口标识通信的 **进程** 
用于区分一个系统里的 **多个进程** 
 **特点** 
1、对于同一个端口，在不同系统中对应着不同的进程

2、对于同一个系统，一个端口只能被一个进程拥有
3、一个进程拥有一个端口后，传输层送到该端口的数据全部被该进程接收，同样，进
程送交传输层的数据也通过该端口被送出

 **端口号** 
类似 pid 标识一个进程 （进程号可变，端口号不变）；在网络程序中，用端口号（port）来标识一个运行的网络程序

 **特点** 
 **1** 、端口号是无符号短整型的类型
2、每个端口都拥有一个端口号
3、TCP、UDP 维护各自独立的端口号
4、网络应用程序,至少要占用一个端口号,也可以占有多个端口号

 **知名端口（1~1023）** 
由互联网数字分配机构(IANA)根据用户需要进行统一分配
例如：FTP—21，HTTP—80 等
服务器通常使用的范围;
若强制使用,须加 root 特权

 **动态端口（1024~65535）** 
应用程序通常使用的范围
一般我们可以使用的端口号就是在这个范围，比如 6666、7777、8888、9999、
10000、10001
注意
端口号类似于进程号，同一时刻只能标志一个进程
可以重复使用

## 数据包组装拆解

 **数据包在各个层之间的传输**  

数据在各个层封装/拆解各种头部

![image-20210310165906997](/assets/img/信号/image-20210310165906997.png)

###  **链路层封包格式** 

![image-20210310170341866](/assets/img/信号/image-20210310170341866.png)

 **大多数使用以太网封装** 

 **目的地址** ：目的 mac 地址
 **源地址** ：源 mac 地址
 **类型：** 确定以太网头后面跟的是哪个协议
0x0800 ip 协议
0x0806 arp 协议
0x0835 rarp 协议

 **注意** 
1、IEEE802.2/802.3 封装常用在无线
2、以太网封装常用在有线局域网

###  **网络层、传输层封包格式** 

![image-20210310170822503](/assets/img/信号/image-20210310170822503.png)

根据协议的不同定义，数据部分会封装不同格式的数据报

## 网络应用程序开发流程

 **TCP—面向连接** 
电话系统服务模式的抽象
每一次完整的数据传输都要经过建立连接、使用连接、终止连接的过程
本质上,连接是一个管道,收发数据不但顺序一致,而且内容相同
保证数据传输的可靠性

 **UDP—面向无连接** 
邮件系统服务模式的抽象
每个分组都携带完整的目的地址
不能保证分组的先后顺序
不进行分组出错的恢复和重传
不保证数据传输的可靠性

 **C/S 架构示例（面向连接）** 
无论采用面向连接的还是无连接，两个进程通信过程中，大多采用 C/S 架构
client 向 server 发出请求,server 接收到后提供相应的服务
在通信过程中往往都是 client 先发送请求，而 server 等待请求然后进行服务

# UDP 编程


### 字节序概念
是指多字节数据的存储顺序
 **分类** 
小端格式:将低位字节数据存储在低地址
大端格式:将高位字节数据存储在低地址
 **注意** 
LSB：低地址 
MSB：高地址 靠近 0x 的为高字节区

![image-20210314181037668](/assets/img/TCPIP/image-20210314181037668.png)

 **特点** 
1、网络协议指定了通讯字节序—大端
2、只有在多字节数据处理时才需要考虑字节序
3、运行在同一台计算机上的进程相互通信时,一般不用考虑字节序
4、异构计算机之间通讯，需要转换自己的字节序为网络字节序
在需要字节序转换的时候一般调用特定字节序转换函数

### 字节序转换函数

小端存储机器和大段存储机器的数据传输

```c
头文件：
#include <arpa/inet.h>

返回值：
成功：返回主机字节序的值

htonl 函数 
uint32_t htonl(uint32_t hostint32);
功能:
将 32 位主机字节序数据转换成网络字节序数据
参数：
hostint32：待转换的 32 位主机字节序数据


htons 函数
uint16_t htons(uint16_t hostint16);
功能：
将 16 位主机字节序数据转换成网络字节序数据
参数：
uint16_t：unsigned short int
hostint16：待转换的 16 位主机字节序数据


ntohl 函数
uint32_t ntohl(uint32_t netint32);
功能：
将 32 位网络字节序数据转换成主机字节序数据
参数：
uint32_t： unsigned int
netint32：待转换的 32 位网络字节序数据


ntohs 函数
uint16_t ntohs(uint16_t netint16);
功能：
将 16 位网络字节序数据转换成主机字节序数据
参数：
uint16_t： unsigned short int
netint16：待转换的 16 位网络字节序数据
```



![image-20210314182920082](/assets/img/TCPIP/image-20210314182920082.png)

0x78563412

### IP 地址转换函数

inet_pton 函数
 **字符串 ip 地址转整型数据** 

```c
int inet_pton(int family,const char *strptr, void *addrptr);
功能：
将点分十进制数串转换成 32 位无符号整数
参数：
family 协议族
AF_INET
strptr 点分十进制数串
addrptr 32 位无符号整数的地址
返回值：
成功返回 1 、失败返回其它
头文件：
#include <arpa/inet.h>
```



```c
#include <stdio.h>
#include <arpa/inet.h>
int main(int argc,char *argv[])
{
char ip_str[] = "10.0.13.100";
unsigned int ip_uint = 0;
unsigned char * ip_p =NULL;//可以用 char 吗？
inet_pton(AF_INET,ip_str,&ip_uint);
printf("ip_uint = %d\n",ip_uint);
ip_p = (unsigned char *) &ip_uint;
printf("ip_uint = %d.%d.%d.%d\n",*ip_p,*(ip_p+1),*(ip_p+2),*(ip_p+3));
return 0;
}
```

![image-20210314183220015](/assets/img/TCPIP/image-20210314183220015.png)

```c
inet_ntop 函数
整型数据转字符串格式 ip 地址
const char *inet_ntop(int family, const void *addrptr,
char *strptr, size_t len);

功能：
将 32 位无符号整数转换成点分十进制数串
参数：
family 协议族
addrptr 32 位无符号整数
strptr 点分十进制数串
len strptr 缓存区长度
len 的宏定义
#define INET_ADDRSTRLEN 16 //for ipv4
#define INET6_ADDRSTRLEN 46 //for ipv6
返回值：
成功:则返回字符串的首地址
失败:返回 NULL
```



`inet_addr(const char *cp)`和`inet_ntoa(struct in_addr in)`只能用在 IPv4 中。

## UDP

 **UDP 协议** 
面向无连接的用户数据报协议，在传输数据前不需要先建立连接；目地主机的运输层收到 UDP 报文后，不需
要给出任何确认
 **UDP 特点** 
1、相比 TCP 速度稍快些
2、简单的请求/应答应用程序可以使用 UDP
3、对于海量数据传输不应该使用 UDP
4、广播和多播应用必须使用 UDP
UDP 应用
DNS(域名解析)、NFS(网络文件系统)、RTP(流媒体)等

一般语音和视频都是 UDP



### 网络编程接口 socket
网络通信要解决的是不同主机进程间的通信
1、首要问题是网络间进程标识问题
2、以及多重协议的识别问题
20 世纪 80 年代初，加州大学 Berkeley 分校在 BSD(一个 UNIX OS 版本)系统内实现了 TCP/IP 协议；其网络程序编程开发接口为 socket
随着 UNIX 以及类 UNIX 操作系统的广泛应用， socket 成为最流行的网络程序开发接口

蓝牙 WIFI 这类通信的实现，也是使用 socket。

 **socket 作用** 
提供不同主机上的进程之间的通信
 **socket 特点** 
1、socket 也称“套接字”
2、是一种文件描述符,代表了一个通信管道的一个端点
3、类似对文件的操作一样，可以使用 read、write、close 等函数对 socket 套接字进行网络数据的收取和发送等操作
4、得到 socket 套接字（描述符）的方法调用 socket()

 **socket 分类** 
SOCK_STREAM，流式套接字，用于 TCP
SOCK_DGRAM，数据报套接字，用于 UDP
SOCK_RAW，原始套接字，对于其他层次的协议操作时需要使用这个类型

### UDP 编程 C/S 架构

![image-20210314193736665](/assets/img/TCPIP/image-20210314193736665.png)

UDP 网络编程流程：
服务器：
	创建套接字 socket( )
	将服务器的 ip 地址、端口号与套接字进行绑定 bind( )
	接收数据 recvfrom()
	发送数据 sendto()
客户端：
	创建套接字 socket()
	发送数据 sendto()
	接收数据 recvfrom()
	关闭套接字 close()

### UDP 编程-创建套接字

int socket(int family,int type,int protocol);
功能：
创建一个用于网络通信的 socket 套接字（描述符）
参数：
family:协议族(AF_INET、AF_INET6、PF_PACKET 等)
type:套接字类(SOCK_STREAM、SOCK_DGRAM、SOCK_RAW 等)
protocol:协议类别(0、IPPROTO_TCP、IPPROTO_UDP 等
返回值：
套接字
特点：

​	创建套接字时，系统不会分配端口
​	创建的套接字默认属性是主动的，即主动发起服务的请求;当作为服务器时，往往需要修改为被动的
头文件：
​	#include <sys/socket.h>

```c
#include <stdio.h>
2 #include <sys/socket.h>
3 #include <sys/types.h>
4 #include <stdlib.h>
5
6 int main(int argc, char const *argv[])
7 {
8 //使用 socket 函数创建套接字
9 //创建一个用于 UDP 网络编程的套接字
10 int sockfd;
11 if((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) == ‐1)
12 {
13 perror("fail to socket");
14 exit(1);
15 }
16
17 printf("sockfd = %d\n", sockfd);
18c
19 return 0;
20 }
```

### UDP 编程-发送、绑定、接收数据

 **IPv4 套接字地址结构** 

在网络编程中经常使用的结构体 sockaddr_in (以下结构体 linux 中都有定义)
头文件：#include <netinet/in.h>

```c
1 struct in_addr
2 {
3 in_addr_t s_addr;//ip 地址 4 字节
4 };
5 struct sockaddr_in
6 {
7 sa_family_t sin_family;//协议族 2 字节
8 in_port_t sin_port;//端口号 2 字节
9 struct in_addr sin_addr;//ip 地址 4 字节
10 char sin_zero[8]//填充，不起什么作用 8 字节
11 };
```

为了使不同格式地址能被传入套接字函数,地址须要强制转换成通用套接字地址结构，原因是因为不同场合所使用的结构体不一样，但是调用的函数却是同一个，所以定义一个通用结构体，当在指定场合使用时，在根据要求传入指定的结构体即可通用结构体 sockaddr
头文件:#include <netinet/in.h>

```c
1 struct sockaddr
2 {
3 sa_family_t sa_family; // 2 字节
4 char sa_data[14] //14 字节
5 };
```

 **两种地址结构使用场合** 
在定义源地址和目的地址结构的时候，选用 `struct sockaddr_in;`
例：
`struct sockaddr_in my_addr;`
当调用编程接口函数，且该函数需要传入地址结构时需要用 struct sockaddr 进行强制转换
例：
`bind(sockfd,(struct sockaddr*)&my_addr,sizeof(my_addr));`

 **发送数据—sendto 函数** 

```c
#include <sys/types.h>
#include <sys/socket.h>
ssize_t sendto(int sockfd, const void *buf, size_t len, int flags,
4 const struct sockaddr *dest_addr, socklen_t addrlen);
5 功能：发送数据
6 参数：
7 sockfd：文件描述符，socket 的返回值
8 buf：要发送的数据
9 len：buf 的长度
flags：标志位
0 阻塞
12 MSG_DONTWAIT 非阻塞
13 dest_addr：目的网络信息结构体（需要自己指定要给谁发送）
14 addrlen：dest_addr 的长度
15 返回值：
16 成功：发送的字节数
17 失败：‐1
```

### 网络调试助手

![image-20210314200131091](/assets/img/TCPIP/image-20210314200131091.png)

```c
#include <stdio.h> //printf
#include <stdlib.h> //exit
#include <sys/types.h>
#include <sys/socket.h> //socket
#include <netinet/in.h> //sockaddr_in
#include <arpa/inet.h> //htons inet_addr
#include <unistd.h> //close
#include <string.h>
#define N 128

int main(int argc, char const *argv[])
{
//./a.out 192.168.3.78 8080
if(argc < 3)
{
fprintf(stderr, "Usage：%s ip port\n", argv[0]);
exit(1);
}

//第一步：创建套接字
int sockfd;
if((sockfd = socket(AF_INET, SOCK_DGRAM, 0))==-1)
{
perror("fail to socket");
exit(1);
}

printf("sockfd = %d\n", sockfd);

//第二步：填充服务器网络信息结构体 sockaddr_in
struct sockaddr_in serveraddr;
socklen_t addrlen = sizeof(serveraddr);

serveraddr.sin_family = AF_INET; //协议族，AF_INET：ipv4 网络协议
serveraddr.sin_addr.s_addr = inet_addr(argv[1]); //transform ip into correct format
serveraddr.sin_port = htons(atoi(argv[2]));

//第三步：发送数据
char buf[N] = "";
while(1)
{
fgets(buf, N, stdin);
buf[strlen(buf)-1]='\0'; //把 buf 字符串中的\n 转化为\0

if(sendto(sockfd, buf, N, 0, (struct sockaddr *)&serveraddr, addrlen)==-1)
{
perror("fail to sendto");
exit(1);
 }
}c

//第四步：关闭套接字文件描述符
close(sockfd);

return 0;
}
```

`./a.out 192.168.3.78 8080` 执行

#### 绑定 BIND

由于服务器是被动的，客户端是主动的，所以一般先运行服务器，后运行客户端，所以服务器需要固定自己的信息（ip 地址和端口号），这样客户端才可以找到服务器并与之通信，但是客户端一般不需要 bind 绑定，因为系统会自动给客户端分配 ip 地址和端口号

```c
1 //第二步：将服务器的网络信息结构体绑定前进行填充
2 struct sockaddr_in serveraddr;
3 serveraddr.sin_family = AF_INET;
4 serveraddr.sin_addr.s_addr = inet_addr(argv[1]);
5 serveraddr.sin_port = htons(atoi(argv[2]));
6
7 //第三步：将网络信息结构体与套接字绑定
8 if(bind(sockfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) == ‐1)
9 {
10 perror("fail to bind");
11 exit(1);
12 }
```

#### 设置服务器（ubuntu）

```c
#include <stdio.h> //printf
2 #include <stdlib.h> //exit
3 #include <sys/types.h>
4 #include <sys/socket.h> //socket
5 #include <netinet/in.h> //sockaddr_in
6 #include <arpa/inet.h> //htons inet_addr
7 #include <unistd.h> //close
8 #include <string.h>
9
10 #define N 128
11
12 int main(int argc, char const *argv[])
13 {
14 if(argc < 3)
15 {
16 fprintf(stderr, "Usage: %s ip port\n", argv[0]);
17 exit(1);
18 }
19
20 //第一步：创建套接字
21 int sockfd;
22 if((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) == ‐1)
23 {
24 perror("fail to socket");
25 exit(1);
26 }
27
28 //第二步：将服务器的网络信息结构体绑定前进行填充
29 struct sockaddr_in serveraddr;
30 serveraddr.sin_family = AF_INET;
31 serveraddr.sin_addr.s_addr = inet_addr(argv[1]); //192.168.3.103
32 serveraddr.sin_port = htons(atoi(argv[2])); //9999
33
34 //第三步：将网络信息结构体与套接字绑定
35 if(bind(sockfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) == ‐1)
36 {
37 perror("fail to bind");
38 exit(1);
39 }
40
41 //接收数据
42 char buf[N] = "";
43 struct sockaddr_in clientaddr;
44 socklen_t addrlen = sizeof(struct sockaddr_in);
45 while(1)
46 {
47 if(recvfrom(sockfd, buf, N, 0, (struct sockaddr *)&clientaddr, &a
ddrlen) == ‐1)
48 {
49 perror("fail to recvfrom");
50 exit(1);
51 }
52
53 //打印数据
54 //打印客户端的 ip 地址和端口号
55 printf("ip:%s, port:%d\n", inet_ntoa(clientaddr.sin_addr), ntohs(c
lientaddr.sin_port));
56 //打印接收到数据
57 printf("from client: %s\n", buf);
58 }
59
60 return 0;
61 }
```

ip ifconfig 查看后，使用正确的 IP， 端口号如果别人用了，就换一个

#### 接受数据 - recvfrom 函数

```c
#include <sys/types.h>
2 #include <sys/socket.h>
3
4 ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags, struct sockaddr *src_addr, socklen_t *addrlen);
6 功能：接收数据
7 参数：
8 sockfd：文件描述符，socket 的返回值
9 buf：保存接收的数据
10 len：buf 的长度
11 flags：标志位
12 0 阻塞
13 MSG_DONTWAIT 非阻塞
14 src_addr：源的网络信息结构体（自动填充，定义变量传参即可）
15 addrlen：src_addr 的长度
16 返回值：
17 成功：接收的字节数
18 失败：‐1
```

#### 接受数据

可将网络调试助手作为客户端，发送数据

## UDP 客户端注意点

 **上文中的流程图是关键** 

1、本地 IP、本地端口（我是谁）
2、目的 IP、目的端口（发给谁）
3、在客户端的代码中，我们只设置了目的 IP、目的端口

客户端的本地 ip、 **本地 port 是我们调用 sendto 的时候 linux 系统底层自动给客户端分配的** ；分配端口的方式为随机分配，即每次运行系统给的 port 不一样

```c
//udp 客户端的实现
2 #include <stdio.h> //printf
3 #include <stdlib.h> //exit
4 #include <sys/types.h>
5 #include <sys/socket.h> //socket
6 #include <netinet/in.h> //sockaddr_in
7 #include <arpa/inet.h> //htons inet_addr
8 #include <unistd.h> //close
9 #include <string.h>
10
11 int main(int argc, char const *argv[])
12 {
13 if(argc < 3)
14 {
15 fprintf(stderr, "Usage: %s <ip> <port>\n", argv[0]);
16 exit(1);
17 }
18
19 int sockfd; //文件描述符
20 struct sockaddr_in serveraddr; //服务器网络信息结构体
21 socklen_t addrlen = sizeof(serveraddr);
22
23 //第一步：创建套接字
24 if((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
25 {
26 perror("fail to socket");
27 exit(1);
28 }
29
30 //客户端自己指定自己的 ip 地址和端口号，一般不需要，系统会自动分配
31 #if 0
32 struct sockaddr_in clientaddr;
33 clientaddr.sin_family = AF_INET;
34 clientaddr.sin_addr.s_addr = inet_addr(argv[3]); //客户端的 ip 地址
35 clientaddr.sin_port = htons(atoi(argv[4])); //客户端的端口号
36 if(bind(sockfd, (struct sockaddr *)&clientaddr, addrlen) < 0)
37 {
38 perror("fail to bind");
39 exit(1);
40 }
41 #endif
42
43 //第二步：填充服务器网络信息结构体
44 //inet_addr：将点分十进制字符串 ip 地址转化为整形数据
45 //htons：将主机字节序转化为网络字节序
46 //atoi：将数字型字符串转化为整形数据
47 serveraddr.sin_family = AF_INET;
48 serveraddr.sin_addr.s_addr = inet_addr(argv[1]);
49 serveraddr.sin_port = htons(atoi(argv[2]));
50
51 //第三步：进行通信
52 char buf[32] = "";
53 while(1)
54 {
55 fgets(buf, sizeof(buf), stdin);
56 buf[strlen(buf) ‐ 1] = '\0';
57
58 if(sendto(sockfd, buf, sizeof(buf), 0, (struct sockaddr *)&serve
raddr, sizeof(serveraddr)) < 0)
59 {
60 perror("fail to sendto");
61 exit(1);
62 }
63
64 char text[32] = "";
65 if(recvfrom(sockfd, text, sizeof(text), 0, (struct sockaddr *)&s
erveraddr, &addrlen) < 0)
66 {
67 perror("fail to recvfrom");
68 exit(1);
69 }
70 printf("from server: %s\n", text);
71 }
72 //第四步：关闭文件描述符
73 close(sockfd);
74
75 return 0;
76 }
```

## UDP 服务器注意点

1、服务器之所以要 bind 是因为它的本地 port 需要是固定，而不是随机的
2、服务器也可以主动地给客户端发送数据
3、客户端也可以用 bind，这样客户端的本地端口就是固定的了，但一般不这样做

udp 是并发服务器，一个服务器可以对多个客户。

```c
//udp 服务器的实现
2 #include <stdio.h> //printf
3 #include <stdlib.h> //exit
4 #include <sys/types.h>
5 #include <sys/socket.h> //socket
6 #include <netinet/in.h> //sockaddr_in
7 #include <arpa/inet.h> //htons inet_addr
8 #include <unistd.h> //close
9 #include <string.h>
10
11 int main(int argc, char const *argv[])
12 {
13 if(argc < 3)
14 {
15 fprintf(stderr, "Usage: %s <ip> <port>\n", argv[0]);
16 exit(1);
17 }
18
19 int sockfd; //文件描述符
20 struct sockaddr_in serveraddr; //服务器网络信息结构体
21 socklen_t addrlen = sizeof(serveraddr);
22
23 //第一步：创建套接字
24 if((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
25 {
26 perror("fail to socket");
27 exit(1);
28 }
29
30 //第二步：填充服务器网络信息结构体
31 //inet_addr：将点分十进制字符串 ip 地址转化为整形数据
32 //htons：将主机字节序转化为网络字节序
33 //atoi：将数字型字符串转化为整形数据
34 serveraddr.sin_family = AF_INET;
35 serveraddr.sin_addr.s_addr = inet_addr(argv[1]);
36 serveraddr.sin_port = htons(atoi(argv[2]));
37
38 //第三步：将套接字与服务器网络信息结构体绑定
39 if(bind(sockfd, (struct sockaddr *)&serveraddr, addrlen) < 0)
40 {
41 perror("fail to bind");
42 exit(1);
43 }
44
45 while(1)
46 {
47 //第四步：进行通信
48 char text[32] = "";
49 struct sockaddr_in clientaddr;
50 if(recvfrom(sockfd, text, sizeof(text), 0, (struct sockaddr *)&c
lientaddr, &addrlen) < 0)
51 {
52 perror("fail to recvfrom");
53 exit(1);
54 }
55 printf("
[%s ‐ %d]: %s\n", inet_ntoa(clientaddr.sin_addr), ntohs(clientaddr.sin_port),
ext);
56
57 strcat(text, " *_*");
58
59 if(sendto(sockfd, text, sizeof(text), 0, (struct sockaddr *)&cli
entaddr, addrlen) < 0)
60 {
61 perror("fail to sendto");
62 exit(1);
63 }
64 }
65
66 //第四步：关闭文件描述符
67 close(sockfd);
68
69 return 0;
70 }
```

















