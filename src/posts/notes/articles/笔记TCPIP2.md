---
title: 嵌入式学习之（八）| 计算机网络 2
date: 2021-04-26
author: Kevin 吴嘉文
keywords: 
language: cn
category:
- 知识笔记
tag:
- Linux
- 嵌入式学习
mathjax: true
toc: true
comments: 
---

> 笔记总结 课程链接： 
> 续 [嵌入式学习之（八）| 计算机网络 1](http://wujiawen.xyz/2021/04/02/TCPIP1/)

<!--more-->

基于 UDP 的应用层协议，最初用于引导无盘系统，被设计用来传输小文件，不进行用户有效性认证

![image-20210405173235051](/assets/img/TCPIP2/image-20210405173235051.png)

1、服务器在 69 号端口等待客户端的请求
2、服务器若批准此请求,则使用临时端口与客户端进行通信
3、每个数据包的编号都有变化（从 0 开始）
4、每个数据包都要得到 ACK 的确认如果出现超时,则需要重新发送最后的包（数据或 ACK）
5、数据的长度以 512Byte 传输
6、小于 512Byte 的数据意味着传输结束

![image-20210405173730450](/assets/img/TCPIP2/image-20210405173730450.png)

 **注意：** 

下载 01，读取 02， 模式：octet 二进制；快编号用于核对文件块。

以上的 0 代表的是'\0'
不同的差错码对应不同的错误信息

 **错误码：** 
未定义,参见错误信息
File not found.
Access violation.
Disk full or allocation exceeded.
illegal TFTP operation.
Unknown transfer ID.
File already exists.
No such user.
Unsupported option(s) requested.

 **实现思路** 
1、构造请求报文，送至服务器(69 号端口)
2、等待服务器回应
3、分析服务器回应
4、接收数据,直到接收到的数据包小于规定数据长度

```c
#include <stdio.h> //printf
#include <stdlib.h> //exit
#include <sys/types.h>
#include <sys/socket.h> //socket
#include <netinet/in.h> //sockaddr_in
#include <arpa/inet.h> //htons inet_addr
#include <unistd.h> //close
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>

void do_download(int sockfd, struct sockaddr_in serveraddr)
{
char filename[128] = "";
printf("请输入要下载的文件名: ");
scanf("%s", filename);

//给服务器发送消息，告知服务器执行下载操作
unsigned char text[1024] = "";
int text_len;
socklen_t addrlen = sizeof(struct sockaddr_in);
int fd;
int flags = 0;
int num = 0;
 ssize_t bytes;

//构建给服务器发送的 tftp 指令并发送给服务器，例如：01test.txt0octet0
text_len = sprintf(text, "%c%c%s%c%s%c", 0, 1, filename, 0, "octet", 0
if(sendto(sockfd, text, text_len, 0, (struct sockaddr *)&serveraddr, addrlen) < 0)
{
perror("fail to sendto");
exit(1);
}

while(1)
{
//接收服务器发送过来的数据并处理
if((bytes = recvfrom(sockfd, text, sizeof(text), 0, (struct sockaddr*)&serveraddr, &addrlen)) < 0)
{
perror("fail to recvfrom");
exit(1);
}
//printf("操作码：%d, 块编号：%u\n", text[1], ntohs(*(unsigned short *)(text+2)));
//printf("数据：%s\n", text+4);

//判断操作码执行相应的处理
if(text[1] == 5)
{
    printf("error: %s\n", text+4);
    return ;
}
else if(text[1] == 3)
{
	if(flags == 0)
    {
    //创建文件
    if((fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0664)) < 0)
    {
        perror("fail to open");
        exit(1);
    }
	flags = 1;
}
    
//对比编号和接收的数据大小并将文件内容写入文件
if((num+1 == ntohs(*(unsigned short *)(text+2))) && (bytes == 516))
{
num = ntohs(*(unsigned short *)(text+2));
if(write(fd, text + 4, bytes ‐ 4) < 0)
{
perror("fail to write");
exit(1);
}

//当文件写入完毕后，给服务器发送 ACK
text[1] = 4;
if(sendto(sockfd, text, 4, 0, (struct sockaddr *)&serveraddr, addrlen) < 0)
{
perror("fail to sendto");
exit(1);
}
}
//当最后一个数据接收完毕后，写入文件后退出函数
else if((num+1 == ntohs(*(unsigned short *)(text+2))) && (bytes < 516))
{
if(write(fd, text + 4, bytes ‐ 4) < 0)
{
perror("fail to write");
exit(1);
}

text[1] = 4;
if(sendto(sockfd, text, 4, 0, (struct sockaddr *)&serveraddr, addrlen) < 0)
{
perror("fail to sendto");
exit(1);
}

printf("文件下载完毕\n");
return ;
}
}
}
}

int main(int argc, char const *argv[])
{
if(argc < 2)
{
fprintf(stderr, "Usage: %s <server_ip>\n", argv[0]);
exit(1);
}

int sockfd;
struct sockaddr_in serveraddr;

//创建套接字
if((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
{
perror("fail to socket");
exit(1);
}

//填充服务器网络信息结构体
serveraddr.sin_family = AF_INET;
serveraddr.sin_addr.s_addr = inet_addr(argv[1]); //tftp 服务器端的 ip 地址，192.168.3.78
serveraddr.sin_port = htons(69); //tftp 服务器的端口号默认是 69
do_download(sockfd, serveraddr); //下载操作

return 0;
}
```

## UDP 广播

广播：由一台主机向该主机所在子网内的所有主机发送数据的方式，
例如 192.168.3.103 主机发送广播信息，则 192.168.3.1~192.168.3.254 所有主机都可以接收到数据
广播只能用 UDP 或原始 IP 实现，不能用 TCP

 **广播的用途** 
单个服务器与多个客户主机通信时减少分组流通
以下几个协议都用到广播
1、地址解析协议（ARP）
2、动态主机配置协议（DHCP）
3、网络时间协议（NTP）

 **广播的特点** 
1、处于同一子网的所有主机都必须处理数据
2、UDP 数据包会沿协议栈向上一直到 UDP 层
3、运行音视频等较高速率工作的应用，会带来大负
4、局限于局域网内使用

 **广播地址** 
{网络 ID，主机 ID}
网络 ID 表示由子网掩码中 1 覆盖的连续位
主机 ID 表示由子网掩码中 0 覆盖的连续位
 **定向广播地址：** 主机 ID 全 1
1、例：对于 192.168.220.0/24，其定向广播地址为 192.168.220.255
2、通常路由器不转发该广播
 **受限广播地址：** 255.255.255.255
路由器从不转发该广播

 **单播**  

![image-20210405180158217](/assets/img/TCPIP2/image-20210405180158217.png)

广播 IP 特殊，MAC 地址为广播所对应的目的地址 ff:ff:ff:ff:ff:ff. 广播的 IP, MAC 可以被当前网段下所有的主机接受

 **4.3 广播流程** 
 **发送者：** 
第一步：创建套接字 socket()
第二步：设置为允许发送广播权限 setsockopt()
第三步：向广播地址发送数据 sendto()
 **接收者：** 
第一步：创建套接字 socket()
第二步：将套接字与广播的信息结构体绑定 bind()
第三步：接收数据 recvfrom()

 **广播示例** 

发送者

```c
#include <stdio.h> //printf
#include <stdlib.h> //exit
#include <sys/types.h>
#include <sys/socket.h> //socket
#include <netinet/in.h> //sockaddr_in
#include <arpa/inet.h> //htons inet_addr
#include <unistd.h> //close
#include <string.h>
int main(int argc, char const *argv[])
{
if(argc < 3)
{
fprintf(stderr, "Usage: %s <ip> <port>\n", argv[0]);
exit(1);
}
int sockfd; //文件描述符
struct sockaddr_in broadcataddr; //服务器网络信息结构体
socklen_t addrlen = sizeof(broadcataddr);
//第一步：创建套接字
if((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
{
perror("fail to socket");
exit(1);
}
//第二步：设置为允许发送广播权限
int on = 1;
if(setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on)) < 0)
{
perror("fail to setsockopt");
exit(1);
}
//第三步：填充广播信息结构体
broadcataddr.sin_family = AF_INET;
broadcataddr.sin_addr.s_addr = inet_addr(argv[1]); //192.168.3.255 2
55.255.255.255
broadcataddr.sin_port = htons(atoi(argv[2]));
//第四步：进行通信
char buf[128] = "";
while(1)
{
fgets(buf, sizeof(buf), stdin);
buf[strlen(buf) ‐ 1] = '\0';
if(sendto(sockfd, buf, sizeof(buf), 0, (struct sockaddr *)&broadcataddr, addrlen) < 0)
{
perror("fail to sendto");
exit(1);
}
}
return 0;
}
```

接收者

```c
#include <stdio.h> //printf
#include <stdlib.h> //exit
#include <sys/types.h>
#include <sys/socket.h> //socket
#include <netinet/in.h> //sockaddr_in
#include <arpa/inet.h> //htons inet_addr
#include <unistd.h> //close
#include <string.h>
int main(int argc, char const *argv[])
{
if(argc < 3)
{
fprintf(stderr, "Usage: %s <ip> <port>\n", argv[0]);
exit(1);
}
int sockfd; //文件描述符
struct sockaddr_in broadcataddr;
socklen_t addrlen = sizeof(broadcataddr);
//第一步：创建套接字
if((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
{
perror("fail to socket");
exit(1);
}
//第二步：填充广播信息结构体
broadcataddr.sin_family = AF_INET;
broadcataddr.sin_addr.s_addr = inet_addr(argv[1]); //192.168.3.255 255.255.255.255
broadcataddr.sin_port = htons(atoi(argv[2]));
//第三步：将套接字与广播信息结构体绑定
if(bind(sockfd, (struct sockaddr *)&broadcataddr, addrlen) < 0)
{
perror("fail to bind");
exit(1);
}
//第四步：进行通信
char text[32] = "";
struct sockaddr_in sendaddr;
while(1)
{
if(recvfrom(sockfd, text, sizeof(text), 0, (struct sockaddr *)&sendaddr, &addrlen) < 0)
{
perror("fail to recvfrom");
exit(1);
}
printf("[%s ‐ %d]: %s\n", inet_ntoa(sendaddr.sin_addr), ntohs(se
ndaddr.sin_port), text);
}
return 0;
}
```

## UDP 多播

数据的收发仅仅在同一分组中进行，所以多播又称之为组播
 **多播的特点：** 
1、多播地址标示一组接口
2、多播可以用于广域网使用
3、在 IPv4 中，多播是可选的

 **多播地址** 
IPv4 的 D 类地址是多播地址
十进制：224.0.0.1~239.255.255.254
十六进制：E0.00.00.01~EF.FF.FF.FE
多播地址向以太网 MAC 地址的映射

![相关图片](/assets/img/TCPIP2/image-20210405192704218.png )

 **多播工作过程** 

![image-20210405192533377](/assets/img/TCPIP2/image-20210405192533377.png

只有加入多播组的用户才可以接受数据。

 **多播流程** 
 **发送者：** 
第一步：创建套接字 socket()
第二步：向多播地址发送数据 sendto()
 **接收者：** 
第一步：创建套接字 socket()
第二步：设置为加入多播组 setsockopt()
第三步：将套接字与多播信息结构体绑定 bind()
第五步：接收数据

 **多播地址结构体** 

![相关图片](/assets/img/TCPIP2/image-20210405193207063.png )

 **示例**  

## TCP 网络编程

1、面向连接的流式协议;可靠、出错重传、且每收到一个数据都要给出相应的确认
2、通信之前需要建立链接
3、服务器被动链接，客户端是主动链接

### 流程对比

![image-20210405193605119](/assets/img/TCPIP2/image-20210405193605119.png)

 **服务器：** 
创建套接字 socket()
将套接字与服务器网络信息结构体绑定 bind()
将套接字设置为监听状态 listen()
阻塞等待客户端的连接请求 accept()
进行通信 recv()/send()
关闭套接字 close()
 **客户端：** 
创建套接字 socket()
发送客户端连接请求 connect()
进行通信 send()/recv()
关闭套接字 close()

### TCP 编程

#### socket()

```c
#include <sys/types.h>
#include <sys/socket.h>
int socket(int domain, int type, int protocol);
```

> 功能：创建一个套接字，返回一个文件描述符
> 参数：
> domain：通信域，协议族
> AF_UNIX 本地通信
> AF_INET ipv4 网络协议
> AF_INET6 ipv6 网络协议
> AF_PACKET 底层接口
> type：套接字的类型
> SOCK_STREAM 流式套接字（tcp）
> SOCK_DGRAM 数据报套接字（udp）
> SOCK_RAW 原始套接字（用于链路层）
> protocol：附加协议，如果不需要，则设置为 0
> 返回值：
> 成功：文件描述符
> 失败：‐1

```c
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdlib.h>
int main(int argc, char const *argv[])
{
//通过 socket 函数创建一个 TCP 套接字
int sockfd;
if((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == ‐1)
{
perror("fail to socket");
exit(1);
}
printf("sockfd = %d\n", sockfd);
return 0;
}
```

#### connect()

```c
#include <sys/types.h> /* See NOTES */
#include <sys/socket.h>
int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
```

> 功能：给服务器发送客户端的连接请求
> 参数：
> sockfd：文件描述符，socket 函数的返回值
> addr：要连接的服务器的网络信息结构体（需要自己设置）
> addrlen：add 的长度
> 返回值：
> 成功：0
> 失败：‐1

 **注意：** 
1、connect 建立连接之后不会产生新的套接字
2、连接成功后才可以开始传输 TCP 数据
3、头文件：`#include <sys/socket.h>`

#### send()

```c
#include <sys/types.h>
#include <sys/socket.h>
ssize_t send(int sockfd, const void *buf, size_t len, int flags);
```

> 功能：发送数据
> 参数：
> sockfd：文件描述符
> 客户端：socket 函数的返回值
> 服务器：accept 函数的返回值
> buf：发送的数据
> len：buf 的长度
> flags：标志位
> 0 阻塞
> MSG_DONTWAIT 非阻塞
> 返回值：
> 成功：发送的字节数
> 失败：‐1

 **注意：** 
不能用 TCP 协议发送 0 长度的数据包

#### recv()

```c
#include <sys/types.h>
#include <sys/socket.h>
ssize_t recv(int sockfd, void *buf, size_t len, int flags);
```

> 功能：接收数据
> 参数：
> sockfd：文件描述符
> 客户端：socket 函数的返回值
> 服务器：accept 函数的返回值
> buf：保存接收到的数据
> len：buf 的长度
> flags：标志位
> 0 阻塞
> MSG_DONTWAIT 非阻塞
> 返回值：
> 成功：接收的字节数
> 失败：‐1
> 如果发送端关闭文件描述符或者关闭进程，则 recv 函数会返回 0

#### 示例

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>
#define N 128
int main(int argc, char const *argv[])
{
if(argc < 3)
{
fprintf(stderr, "Usage: %s [ip] [port]\n", argv[0]); //客户端使用的 ip 和端口应该是服务器的。
exit(1);
}
//第一步：创建套接字
int sockfd;
if((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == ‐1)
{
perror("fail to socket");
exit(1);
}
//printf("sockfd = %d\n", sockfd);
//第二步：发送客户端连接请求
struct sockaddr_in serveraddr;
socklen_t addrlen = sizeof(serveraddr);
serveraddr.sin_family = AF_INET;
serveraddr.sin_addr.s_addr = inet_addr(argv[1]);
serveraddr.sin_port = htons(atoi(argv[2]));
if(connect(sockfd, (struct sockaddr *)&serveraddr, addrlen) == ‐1)
{
perror("fail to connect");
exit(1);
}
//第三步：进行通信
//发送数据
char buf[N] = "";
fgets(buf, N, stdin);
buf[strlen(buf) ‐ 1] = '\0';
if(send(sockfd, buf, N, 0) == ‐1)
{
perror("fail to send");
exit(1);
}
//接收数据
char text[N] = "";
if(recv(sockfd, text, N, 0) == ‐1)
{
perror("fail to recv");
exit(1);
}
printf("from server: %s\n", text);
//第四步：关闭套接字文件描述符
close(sockfd);
return 0;
} 
```

### TCP 服务器

做为 TCP 服务器需要具备的条件
1、具备一个可以确知的地址
2、让操作系统知道是一个服务器，而不是客户端
3、等待连接的到来
对于面向连接的 TCP 协议来说，连接的建立才真正意味着数据通信的开始

### bind()

```c
#include <sys/types.h>
#include <sys/socket.h>
int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
```

> 功能：将套接字与网络信息结构体绑定
> 参数：
> sockfd：文件描述符，socket 的返回值
> addr：网络信息结构体
> 通用结构体（一般不用）
> struct sockaddr
> 网络信息结构体 sockaddr_in
> #include <netinet/in.h>
> struct sockaddr_in
> addrlen：addr 的长度
> 返回值：
> 成功：0
> 失败：‐1

#### listen()

```c
#include <sys/types.h> /* See NOTES */
#include <sys/socket.h>
int listen(int sockfd, int backlog);
```



> 功能：将套接字设置为被动监听状态，这样做之后就可以接收到连接请求
> 参数：
> sockfd：文件描述符，socket 函数返回值
> backlog：允许通信连接的主机个数，一般设置为 5、10
> 返回值：
> 成功：0
> 失败：‐1

#### accept()

```c
#include <sys/types.h> /* See NOTES */
#include <sys/socket.h>
int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
```



> 功能：阻塞等待客户端的连接请求
> 参数：
> sockfd：文件描述符，socket 函数的返回值
> addr：接收到的客户端的信息结构体（自动填充，定义变量即可）
> addrlen：addr 的长度
> 返回值：
> 成功：新的文件描述符（只要有客户端连接，就会产生新的文件描述符，
> 这个新的文件描述符专门与指定的客户端进行通信的）
> 失败：‐1

#### 例子

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>
#define N 128
int main(int argc, char const *argv[])
{
if(argc < 3)
{
fprintf(stderr, "Usage: %s [ip] [port]\n", argv[0]);
exit(1);
}
//第一步：创建套接字
int sockfd;
if((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == ‐1)
{
perror("fail to socket");
exit(1);
}
//第二步：将套接字与服务器网络信息结构体绑定
struct sockaddr_in serveraddr;
socklen_t addrlen = sizeof(serveraddr);
serveraddr.sin_family = AF_INET;
serveraddr.sin_addr.s_addr = inet_addr(argv[1]);
serveraddr.sin_port = htons(atoi(argv[2]));
if(bind(sockfd, (struct sockaddr *)&serveraddr, addrlen) == ‐1)
{
perror("fail to bind");
exit(1);
}
//第三步：将套接字设置为被动监听状态
if(listen(sockfd, 10) == ‐1)
{
perror("fail to listen");
exit(1);
}
//第四步：阻塞等待客户端的链接请求
int acceptfd;
struct sockaddr_in clientaddr;
if((acceptfd = accept(sockfd, (struct sockaddr *)&clientaddr, &addrl
en)) == ‐1)
{
perror("fail to accept");
exit(1);
}
//打印连接的客户端的信息
printf("ip:%s, port:%d\n", inet_ntoa(clientaddr.sin_addr), ntohs(cli
entaddr.sin_port));
//第五步：进行通信
//tcp 服务器与客户端通信时，需要使用 accept 函数的返回值
char buf[N] = "";
if(recv(acceptfd, buf, N, 0) == ‐1)
{
perror("fail to recv");
}
printf("from client: %s\n", buf);
strcat(buf, " *_*");
if(send(acceptfd, buf, N, 0) == ‐1)
{
perror("fail to send");
exit(1);
}
//关闭套接字文件描述符
close(acceptfd);
close(sockfd);
return 0;
}
```

### TCP 编程 - close、三次握手、四次挥手

服务器对每个客户端都会创建一个独立的文件描述符，再客户端和服务器通信时，不能在使用 sock 返回值，需要使用 accept 返回值

 **close 关闭套接字** 
1、使用 close 函数即可关闭套接字
关闭一个代表已连接套接字将导致另一端接收到一个 0 长度的数据包
2、做服务器时
1>关闭监听套接字将导致服务器无法接收新的连接，但不会影响已经建立的连接
2>关闭 accept 返回的已连接套接字将导致它所代表的连接被关闭，但不会影响服务器的监听
3、做客户端时
关闭连接就是关闭连接，不意味着其他

 **三次握手**  

内部完成，编程时没有看到。[详细链接](https://www.cnblogs.com/bj-mr-li/p/11106390.html)

![相关图片](/assets/img/TCPIP2/image-20210406150850175.png =x300)

第一次握手：建立连接时，客户端发送 syn 包（syn=j）到服务器，并进入 SYN_SENT 状态，等待服务器确认；SYN：同步序列编号（Synchronize Sequence Numbers）。

第二次握手：服务器收到 syn 包，必须确认客户的 SYN（ack=j+1），同时自己也发送一个 SYN 包（syn=k），即 SYN+ACK 包，此时服务器进入 SYN_RECV 状态；

第三次握手：客户端收到服务器的 SYN+ACK 包，向服务器发送确认包 ACK(ack=k+1），此包发送完毕，客户端和服务器进入 ESTABLISHED（TCP 连接成功）状态，完成三次握手。

#### 四次挥手

![相关图片](/assets/img/TCPIP2/image-20210406151221826.png =x300)

1）客户端进程发出连接释放报文，并且停止发送数据。释放数据报文首部，FIN=1，其序列号为 seq=u（等于前面已经传送过来的数据的最后一个字节的序号加 1），此时，客户端进入 FIN-WAIT-1（终止等待 1）状态。 TCP 规定，FIN 报文段即使不携带数据，也要消耗一个序号。
2）服务器收到连接释放报文，发出确认报文，ACK=1，ack=u+1，并且带上自己的序列号 seq=v，此时，服务端就进入了 CLOSE-WAIT（关闭等待）状态。TCP 服务器通知高层的应用进程，客户端向服务器的方向就释放了，这时候处于半关闭状态，即客户端已经没有数据要发送了，但是服务器若发送数据，客户端依然要接受。这个状态还要持续一段时间，也就是整个 CLOSE-WAIT 状态持续的时间。
3）客户端收到服务器的确认请求后，此时，客户端就进入 FIN-WAIT-2（终止等待 2）状态，等待服务器发送连接释放报文（在这之前还需要接受服务器发送的最后的数据）。
4）服务器将最后的数据发送完毕后，就向客户端发送连接释放报文，FIN=1，ack=u+1，由于在半关闭状态，服务器很可能又发送了一些数据，假定此时的序列号为 seq=w，此时，服务器就进入了 LAST-ACK（最后确认）状态，等待客户端的确认。
5）客户端收到服务器的连接释放报文后，必须发出确认，ACK=1，ack=w+1，而自己的序列号是 seq=u+1，此时，客户端就进入了 TIME-WAIT（时间等待）状态。注意此时 TCP 连接还没有释放，必须经过 2∗∗MSL（最长报文段寿命）的时间后，当客户端撤销相应的 TCB 后，才进入 CLOSED 状态。
6）服务器只要收到了客户端发出的确认，立即进入 CLOSED 状态。同样，撤销 TCB 后，就结束了这次的 TCP 连接。可以看到，服务器结束 TCP 连接的时间要比客户端早一些。

### TCP 并发服务器

TCP 原本不是并发服务器，TCP 服务器同一时间只能与一个客户端通信
原始代码：

TCP 不能实现并发的原因：
由于 TCP 服务器端有两个读阻塞函数，accept 和 recv，两个函数需要先后运行，所以导致运
行一个函数的时候另一个函数无法执行，所以无法保证一边连接客户端，一边与其他客户端通信

如何实现 TCP 并发服务器：
使用多进程实现 TCP 并发服务器
使用多线程实现 TCP 并发服务器

#### 多进程实现并发

> int sockfd = socket()
> bind()
> listen()
> while(1)
> {
> acceptfd = accept()
> pid = fork();
> if(pid > 0)
> {
> }
> else if(pid == 0)
> {
> while(1)
> {
> recv()/send()
> }}}

 **案例** 

#### 多线程实现并发

> void *thread_fun(void *arg)
> {
> while(1)
> {
> recv() / send()
> }
> }
> sockfd = socket()
> bind()
> listen()
> while(1)
> {
> accept()
> //只要有客户端连接上，则创建一个子线程与之通信
> pthread_create(, , thread_fun, );
> pthread_detach();
> }

案例

## Web 服务器

web 服务器简介

Web 服务器又称 WWW 服务器、网站服务器等
 **特点** 
使用 HTTP 协议与客户机浏览器进行信息交流
不仅能存储信息，还能在用户通过 web 浏览器提供的信息的基础上运行脚本和程序
该服务器需可安装在 UNIX、Linux 或 Windows 等操作系统上
著名的服务器有 Apache、Tomcat、 IIS 等

### HTTP

Webserver—HTTP 协议（超文本协议）
 **概念** 
一种详细规定了浏览器和万维网服务器之间互相通信的规则，通过因特网传送万维网文档的
数据传送协议
 **特点** 
1、支持 C/S 架构
2、简单快速：客户向服务器请求服务时，只需传送请求方法和路径 ，常用方法:GET（明文）、POST（密文）
3、无连接：限制每次连接只处理一个请求
4、无状态：即如果后续处理需要前面的信息，它必须重传，这样可能导致每次连接传送的数据量会增大

### Webserver 通信过程

![image-20210406215546269](/assets/img/TCPIP2/image-20210406215546269.png)

web 服务器的 ip 地址是 192.168.3.103，端口号是 9999，要访问的网页是 about.html

浏览器输入的格式为：192.168.3.103:9999/about.html

 **服务器应答的格式** ：
服务器接收到浏览器发送的数据之后，需要判断 GET/后面跟的网页是否存在，如果存在则请求成功，发送指定的指令，并发送文件内容给浏览器，如果不存在，则发送请求失败的指令

 **案例** 

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>
#include <pthread.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#define N 1024
#define ERR_LOG(errmsg) do{\
perror(errmsg);\
printf("%s ‐ %s ‐ %d\n", __FILE__, __func__, __LINE__);\
exit(1);\
}while(0)

void *pthread_fun(void *arg)
{
int acceptfd = *(int *)arg;
char buf[N] = ""; char head[]="HTTP/1.1 200 OK\r\n" \
"Content‐Type: text/html\r\n" \
"\r\n";
char err[]= "HTTP/1.1 404 Not Found\r\n" \ 
    "Content‐Type: text/html\r\n" \
"\r\n" \
"<HTML><BODY>File not found</BODY></HTML>";

//接收浏览器通过 http 协议发送的数据包
if(recv(acceptfd, buf, N, 0) < 0)
{
ERR_LOG("fail to recv");
}

printf("*****************************\n\n");
printf("%s\n", buf);
// int i;
// for(i = 0; i < 200; i++)
// {
// printf("[%c] ‐ %d\n", buf[i], buf[i]);
// }
printf("\n*****************************\n");

//通过获取的数据包中得到浏览器要访问的网页文件名
//GET /about.html http/1.1
char filename[128] = "";
sscanf(buf, "GET /%s", filename); //sscanf 函数与空格结束，所以直接可以获取文件名

if(strncmp(filename, "HTTP/1.1", strlen("http/1.1")) == 0)
{
strcpy(filename, "about.html");
}
printf("filename = %s\n", filename);

char path[128] = "./sqlite/";
strcat(path, filename);

//通过解析出来的网页文件名，查找本地中有没有这个文件
int fd;
if((fd = open(path, O_RDONLY)) < 0)
{
//如果文件不存在，则发送不存在对应的指令
if(errno == ENOENT)
{
if(send(acceptfd, err, strlen(err), 0) < 0)
{
ERR_LOG("fail to send");
}

close(acceptfd);
pthread_exit(NULL);
}
else
{
ERR_LOG("fail to open");
} }

//如果文件存在，先发送指令告知浏览器
if(send(acceptfd, head, strlen(head), 0) < 0)
{
ERR_LOG("fail to send");
}

//读取网页文件中的内容并发送给浏览器
ssize_t bytes;
char text[1024] = "";
while((bytes = read(fd, text, 1024)) > 0)
{
if(send(acceptfd, text, bytes, 0) < 0)
{
ERR_LOG("fail to send");
}
}

pthread_exit(NULL);
}

int main(int argc, char const *argv[])
{
if(argc < 3)
{
fprintf(stderr, "Usage: %s <server_ip> <server_port>\n", argv[0])
exit(1);
}

int sockfd, acceptfd;
struct sockaddr_in serveraddr, clientaddr;
socklen_t addrlen = sizeof(serveraddr);

//第一步：创建套接字
if((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
{
ERR_LOG("fail to socket");
}

//将套接字设置为允许重复使用本机地址或者为设置为端口复用
int on = 1;
if(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0)
{
ERR_LOG("fail to setsockopt");
}

//第二步：填充服务器网络信息结构体
serveraddr.sin_family = AF_INET;
serveraddr.sin_addr.s_addr = inet_addr(argv[1]);
serveraddr.sin_port = htons(atoi(argv[2]));

//第三步：将套接字与服务器网络信息结构体绑定
if(bind(sockfd, (struct sockaddr *)&serveraddr, addrlen) < 0)
{
ERR_LOG("fail to bind");
}

//第四步：将套接字设置为被动监听状态
if(listen(sockfd, 5) < 0)
{
ERR_LOG("fail to listen");
}

while(1)
{
//第五步：阻塞等待客户端的连接请求
if((acceptfd = accept(sockfd, (struct sockaddr *)&clientaddr, &addrlen)) < 0)
{
ERR_LOG("fail to accept");
}

//打印客户端的信息
printf("%s ‐‐ %d\n", inet_ntoa(clientaddr.sin_addr), ntohs(clientaddr.sin_port));

//创建线程接收数据并处理数据
pthread_t thread;
if(pthread_create(&thread, NULL, pthread_fun, &acceptfd) != 0)
{
ERR_LOG("fail to pthread_create");
}
pthread_detach(thread);

}

return 0;
}
```





