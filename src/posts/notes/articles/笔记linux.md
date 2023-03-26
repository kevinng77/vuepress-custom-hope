---
title: 嵌入式学习之（五）|Linux 命令
date: 2020-07-03
author: Kevin 吴嘉文
keywords: 
language: cn
category:
- 知识笔记
tag:
- 计算机语言
- 嵌入式学习
mathjax: true
toc: true
comments: 

---

# Linux 命令笔记

> 嵌入式自学开始一个月啦~ 为了自己的贾维斯！！
>
> Linux 笔记比较少，入门级笔记。用多了就记住了。
>
> 笔记总结 课程链接：[千峰嵌入式教程](https://www.bilibili.com/video/BV1FA411v7YW?p=530&spm_id_from=pageDriver)

<!-- more-->

## 命令

+ ctrl + shift + "+"
+ user@主机名: + [ ~家目录；/根目录|] $ 普通用户权限，#管理员权限
+ 命令 选项 参数，之间用空格隔开
+ 一次 tab 补全，两次 tab 查看全部相关命令


```shell
命令 + --helpls
ls, more, man,cat, head, tail
cat 
```

```shell
man 章节 查找命令，reshuffle 文件信息
可以用 man man 查询 man 的信息
```

```shell
命令 > file 先清空，后输入
命令 >> file 追加
ls --help | more more(ls --help)
ls | wc -w
```

 **ls** 

```shell
ls / -a -l -al -hl
```

+ drwxr-xr-x 2 youtiaowenzi youtiaowenzi 4096 Sep 28 21:43 Downloads
+ d: 文件类型(bcd-lsp)
  + b：块设备 c：字符设备 d 目录文件 -普通文件 l 软连接文件 s 套接字 p 管道文件
+ rwxr-xr-x：文件权限，三个为一组
  + r w x -读，写，可执行，无权限
+ 2： 链接文件个数
+ 用户名 用户组名
+ 4096 文件大小（bytes）
+ 文件最后修改时间

+ tree (apt-get install tree) 
  
  + tree -L 3  只显示 3 层
+  **clear**  清屏 或 ctrl + L
+  **cd pwd** 
  + cd / 根目录 直接 cd 或 ~进入 home
  + cd .. 上级目录 . 当前目录 - 返回上一次的路径
  + pwd 当前绝对路径

+  **cat**  显示文本文件内容
+  **rm -rf**  删除目录
+  **cp file_name dir**  复制  **cp dir1 dir2 -a**  复制 dir1 到 dir2 ，  **cp file file2**  复制 file 副本 file2，覆盖原先 file2
+  **mv file dir/**  移动文件  mv file1 file2 如果 file2 不存在，则为重命名
+  **mkdir/ touch** 
  
  + mkdir -p dir1/dir2/dir3/dir4 创建嵌套文件
+ touch 会更新更新时间戳
+  **find path -name filename** 
  
+ find . - name dir2
  
+  **grep**  查找信息 file 参数-n (-n 返回行号)
  
  + grep 信息 * -R -n 对所有文件*查询，-R 对嵌套的文件进行查询
+  **ln**  创建链接文件
  
  + ln 源文件 链接文件名 -s 

## tar 

+ tar zcvf 压缩包.tar.gz file1 file2 file3 压缩
+ tar zxvf 压缩包.tar.gz 解压
+ tar zxvf 压缩包.tar.gz  -C 目的路径
+ tar jcvf name.tar.bz2
+ tar jxvf name.tar.bz2 -C 目的路径

## vi

+ vi +n filename 打开文件，光标置于 n 行行首
+ 命令模式或插入模式 进入 编辑模式 ESC
+ 搜索，命令模式下输入/加上要搜索内容
+ 编辑 进入 插入 按 a i o；进入命令模式按：
  + 命令 w，wq,x 保存并推出, q! 不保存，w filename 另存为
    + :set nu 设置显示行号
  + 编辑模式
    + u 撤销
    + nx 删除光标后 n 字符
    + nX 删除光标前
    + ndd 剪切 p 粘贴 4yy 复制当前开始的 n 行
    + shift + zz 保存并退出
    + `n<space>` 移动光标到 n 字节后
    + n Enter 向下移动光标 n 行
    + nG 移动光标到 n 行
    + /字符串 从光标开始向后查询
      + 显示搜索结果的：按 n 下一个 N 前一个
    + ctrl + b/f 上/下一页
    + dd 删除光标所在行
    + d1G 删除光标所在到第一行的所有数据
    + dG 删除光标到最后一行所有数据
    + d$ 删除光标所在，到本行最后一个字符位置

## 编译器 gcc

+ 一步到位
  + gcc hello.c  自动生成 a.out 可执行文件
  + gcc hello.c -o hello 生成 hello 可执行文件
  + 运行： ./a.out
+ 分布
  + gcc -E hello.c -o hello.i 预处理 （头文件展开）
  + gcc -S hello.i -o hello.s 编译 
  + gcc -c hello.s -o hello.o 汇编
  + gcc hello.o -o filename 链接

## makefile

+ 只会重新编译修改过时间戳的文件


```shell
目标（最终生成文件）：依赖文件列表
`<TAB>`命令列表
makefile:
main:main.c main.h
	gcc main.c -o main
clean:
	rm main
```

make

```shell
[-f file] 默认找 GNUmakefile，makefile, Makefile 作为输入文件
[targets] 默认实现 makefile 文件内第一个目标

一般使用直接 make
make [-f file] [targets]
```

多文件编译

```shell
main:main.o sub.o sum.o
	gcc main.o sub.o sum.o -o main
main.o:main.c
	gcc -c main.c -o main.o
sub.o:sub.c
	gcc -c sub.c -o sub.o
sum.o:sum.c
	gcc -c sum.c -o sum.o
clean:
	rm *.o main a.out -rf
	
make
make clean
```

makefile 变量

```shell
cc = gcc
#cc = arm-linux-gcc 指定编译器
obj = main.o printf1.0
target=main
cflags=-Wall -g

$(target):$(obj)
	$(cc) $(obj) -o $(obj)  #用变量替换上一步的程序
```

系统环境变量

```shell
#export test=10 定义环境变量
```

预定义变量

```shell
$@ 目标名
$<	依赖文件中的第一个文件
$^ 	所有依赖文件
CC	C 编译器名称
CFLAGS		C 编译器选项


CC = gcc
obj = main
obj1 = sub
obj2 = sum
OBJ = main.o sub.o sum.o
CFLAGS = -Wall -g

$(obj):$(OBJ)
	$(CC) $^ -o $@
$(obj).o:$(obj).c
	$(CC) $(CFLAGS) -c $< -o $@
$(obj1).o:$(obj1).c
	$(CC) $(CFLAGS) -c $< -o $@
$(obj2).o:$(obj2).c
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	rm *.o $(obj) a.out -rf
```

进阶版

```shell
CC = gcc
obj = main
OBJ = main.o sub.o sum.o
CFLAGS = -Wall -g

$(obj):$(OBJ)
	$(CC) $^ -o $@
$*.o:$*.c
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	rm *.o $(obj) a.out -rf
```

#### 链接

硬链接：B 是 A 的硬链接，A 删了，B 还可以访问（类似创建一个新指针）。允许一个文件有多个路径。可以通过硬链接建立多个链接，防止文件误删。`ln A B`

软连接：类似 windows 的快捷方式。`ln -s A C` 软连接 `ls -l` 查看后为 `l` 前缀

`echo "写入文件的内容">>A` 

## 用户

`useradd -m 用户名`，-m 自动创建用户目录 -g 给用户分给组。用户信息被写入 `/etc/passwd`

`userdel -r 用户名` 删除用户与用户目录  

`su 用户名` 切换用户

`passwd 用户名` 设置用户密码

`passwd -l 用户` 锁定用户，用户无法登录

## 用户组

`groupadd groupname` 创建用户组，`-g` 指定用户组 ID，`/etc/group` 下可查看

`groupdel name` 删除用户组

`groupmod name` 修改组信息，`-g` 改 ID，`-n`改名字

## 磁盘管理

`df` 整体磁盘使用量

`du -h --max_depth=1` 查看当前文件夹容量

`mount /dev/devicename /mnt/dirname` 将外部设备挂在到目的目录下

`umount -f` 强制卸载 

## 进程

`ps` 

-a：显示终端运行的进程信息

-u：以用户的信息显示进程

-x：显示用户运行进程的参数

`ps -ef` 可以查看到父进程的信息

`pstree` 显示进程树 -p 显示父 ID -u 显示用户组

## 安装

yum 在线安装

## 系统环境变量

[Linux 下的环境变量](https://blog.csdn.net/jiangyanting2011/article/details/78875928)

 **系统级：** 

/etc/environment: 是系统在登录时读取的第一个文件，用于为所有进程设置环境变量。系统使用此文件时并不是执行此文件中的命令，而是根据 KEY=VALUE 模式的代码，对 KEY 赋值以 VALUE，因此文件中如果要定义 PATH 环境变量，只需加入一行形如 PATH=$PATH:/xxx/bin 的代码即可。

/etc/profile：是系统登录时执行的第二个文件，可以用于设定针对全系统所有用户的环境变量。该文件一般是调用/etc/bash.bashrc 文件。
/etc/bash.bashrc：系统级的 bashrc 文件，为每一个运行 bash shell 的用户执行此文件。此文件会在用户每次打开 shell 时执行一次。

注意：　/etc/environment 是设置整个系统的环境，而/etc/profile 是设置所有用户的环境，前者与登录用户无关，后者与登录用户有关。 这两个文件修改后一般都要重启系统才能生效。

 **用户级：** 

~/.profile: 是对应当前登录用户的 profile 文件，用于定制当前用户的个人工作环境。
每个用户都可使用该文件输入专用于自己使用的 shell 信息,当用户登录时,该文件仅仅执行一次!默认情况下,他设置一些环境变量,执行用户的.bashrc 文件。这里是推荐放置个人设置的地方
 **~/.bashrc:**   **是对应当前登录用户的 bash 初始化文件，当用户每次打开 shell 时，系统都会执行此文件一次。平时设置这个文件就可以了。** 

### 更换阿里源

`lsb_release -c` 查看 ubuntu 版本。
`cp /etc/apt/sources.list /etc/apt/source.list.bak` 备份原文件
`vim /etc/apt/sources.list` 修改文件为：

```shell
deb-src http://archive.ubuntu.com/ubuntu xenial main restricted #Added by software-properties
deb http://mirrors.aliyun.com/ubuntu/ xenial main restricted
deb-src http://mirrors.aliyun.com/ubuntu/ xenial main restricted multiverse universe #Added by software-properties
deb http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted multiverse universe #Added by software-properties
deb http://mirrors.aliyun.com/ubuntu/ xenial universe
deb http://mirrors.aliyun.com/ubuntu/ xenial-updates universe
deb http://mirrors.aliyun.com/ubuntu/ xenial multiverse
deb http://mirrors.aliyun.com/ubuntu/ xenial-updates multiverse
deb http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse #Added by software-properties
deb http://archive.canonical.com/ubuntu xenial partner
deb-src http://archive.canonical.com/ubuntu xenial partner
deb http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted multiverse universe #Added by software-properties
deb http://mirrors.aliyun.com/ubuntu/ xenial-security universe
deb http://mirrors.aliyun.com/ubuntu/ xenial-security multiverse
```

#### host 映射

`vim /etc/hosts` 添加映射 `ip 简称 全称`：

```
172.17.0.3      node2
172.17.0.4      node3
172.17.0.2      node1
```

#### 防火墙

`apt-get install ufw` 
`ufw enable`, `ufw disable`, `ufw reset`
`ufw default allow outgoing` 默认传入链接



