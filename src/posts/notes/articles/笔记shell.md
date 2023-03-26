---
title: 嵌入式学习之（七）| Linux 高级程序设计 1
date: 2020-10-17
author: Kevin 吴嘉文
keywords: 
language: cn
category:
- 知识笔记
tag:
- Linux
- 计算机语言
- 嵌入式学习
mathjax: true
toc: true
comments: 
---

# shell 笔记

> 嵌入式自学开始四个月啦~ 为了自己的贾维斯 ？！！！
>
> shell 学了就没用过，要找个时间试试！
>
> 笔记总结 课程链接：[千峰嵌入式教程](https://www.bilibili.com/video/BV1FA411v7YW?p=530&spm_id_from=pageDriver)

<!-- more-->

## shell 脚本

+ 定义以开头: #!/bin/bash	
  + 声明脚本由 shell 解释

+ .sh
+ 执行

```
chmod +x test.sh ./test.sh # 增加可执行权限后执行
# 或者使用：（不需要修改权限）
bash test.sh
. test.sh # 使用当前 shell 读取解释
```

如果使用 `bash test.sh`那么将明确指定 bash 解释器去执行脚本，脚本中的#!指定的解释器不起作用。第一种首先检测#!，如果没有则使用默认的 shell

例

```
#!/bin/bash
# shell 脚本是 shell 命令的有序集合
ls
pwd
echo "hello world"
```

### 自定义变量

+ shell 无数据类型

+  **赋值等号两边不能加空格**  

```
num=10 
i=$num
echo $num
unset varname #清楚变量值
```

```
# 使用 read 终端读取数据
read str
echo "str = $str"
# readonly 只读变量
readonly n=999
```

#### 环境变量（规范使用大写）

env 命令查看环境变量

unset 清除环境变量

在终端执行命令，临时设置变量

```
MYVAL=999
export MYVAL
```

永久设置，需要在配置文件（~/.bashrc 或/etc/profile）中进行设置即可，设置完毕后需要通过`source ~/.bashrc` 命令配置文件立即生效，否认需要重启终端来生效

#### 预设变量

```
$#：传给 shell 脚本参数的数量
$*：传给 shell 脚本参数的内容
$1、$2、$3、...、$9,${10}：运行脚本时传递给其的参数，用空格隔开
$?：命令执行后返回的状态
"$?"用于检查上一个命令执行是否正确(在 Linux 中，命令退出状态为 0 表示该命令正确执
行，任何非 0 值表示命令出错)。
$0：当前执行的进程名
$$：当前进程的进程号
"$$"变量最常见的用途是用作临时文件的名字以保证临时文件不会重复
```

#### 脚本变量的特殊用法

```
""（双引号）：包含的变量会被解释
''（单引号）：包含的变量会当做字符串解释
``(数字键 1 左面的反引号)：反引号中的内容作为系统命令，并执行其内容，可以替换输出为
一个变量，和$()的结果一样
$ echo "today is `date` "
today is 2012 年 07 月 29 日星期日 12:55:21 CST

\ 转义字符：
同 c 语言 \n \t \r \a 等 echo 命令需加-e 转义
echo ‐e "this \n is\ta\ntest"

(命令序列)：
由子 shell 来完成,不影响当前 shell 中的变量
( num=999;echo "1 $num" )
echo 1:$num
{ 命令序列 }：
在当前 shell 中执行，会影响当前变量
{ num=666; echo "2 $num"; }
echo 2:$num
```

#### 条件测试语句

test 命令：用于测试字符串、文件状态和数字
test 命令有两种格式:
`test condition` 或 `[ condition ]`
 **使用方括号时，要注意在条件两边加上空格** 
shell 脚本中的条件测试如下：
文件测试、字符串测试、数字测试、复合测试
测试语句一般与后面讲的条件语句联合使用

##### 文件测试

 **1）按照文件类型** 
-e 文件名 文件是否存在
-s 文件名 是否为非空
-b 文件名 块设备文件
-c 文件名 字符设备文件
-d 文件名 目录文件
-f 文件名 普通文件
-L 文件名 软链接文件
-S 文件名 套接字文件
-p 文件名 管道文件

 **2）按照文件权限** 
-r 文件名 可读
-w 文件名 可写
-x 文件名 可执行

 **3）两个文件之间的比较** 
文件 1 -nt 文件 2 文件 1 的修改时间是否比文件 2 新
文件 1 -ot 文件 2 文件 1 的修改时间是否比文件 2 旧
文件 1 -ef 文件 2 两个文件的 inode 节点号是否一样，用于判断是否是硬链接

```
#! /bin/bash

echo "please input a filename >>> "
read FILE

test ‐e $FILE
echo "存在？$?"
```

##### 字符串测试
s1 = s2 测试两个字符串的内容是否完全一样
s1 != s2 测试两个字符串的内容是否有差异
-z s1 测试 s1 字符串的长度是否为 0

-n s1 测试 s1 字符串的长度是否不为 0

```
test "hello" = "hello"
echo "相等？$?"
test ‐z "hello"
echo "长度是否为 0？$?"
```

##### 数字测试
a -eq b 测试 a 与 b 是否相等
a -ne b 测试 a 与 b 是否不相等
a -gt b 测试 a 是否大于 b
a -ge b 测试 a 是否大于等于 b
a -lt b 测试 a 是否小于 b
a -le b 测试 a 是否小于等于 b

##### 复合测试
第一种形式：命令执行控制 (两边需要完整的命令)
&&：
command1 && command2
&&左边命令（command1）执行成功(即返回 0）shell 才执行&&右边的命令
（command2）

||
command1 || command2
||左边的命令（command1）未执行成功(即返回非 0）shell 才执行||右边的命令
（command2）

第二种形式：多重条件判定

-a : test -r file -a -x file

-o : test -r file -o -x file

! : test ! -x file

### 控制语句

#### if

#### 格式一

> if [ 条件 1 ]; then
> 	执行第一段程序
> else
> 	执行第二段程序
>
> fi
>

 **中括号的空格一定要** 

#### 格式二：
> if [ 条件 1 ]; then
> 执行第一段程序
> elif [ 条件 2 ]；then
> 执行第二段程序
> else
> 执行第三段程序
> fi

#### case

```
case "$1" in
	"one" | n*)  # n* 星可以匹配任意多个字符
		echo "your choice is one"
		;;
	"two")
		echo "your choice is two"
		;;
	"three")
		echo "Your choice is three"
		;;
	*)
		echo "Error Please try again!"
		exit 1  # 推出整个程序，后面的都不会执行
		;;
esac
```

#### for

我们可以讲命令的输出放在 for 循环后面，来进行骚操作

![image-20210224141934571](/assets/img/shell/image-20210224141934571.png)

形式一：

可以声明整数类型变量`declare -i sum`
需要声明 sum 才可以进行加法运算

```
declare -i sum
for (( i=1; i<=100; i++ ))
do
	sum=sum+i
done
```

形式二：

```
for i in 1 2 3 4 5 6 7 8 9
do
	echo $i
done
```

#### while

 **中括号两边要加空格** 

```
while [ "$i" != "101" ] 
do
	s+=i;
	i=i+1;
done
```

#### until

条件成立后中断

```
until [ "$i" = "101" ]
do
s+=i;
i=i+1;
done
```

#### break, continue

与 C 中一样

### 函数

+  **shell 中，出了括号里定义的变量，函数中定义的变量不加修饰的话，都可以认为是全局变量** 

格式二：
function 函数名（）
{
命令 ...
}

return 从函数中返回，用最后状态命令决定返回值。
return 0 无错误返回
return 1 有错误返回

```
myadd()
{
        A=$1
        B=$2
        SUM=`expr $A + $B`
        echo "$A + $B = $SUM"
        return $SUM
}
myadd 33 33
echo "$?"

#函数的返回值一般通过$?可以获取到，但是$?获取到的最大值是 255，如果超过这个值，会出错


```

 























