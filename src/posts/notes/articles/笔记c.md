---
title: 嵌入式学习之（一）|C 语言
date: 2020-06-07
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

#   C 非常入门笔记

> 嵌入式自学开始啦~ 为了自己的贾维斯
>
> 笔记总结 课程链接：[千峰嵌入式教程](https://www.bilibili.com/video/BV1FA411v7YW?p=530&spm_id_from=pageDriver)

<!-- more-->


## 1.1 入门 

+ key word
  + rigister (for char, int) - 不能对 register 取地址

  + static - 修饰的变量保存再内存的静态空间种 | const - 不能修改的值 | extern - 用于全局变量声明

```c
char a = 'e';  // char a = 97; 其中 97 为 ASCII 值
	//%c 对数字输出 ASCII 对应字符, %d 对字符，输出 ASCII 值
	
	%d 十进制；%ld long 十进制有符号； %u 十进制无符号； %#o 八进制表示；%#x 十六进制表示；%f float；%lf double；%e 指数型浮点数；%c 单个字符；%s 字 c 符串；%p 指针的值
```

+ 类型转换


```c
- 自动转换：char, short > signed int > unsigned int > long > double
5/2 // = 2
5.0/2 //  = 2.5

int c = 7;
float a = 8.1f;
c = a; //c: 8
- 强制转换
(float)a; (int)(x + y);  //计算后再转换
```

+ 运算符：

  +  三目运算符：(A)?(B):(C) -- return B if A else C
+  A = (B,C,D) -- 依次执行 BCD，A 为 D 的结果
  
  +  % 只对整数有用
+  && ( and), || (or), ! (not)
  + ==位运算(二进制的运算，计算机自动转换)==

  
  + | , 等于 or
  + ~， 1 变 0，0 变 1
  + ^，异或
    + 、>> 右移，逻辑右移：低位溢出，高位补 0/ 算数右移：高位补符号位
  + b = a++  same as  b = a, a += 1
    b = ++a  same as  a += 1, b = a
  

```c
switch (int/char condition)
{case condition1:  // if condition = condition1
 		break; // 如果没有 break，执行完后会执行下一个 case 的语句，知道遇到 break
 default:
	break;}
```

```c
if(){goto NEXT;}
NEXT:
	跳转到这里
```

### 函数

  ```c
// myfun.h 头文件中用来声明函数
#pragma once
void myfun1();

//myfun.c 中
#include<stdio.h> #include "myfun.h"
void myfun1(){printf("my functino");	return;}

//main
#include<stdio.h> #include "myfun.h"//用“”包含当前目录下找文件，<>包含系统中头文件。不要#include xxx.c 文件
int main(void){myfun1();return 0;}
  ```

内存分区

+ 虚拟内存

  + 再写程序的，%p 看到的都是虚拟内存
  + 32 位操作系统中，虚拟内存分为 3G 用户空间和 1G 内核空间
  + 堆，栈（存放局部变量），静态全局区，代码区，文字常量区

    ```c
    static int num;  // 静态全局变量， 只在定义的.c 文件中有效
    void fun(){    int a; //默认随机值   
      static int num = 8; //静态局部变量只初始化一次，默认赋值 0}
    ```

## 宏



```c
#define PI 3.14  //不带参宏，一般大写
#undef PI//
#define S(a,b) ((a)*(b))  //加上括号防止计算顺序错误  
num = S(2 + 8, 4)； //带参宏浪费空间，节省时间； 带参函数浪费空间，节省空间
```
选择编译: 只有一块代码会被编译，都是再预编译，预处理阶段执行的 // 常用于头文件中，防止重复包含

```c
#define AAA
#ifdef AAA
	...
#else
	...
#endif
```




### 指针 （以下内存为虚拟内存用户空间）

 **指针类型的转换** 

 ``` c
  int a = 0x1234；
  char *p1,*p2;
  p1 = (char *)&a;  //*p1 = 0x34,把&a 转换成 char 地址，
p1++;  //*p1 = 0x12，*+指针 取值，char 类型后取一个字节，指针++为加上指针字节数量，char 加 1，int 指针加 4
 ```

 **数组与指针** 

 ```c
  int a[10];  int *p; p = a;  
  //p 指向 a[] 第一个元素所在地址  
  //p 和 a 的不同：p 是变量，可以被赋值，a 是常量，不能被修改,其他功能大致相同
  p[2] = 100; 
  *(p+2) = 100; //等于*(a+2), p+2 等于&a[2]为 index 2 所在位置（对于 int，p+2 相当于 16 进制的地址+8）
  //指针相减，等于指针之间数据个数
 ```

 **指针数组**  

```c
int *p[10]; //指针变量的类型相同 这边都是 int
int a,b; p[1] = &a; p[2] = &b;
q = &p[1];
char *name[3] = {"follow me","hello","???"} //常用于保存字符串
```

 **指针字符串**  

```c
char str[100] = "abc"; //str[0] = "y"可以执行修改。栈和全局内存中的内容课修改。通常使用再局部的函数
char *str = "abc"; //*str == "a"，几乎不用，执行*str = "y" error,文字常量区的内容不可修改. str == "abc"
char *str = (char*)malloc(10*sizeof(char));//动态申请了 10 个字节的储存空间。堆区中的内容可修改。
 **如果想要再其他的文件中也是用对应的字符串，建议使用堆区。堆区空间手动申请，手动释放** 

char buf_aver[] = "hello";
char *buf_point = "hello"; //字符数组，指针指向的字符串，可以直接初始化
char *buf_heap = (char*)malloc(10*sizeof(char));  //堆中存放的字符串不能初始化，只能用 srcpy/scanf 复制

strcpy(buf_heap,"hello");
buf_aver = "hello";  //error,这边 buf_aver 被定义成数组名。用 strcpy(buf_aver,"hello")
buf_point = "hello";  //可以，*buf_point 为 h，且无法修改

```

 **数组指针**   -- 用于给传入二维数组参数

```C
int (*p)[5]; int a[3][5]; // 数组指针的作用是，可以保存二维数组的首地址；
p = a; //
void fun(int(*p)[5]);//可以将二维数组传参, 调用时候： fun(a);

int a[10]; // &a 变成一个一维数组指针（升级指针）， (&a) + 1 跳跃了 40 个字节。但是对 p 取地址变成一个 int 地址
int b[10][8]; // *b 表示地址的降级为列指针，b 为行指针，b+1 为下一行的首地址，*b +1 为下一个元素的地址
```

 **指针与函数**  (修改变量地址上的数值，实现对全局变量的修改)

```c
void myfun(int *p,int *q,int **z){
}
char *z; int a,b; myfun(&a,&b,&z)// 传参时取地址，myfun 函数中用取值
```

 **函数传参** 

```c
void fun1(int *p) // 传入一维数组的地址
void fun2(int (*p)[4]) //传入 2 维数组，a[0][2]相当于 p[0][2]
void fun3(char **q) //传指针数组
```



 **指针函数** ，函数返回指针（用于返回字符串）


  ```c
  char *fun4()
  {char str[100] = "hello"; //栈区定义的空间会在函数运行结束后释放，这样 return 的指针指向 null
   // static char str[100] = "hello"; //静态区的空间不会随着函数的结束而释放
  return str;}
  ```

 **函数指针** , 将一个函数作为参数传递给另一个函数

```c
int (*p)(int,int);
int max(int x,int y);
p = max;
```

 **回调函数** 

```c
int add(int x,int y){return x+y}
int process(int (*p)(int,int),int a,int b)
{int ret;
ret = (*p)(a,b);
return ret;}
int main(){num = process(add,1,2);}
```

 **空类型指针** 

````c
void *q;int *p;q = p; //void 通用指针,主要用在函数的参数和返回值
char *z = NULL;
````

 **main 函数传参** 

```c
int main(int argc, char *argv[]) //argc：传入参数的个数；argv：保存每一个命令终端传入的参数
```

### 动态内存申请

malloc, calloc

+ 申请后一定要判断是否申请成功，多次申请的内存，内存不一定连续；

```c
#include <stdlib.h>
void *malloc(unsigned int size); //一般用 memset 初始化
void free(void *ptr) //释放堆区的空间，必须释放 malloc，calloc 或 relloc 的返回值对应的所有空间。

char *str = (char *)malloc(100 * sizeof(char)); //强转指针类型，为 str 开辟堆区空间

void *calloc(size_t nmemb, size_t size);  //参数（申请个数，每块的大小）
void *realloc(void *s, unsigned int newsize); // 增加或减少原先申请的空间，参数（原空间首地址，重新开辟空间大小）
```

内存泄漏（首地址丢了）

```c
void fun(){ char *p;
p = (char *)malloc(100);
return p;
}
//或 free(p)；释放
// p = NULL； 防止野指针
```

### 字符串处理函数

字符串遇到\0 结束

```c
#include <string.h>

char s1[100] = "hello";
printf("%d",strlen(s1));
char *strcpy(char *dest, const char *src); //赋值包括\0，输出遇到\0 结束，保证目的地参数足够大
char *strnpy(char *dest, const char *src, size_t_n); //复制前 n 个 char
char *strcat(char *dest, const char *src); //append 字符串,包括\0
char *strncat(char *dest, const char *src, size_t_n); //append n 个字符串,包括\0
char *strcmp(const char *s1, const char *s2)// 相等返回 0，strncmp 比较前 n 字符
char *strchr(const char *s, int c) //找 ascii 码为 c 的字符，首次匹配。strrchr 末次匹配。 返回所在地址
char *strstr(const char *haystack, const char *needle);//在 haystack 指向的字符串中找 needle 指像字符串
int atoi(const char *nptr); //将 nptr 转为 int，“123”转成 123 #include <stdlib.h> 用 atof 转化字符串为浮点型的数字
char *strtok(char *str,const char *delim); //切割字符串
char *ret; char s[] = "11:22:33:44:55"
ret  = strtok(s,":"); // 返回 11
while((ret = strtok(NULL,":"))){ printf("%s\n",ret) } //返回 22 33 44 55

```

格式化字符串

```c
int sprintf(char *buf, const char*format);
sprintf(buf,"%d,%d,%d",1,2,3); // buf = ‘1,2,3'
sscanf("1:2:3","%d:%d:%d",&a,&b,&c); //a = 1, b = 2 匹配同样类型的字
sscanf("1234 5678","%*d %4s",buf); //跳过数字， %[width]s 保存 n 个字
```

const

```c
const int a = 100; //全局变量，只读，不能修改
void fun(){ const int b = 100;} //不能直接赋值修改，可以通过地址修改局部变量
void fun(){ 
    int c = 100;
    const int *p = &c; //这样无法修改 *p = 1;
    int * const p = &c; //这样无法修改 p = &d;
	}
```

### 结构体，共用体，枚举

 **普通结构体变量** 

```c
struct stu{ int num; char name[20];}xiaoming, xiaohong;//一般结构体都会定义在全局
struct stu bob = {123，"bob"}; //初始化变量

typedef struct{int year; int month; int day}BD;
typedef struct {int num; char name[20]; BD birthday;}STU; // STU 为结构体类型，相当于 struct + 结构体名
STU xiaohong = {123，"小明",{2008,12,12}}; //或{123，“小明”,2008,12,12}

xiaohong.num = 1001;
strcpy(xiaohong.name,"小红");
xiaohong.birthday.year = 2008;
xiaohong.birthday.month = 12;

xiaohong = xiaoming; //相同结构的结构体可以互相赋值
```

 **结构体数组** 

```c
struct stu edu[3] = {{123,"xiaoming"},{23,"asd"}}; //edu[0],edu[1]..
```

 **结构体指针** 

```c
#include <stdlib.h>
struct stu *s;  (*s).num = 101; s -> 101;
s = (struct stu *)malloc(sizeof(struct stu)); //常用
s->num = 101;
```

结构体内存分配，

+ 按照最大的字节类型开辟空间，根据最大的字节类型进行字节对齐
+ 出现数组时，可以看作多个变量

```c
struct stu{ char a;short int b;int c; } //8 字节 
struct stu{ char a; int c; short int b;} //12 字节
```

 **位段** 

```c
struct stu{unsigned int a:2; int b:4};  // a 只能取 0-3
```

 **共用体 （struct 换成 union)** 

```c
union un{int a; int b; int c};
union un myun;
myun.a = 100; myun.b = 200; myun.c = 300; //最后 abc 都是 300
```

 **枚举** 

```c
enum week {mon = 1,tue= 2, wed = 3, fri = 5}; 
enum week day1 = mon; //变量只能用里面的值，这边 day1 = 1
```

### 链表

 **单项链表** 

```c
typedef struct stu{ int num; char name[20]; struct stu *next;}STU;
void linked_list_append(STU **p_head,STU *p_new){
    STU *p_mov = *p_head;
    if(*p_head == NULL)
    {  *p_head = p_new;
        p_new -> next = NULL;
    }else{
        while(p_mov->next != NULL)
        { p_mov = p_mov->next; }
        p_mov -> next = p_new;
        p_new->next = NULL;
    }
}
int main()
{
    STU *head = NULL,*p_new = NULL; int num,i;
    for (i = 0;i<num;i++){
        p_new = (STU*)malloc(sizeof(STU));
        p_new->num = 100;
        strcpy(p_new->name,"小红");
        linked_list_append(&head,p_new)
    }
}


STU head = {1,"name",NULL};
STU node = {2,"node1",NULL}; head -> next = node;
```

+ 链表释放

  + 重新定义一个指针，保存 p-> next。 然后释放 p free(p);p = NULL;

### 文件

行缓冲 - 到换行，程序退出 或缓冲区满 1024 bytes 时候才刷新，执行读写操作

```c
printf("ello");  //不加换行符不输出
while(1){}
ffulsh(stdout); //刷新
```

文件结构体指针

```c
stdin , stdout, stderr // 三个特殊文件指针
FILE *fp;
fp = fopen( "c:/Users/file.txt","r");
if(fp == NULL)
{
    printf(" fail to open");
    return -1;
}
int c = fgetc(fp);  //int fgetc(FILE *stream); 读取一个字节,可以读取到换行符。while((c = fgetc(fp)) != EOF)
printf("%d",c);
fputc("w",fp); //写入一个字节

char buf[32] = "";
fgets(buf,8,fp); // 读取 7 个字节加上一个\0， 遇到换行结束
fputs(“hhhhhh”，fp); 
fclose(fp);

```



```c
num = fread(str,100,3,fp); //读取 3 块，每块 100 字节； 读到 200-299 个字，返回 2。
fwrite(str,100,3,fp); //返回实际写的块数。可以写入结构体
rewind(fp); //重置文件偏移量

fprintf(fp,"%d%d",1,2); //写入字符串
fscanf(fp,"%d%d",a,b);  //读取文件，保存到 a,b

```

 **文件指定位置读写** 

```c
ftell(fp); //获取当前指针位置 %ld
fseek(fp,3,SEEK_SET); //开头后移 3 个位置，SEEK_SET,SEEK_CUR(当前位置),SEEK_END
```



