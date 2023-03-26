---
title: 嵌入式学习之（三）|C++ 2
date: 2020-06-28
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

#   C++ 基础 2

>  嵌入式自学开始啦~ 为了自己的贾维斯
>
> 太忙了，C++学了快 1 个月。没怎么用竟然也忘得差不多了。
>
> 部分笔记总结于交材：《c++ primer》 , 部分笔记课程链接：[千峰嵌入式教程](https://www.bilibili.com/video/BV1FA411v7YW?p=530&spm_id_from=pageDriver)

<!-- more-->

## C++模板

参数的类型不具体指定，用通用类型代替。在调用时，编译器会根据实参的类型，推导出形参的类型（类型参数化）

### 函数模板

```cpp
template <class T> //定义一个模板，模板的通用类型未 T
void swap_temp(T &a,T &b)
{
    T temp = a;
    a = b;
    b = temp;
}
int a = 1;
int b = 2;
swap_temp(a,b); //系统自动推导输入类型
```

函数模板不能进行自动类型转换

```cpp
T Myadd(T a, T b)
{
    return a+b;
}
int a = 10;
char b = 20;
Myadd<>(a,a); //调用模板函数
Myadd<int>(a,a);
Myadd(a,b)； //系统调用普通函数，因为普通的函数可以自动类型转换
```

+ C++优先考虑普通函数
+ 可以通过<>调用模板函数
+ 函数模板可以像普通函数那样被重载
+ 如果函数模板可以产生一个更好的匹配，那么选择模板

函数模板的本质就是函数重载，二次编译

#### 类模板

类模板不能自动类型推导

```cpp
template <class T1, class T2>
class animal
{
public:
    animal(T1 a, T2 b)
    {
        age = a;
        data = b;
    }
	T1 age;
    T2 data;
}; //类模板不能自动推导类型
void test()
{
	animal<int,int> dog(10,10);
    animal<int,string>cat(4,"lili");
    
}
void show(T1 &p){}

void show(animal<T1,T2> &p)
{
    cout << p.age <<p.data<<endl;
}//类模板作为形参,该函数需要写成函数模板
show(dog);
show(cat);//函数会自动推导类型
```

类模板继承

```cpp
template<class T>
class base
{
    base(T a){}
    public:
    T a;
}
tenplate<class T1, class T2>
class son: public base<T2>
{
public:
    son(T1 x1,T2 a):base<T2>(a),x(x1)
{}
    T1 x1;
}
    
```

类模板类外实现

```cpp
template <class T1,class T2>
    class person
    {public:
        person(T1 a, T2 b){
        }
     void show(){}
        T1 a;
        T2 b;
    }
template <class T1,class T2>
person<T1,T2>::person(T1 a,T2 b) 
{
                this-> a = a;
            this ->b = b;
}
    //类模板的成员函数在类外实现 需要写成函数模板

```

 类模板头文件和源文件分离问题

类模板如果不调用，就不会被创建

```cpp
#include "132jjj.h"
template <class T1>
    PERSON<t1>::PERSON(T1)
    {
        
    }
//调用的时候
int main()
    person<int,int>p(10,20);
    p.show()
}//调用构造函数和 show 函数需要创建，但是没有这两个函数的定义，不能创建函数 .h 函数的定义和声明写到一起去中去 如果用类内实现模板就不存在这种问题
```

函数模板与友元

```cpp
friend void showperson<>(person<T1,T2> &p)
    friend void showPerson2(person<T1,T2> 7p) //定义一个全局函数并声明为类的友元
```

## 类型转换

+ 静态转换：用于转换 int，char 等基本类型
+ 动态转换：用于转换类和类，不能转内置的数据类型
+ 常量转换：用于修改类型的 const 属性
+ 重新解释转换：用于指针间转换，整数和指针间也可以转

```cpp
a = static_cast<int>(b);  //静态转换，和 c 的强制类型转换一样。不能转换没有发生继承关系之间的类
A *p1 = new A; //父类
B *p2 = new B; //子类
p1 = dynamic_cast<A *>(p2); //不能用于没有继承关系的类之间的转换，只能子类转父类
const int *p2 = NULL;
int *p1 = NULL;
p1 = cont_cast<int *>(p2);
p2 = const_cast<const int *>(p1);

int *p = NULL;
char *p2 = NULL;
int c = 0;
p = interpret_cast<int *>(p2);
c = reinterpret_cast<int>(p2); //重新解释转换，指针
```

## 异常

+ 尝试捕获异常

```cpp
itn mydive(int a,int b)
{
    if (b==0)
        throw 1; //抛出异常
}

try
{
mydive(2,0);    
}
catch(int)
{
    cout <<"捕获了 int 类型的异常"<<endl;
}
catch(...){
}//捕获其他类型的异常
```

+ try 和 catch 必须同时存在，如果抛出异常然而 catch 没有捕获到该类型异常，那么程序会终止

+ 栈解旋: 在 try 到 throw 之间定义的对象，在 throw 之后会被释放
+ 异常接口申明：linux 和 qt 下可以，vs 不行

```cpp
void test()throw(int,char,char*)
void test()throw()//不能抛出任何异常
```

+ 异常变量：如果抛出匿名对象，它的声明周期在 catch 里

```cpp
class myexception
{  }
void fun(){
    throw myexception();
}
void test(){
try {
    fun()}
}
catch(myexception &p){ 
    p.errror()
}
```

+ 多态实现异常处理

```cpp
class myexception
{
    public:
    virtual void error() = 0;
};
class out_of_range(): public myexception
{
    public:
    void error()
    {
        cour<<"out of range"<<endl;
    }
};
class bad_cast:public myexception
{.....}
void fun(){
    throw out_of_range(); //也可以
    throw bad_cast();
}
try{
    fun();
}
catch(myxeception &p) //用父类引用
{ 
}
```

### C++ 异常库

```cpp
#include <stdexcept>
exception p;
p.what() //打印出错信息 

class error1:public exception
{
    public:
    const char *what() const 
    {
        return data.c_str();
    }
    string data
}//自定义异常类
```

## STL

### 容器

+ string

```cpp
string str;
string str1("hello");
string str3(5,'K');

str = str1;
str = "string";
str = 'c';
str.assign(str1);
str.assign("string");
str.assign("abc",2); //str = "ab"
str.assign("123123",2,3) //切片
    
str[4] = 'c';//str.at(4)
str += "123"; // str += 'c'
str.append("123"); // str.append("123",2,3);
str.find(str1,1); //从 1 开始查找 str1


```

+ vector

```cpp
#include <vector>
vector<int> v;
v.push_back();
vector<int> v2(v1.begin()+1,v1.end())
v2 = v1;
vector<int>(v1).swap(v1); //swap 收缩空间

vector<int>::iterator it_start = v.begin();
vector<int>::iterator it_end = v.end();
for (;it_start != it_end;it_start ++){}

v.size();//v.resize() resize 后 size 变了，但是 capacity 不变
//.empty().capacity().resize().reserve(提前预留位置)

vector<int>::iterator iter;
      for (iter = ivec.begin(); iter != ivec.end(); iter ++) {
          cout << *iter << " ";
      }
cout << endl;

v[1]; //v.at(1)
v.front();//v.back

v.insert(2,3,9); //在第 2 个位置插入 3 个 9
v.pop_back();//删除最后一个
v.erase();//删除迭代器
v.clear();

bool compare(int a, int b)
{
    return a < b;
}
sort(v.begin(),v.end(),compare) //compare 编写排序规则

```

+ deque

```cpp
deque<int> d;
deque<int> d1(d);
deque<int>d3(d1.begin(),d1.end())
d.push_back(); 

d.assign(d1.begin(),d1.end());
d.size();//empty;resize

d.push_back(); //push_front,pop_back,pop_front

d[1]; //d.at(1)
d.front();//d.back() 第一个与最后一个
d.insert(pos,elem);
d.clear();
d.erase(beg,end);
```

![image-20210107170029762](/assets/img/C++2/image-20210107170029762.png)



### 算法

+ 函数对象

```cpp
negate<int> p; //plus,minus,multiplies,divides,modulus
greater<int>(); //equal_to,not_equal_to,greater_equal,less,less_equal
```

+ 适配器

```cpp
//二元继承
class print:public  binary_function<int,int,void>
{
    public：
        void operator()(int a, int num) const{};
//绑定参数
bind2nd(Print(),100)
    
```

### Lambda

```
[capture](parameters) mutable ->return-type
{statement}
[函数对象参数](操作符重载函数参数)mutable ->返回值{函数体}
```

函数对象参数；

[]，标识一个 Lambda 的开始，这部分必须存在，不能省略。函数对象参数是传
递给编译器自动生成的函数对象类的构造函数的。函数对象参数只能使用那些到
定义 Lambda 为止时 Lambda 所在作用范围内可见的局部变量（包括 Lambda 所在类的
this）。函数对象参数有以下形式：
n 空。没有使用任何函数对象参数。
n =。函数体内可以使用 Lambda 所在作用范围内所有可见的局部变量（包括
Lambda 所在类的 this），并且是值传递方式（相当于编译器自动为我们按值传
递了所有局部变量）。
n &。函数体内可以使用 Lambda 所在作用范围内所有可见的局部变量（包括 Lambda 所在类的 this），并且是引用传递方式（相当于编译器自动为我们按引
用传递了所有局部变量）

n this。函数体内可以使用 Lambda 所在类中的成员变量。
n a。将 a 按值进行传递。按值进行传递时，函数体内不能修改传递进来的 a 的拷
贝，因为默认情况下函数是 const 的。要修改传递进来的 a 的拷贝，可以添加
mutable 修饰符。
n &a。将 a 按引用进行传递。
n a, &b。将 a 按值进行传递，b 按引用进行传递。
n =，&a, &b。除 a 和 b 按引用进行传递外，其他参数都按值进行传递。
n &, a, b。除 a 和 b 按值进行传递外，其他参数都按引用进行传递。

操作符重载函数参数；
标识重载的()操作符的参数，没有参数时，这部分可以省略。参数可以通过按值
（如：(a,b)）和按引用（如：(&a,&b)）两种方式进行传递。
③ 可修改标示符；
mutable 声明，这部分可以省略。按值传递函数对象参数时，加上 mutable 修饰符后，可以修改按值传递进来的拷贝（注意是能修改拷贝，而不是值本身）。

函数返回值；
->返回值类型，标识函数返回值的类型，当返回值为 void，或者函数体中只有一
处 return 的地方（此时编译器可以自动推断出返回值类型）时，这部分可以省
略。

