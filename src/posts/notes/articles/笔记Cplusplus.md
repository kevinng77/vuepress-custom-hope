---
title: 嵌入式学习之（二）|C++基础
date: 2020-06-13
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

# C++ 非常入门笔记

> 嵌入式自学开始啦~ 为了自己的贾维斯
>
> 想不到一个金融学生竟然走上了这样的路
>
> 笔记总结 课程链接：[千峰嵌入式教程](https://www.bilibili.com/video/BV1FA411v7YW?p=530&spm_id_from=pageDriver)

<!--more-->

## 基础

```cpp
#include <iostream> //标准输入输出流
```

## Namespace ::

```cpp
int a = 100;
//namespace 定义在全局
namespace B {int a = 20;} //添加项目进 namespace B
namespace B{ namespace A {int a = 10;void fun();}}
void test(){
cout << ::a <<endl; //全局变量 100
cout << B::a <<endl; //a = 20;
cout << B::A::a <<endl; //a = 10;
}
```

`void A::fun(){int a = 19;}` 在外部定义 namespace A 声明的函数

`namespace{int a = 1;}` namespace 没有取名，则全部成员只能被当前文件调用

`namespace newname = oldname;` 改名

```cpp
void test(){
using nameA::a; //声明后，函数中不能定义其他 a
cout<<a<<endl;
}
```

`using namespace nameA;` 后 namespaceA 中的全部标识符都可以直接使用，可以省略 nameA::，有变量冲突则使用局部变量

## C++ /C 对比

+ C++：全局变量无法重复定义。C 中多次定义会默认一次定义，多次声明
+ C 编译函数形参可以没有类型定义，C++必须有参数类型
+ C 编译函数 void 可以有返回值
+ C++：更严格的类型转换

```cpp
char * p = malloc(100); //不允许
char * p = (char*)malloc(100); //允许
```
+ C++：定义结构体可以省略 struct

+ C++: bool 类型关键字可以直接使用

+ 三目运算符：

  ```cpp
  a<b?a:b; //C 返回 a 的值，C++返回变量 a
  (a<b?A:b) = 100; //c++ 允许
  *(a<b?&a:&b) = 100; //c 中操作
  ```

+ const:

  + C:  函数中 const 修饰的局部变量保存在栈区，可以用*p = &a; 改变 a 的值；

  + c++: 局部的 const 变量保存在符号表中，无法修改

  + C/C++: 全局变量 const 保存在常量区，不能修改;

  + C: extern const int b; //声明外部可用（其他文件使用）

  + c++: const 局部变量

    ```cpp
    const int a = 10;//如果是函数局部变量，onst 修饰局部变量没有分配内存
    int *p = (int *)&a; //会分配一个临时空间，int tmp = a, *p = &tmp
    *p = 100; //无法修改 a 的值

    const int a = b; //const 修饰的局部变量为变量，a 可以被修改，存在栈区
    
    cosnt struct stu obj; //保存在栈区
    struct stu *p = (struct stu *)&obj； //修改 obj
    
    extern const int num=1; //定义外部文件可以引用的 const 变量
    
    ```

###  **引用**  

+ 实质是为变量取别名

```cpp
int a = 10;
int &b = a; //等号左边&为引用，等号右边为取地址; 引用初始化后无法改变引用指向的目标；引用定义时必须初始化，不能写 int b; 
b = 100;
b = c; //表示把 c 的值赋值给 b，不是给 c 取别名为 b

int c[]={1,2,3,4};
int (&arr)[4] = c; //typedef int ARR[5]; ARR & arr = c;
cout << arr[i] << " ";
```

+ 函数的引用 (190)

```cpp
void swap(int &x,int &y){int tmp = x; x = y; y = tmp}//交换 x，y;引用做函数形参
swap(a,b);

int & fun(){static int b = 100; return b;}//返回 int 的引用；不能返回局部变量的引用

type & q = p;

const int &b = a; //常量引用,无法修改引用指向空间的值
const int &b = 100; //允许； int tmp = 1; int &b = tmp;

```

+ **引用本质是指针常量 // Type * const ref = &val; 指针常量只能指向同一块地址**

### 内联函数

+ （保持宏函数的效率）（用空间换效率）（不能存在循环，判断，取地址）

+ 内联函数只是给编译器一个建议，编译器不一定会接受这种建议

```cpp
#define FUN(a,b) a+b // 宏问题 FUN(a,b) * 5 = a + b*5,预处理只会替换，不会检查函数问题
inline int fun(int a,int b){return a+b;} //替换发生在编译阶段，不会有函数调用的开销
#define FUN(a,b) a<b?a:b // FUN(++a,b)出错
inline int fun(int a,int b){return a<b?a:b;}
//任何在类中的函数自动成为内联函数

```

+ 函数参数默认值，占位参数

  ```cpp
  void fun(int a = 10,int b = 20); // 声明和定义只能一处设置默认参数
  void fun(int a, int){}//占位参数，没什么用
  ```

+ 函数重载: 同一 namespace, 不同参数个数，参数顺序，参数类型 （编译器自动更换函数名）

+ extern C (205) ：c++编译修改函数名

  ```cpp
  int main(){
      cout << myfun(a,b)<<endl;
      return 0;
      
  }
  
  //C 的头文件加入
  #if __cplusplus
  extern "c" {
  #endif
      int myfun(int a,int b);
  #if __cplusplus
  }
  #endif
  ```

+ 结构体，C 中不能放函数，C++可以放函数

  ```cpp
  struct _stu
  {
      int a; int b[5];void print_stu(){cout<<a<<endl;}
  };
  ```

## 类

```cpp
class Person{
    public:int age;
    protected:int tall;//子类可访问
    private:int money; //子类不可访问
};//struct 换成 class 就行；class 中成员默认都是私有访问
//尽量提供接口（public 函数）来实现类中变量的修改和初始化
```

```cpp
class Person{
public:
    void person_init(int age, char *name ){
        m_age = age;
        strcpy(m_name, name);      
    }
    void show_person(){ }
    int get_age(){
        return m_age;
    }
    void set_age(int age)
    {
        if (age >= 0 && age <= 100){
            m_age = age;
        }
    }
    char *get_name()
    {
        return m_name;
    }
    void set_name(char *name)
    {
        strcpy(m_name, name);
    }
    
private:  //建议成员设为私有，然后提供接口进行访问
    int m_age;
    char m_name[128];
}

void test()
{
    Person p1;
    p1.person_init(20,"lucy");
    p1.show_person();
    p1.set_age(30);
}
```



+  **构造和析构（初始化和清理）系统会默认提供构造和析构函数，但系统提供的构造和析构函数不做事，一般需要人为提供**  （及时清理指针等，节约内存）


构造函数可分为有参构造，无参构造

+ 拷贝构造：旧对象初始化新对象

```cpp
class Person{
    public:
    {
        //构造函数 1. 函数名和类名一致 没有返回值，不能写 void，可以有参数， 可以发生函数重载
        Person(){}//无参构造
        Person(int a, int b){m_a = a; m_b = b;} //有参构造，提供构造函数后系统不在提供默认构造函数
        person(person &p)
        //添加 explicit ，禁止通过隐式法调用构造函数
        explicit Person(const Person &p){
            m_a= p.a; name = (char *)malloc(strlen(str)+1);//再做拷贝时候，重新给对象开辟一个空间
            strcpy(name,p.name);
        }//使用深拷贝，代替系统默认的浅拷贝构造，简单的浅拷贝做了简单的值拷贝，类中变量指向同一个地址，析构时会出错
        
    ~Person(){free(name);
             name = NULL;} //析构函数 类名前加上~， 没有返回值，无参数，不能发生函数重载   
    }
}
```

构造与调用

```cpp
void fun(){
    //括号法
Person p1(1,2); //构造函数再实例化对象时自动调用
    Person P2; //无参构造不使用括号
  person(1,2);//匿名对象；或 person()无参构造调用匿名对象;生命周期再当前行
    //销毁之前（如出了函数，或作用域）析构函数自动调用
    person(p2);//再定义时，不能用括号法调用匿名对象
    
    
  	//显示法
    person p1 = person(10,"luchy"); //显示法调用有参构造
  
    
    //隐式法(不大使用)
    person p1 = {10,"lucy"}
}
```

+ 初始化列表

```cpp
person(int a, int b): m_a(a),m_b(b){}//下方先声明 int m_a,int m_b，再根据声明的顺序进行定义初始化
```

类对象作为类成员

```cpp
class Phone{}
class Person{
    public:
    Person(string per_name,string phone_name):per_name(name1),phone(phone_name){} //通过初始化列表调用有参构造
    
    string per_name;
string phone；//先构造内部类，先声明先构造；析构顺序与构造相反
}
```

+ 类的其他特性

```cpp
class Screen {
public:
	using pos = string::size_type;
	Screen() = default;
	Screen(pos h,pos w):height(h),width(w), contents(h * w,' '){}
	Screen(pos w, pos h, char c) : width(w), height(h), contents(h* w, c) {}
	
	Screen &move(pos w, pos h) { cursor = w +  h*height; return *this; }
	Screen &set(char c) { contents[cursor] = c; return *this; }
	void display(ostream& os) { do_display(os);  }
private:
	void do_display(ostream &os) { os << contents; }
	pos height=0, width=0;
	pos cursor=0;
	string contents;
};

int main(int argc, char *argv[] ) {
	Screen myScreen(5, 5, 'x');
	myScreen.move(4, 0).set('#').display(cout);
	cout << "\n";
	myScreen.display(cout);
	cout << "\n" << endl;
	
	return 0;
}
```



动态内存分配，new

+  C++，使用 malloc 动态申请对象，不会调用构造和析构函数
+ new 先为对象再堆中分配内存，然后调用构造函数

  ```cpp
  类型 *p = new 类型; //int *p = new int; new 出来返回的是指针，可用于无参构造并调用析构函数
  int *p = new int[10]; //new 数组，返回首元素地址
  delete []p; //释放数组申请的空间;delete p;释放非数组申请的空间
  //new 和 delete 采用相同的类型
  ```
  ```cpp
  person *p = new person(1,"lucy");
  // new 时候调用有参构造
  ```

```
void *p = new person; //不会调用析构函数
//new 调用有参构造;new 对象的数组时不能调用有参构造
```

  + 类中可以使用静态成员

```cpp
public: 
  	static int b; //编译阶段就分配内存；不能再类内初始化，只能声明；存在静态全局区；所有类成员共享 1 个静态成员
  	static void fun()//可以访问静态成员变量，不能访问非静态成员变量
      const static int c = 100; //保存在常量区，不可修改，可以在类中初始化。
  int person::b = 10; //类的作用域，静态变量的访问定义 或通过对象访问静态成员函数: person p1; p1.b = 100;
person::c; //访问类中的一个成员变量
```

  + 单例模式：一个类只能有一个对象

  ```cpp
  class person{
      public:
      	static Person * instance(){return single;} //提供接口操作
      private:
      	static Person *single; //私有化静态指针 new 对象，先创建指针，后在类外定义
      	Person(){}//无参构造私有化
      	Person(const Person &p){} //拷贝构造私有化
  }
  Person *p = Person::instance();
p->age = 10;
  ```



+ 类成员的空间占用

  ```cpp
  class person{ 
      //空类至少为 1 个字节
  public:
  	int a; //4 bytes，占用对象空间
      static int b; //0 bytes 不存在类中，存在静态全局区
      void show(){} //0 bytes 不存在类中
      static void show1(){} // 不存在类中
  }
//函数存在代码区，
  ```

  + this(对象调用成员函数，会将对象地址传给 this 指针)

  ```cpp
  public：
  	Person(int a, int b){ this->a = a;this->b = b;}
  	person fun()const{} //常函数，不能通过 this 指针修改 this 指针指向的对象内容；const 修饰 const type * const this
    
  ```

  

## 友元：

+ 让 person 类中的 visit 函数可以访问 building

```cpp
class Building{
friend class Person;
} //类中所有的函数和成员都能访问 bilding 中的私有属性
或
class Building{
friend void person::visit();
}    //person 的 visit 函数能访问，其他不能

Person::person(
){
    
} //类外实现构造
```



+ 全局函数声明为类的友元，以访问私有属性

	```cpp
class Person{
      
  friend void fun(Person &b);
	...}
	
	```



### 运算符重载

给符号赋予新的意义，运算符只能运算内置的数据类型，如 char，int。所以我们可以做运算符重载

string 类相当于对字符串的操作进行了各种重载

```cpp
person p1;
person p2;
person p3 = p1 + p2;
//系统调用
operator+(p1,p2)
或
p1.operator(p2)

person operator+( person &p1, person &p2){
person p(p1.age + p2.age)
return p;
}
// 或类内实现
person operator+(person &p2){
    person p(this -> age+pe.age)
        return p;
}
```

+ 重载左移运算符( 右击 cout，转到定义可以查看函数定义)

```cpp
class person{
    friend ostream& operator<<(ostream &count,person &p);
    person& operator++(){
        this->num = this->num +1
            return *this;
    }
    private:
    int age;
}

ostream& operator<<(ostream &count,person &p){
    cout << p.age;
    return cout;
}

void test(){
person p1(10);
    cout << p1;
}
```

可以重载的运算符，但不要改变符号的意义，不能重载的又. :: .* ?: sizeof

+ 自加自减运算符重载

```cpp
++a; //operator++(p1)  pi.operator++()
a++;//p1.operator++(int)
```

+ =重载

```cpp
p2 = p; //p2.operator=(p1)

person& operator=(person &p1){
this->age = p1.age;
this->name = p1.name;
return *this;
} //系统自动生成的=重载只做简单的值拷贝；这样 p1 的 name 与 p2 的 name 指向同一个地址。在释放空间时会出错
person& operator=(person &p1){
this->age = p1.age;
this->name = new char[strlen(p1.name)+1];
    strcpy(this->name, p1.name);
return *this;
} 

```

+ 重载不等号

```cpp
class person
{
public:
    person(int age,string name1)
    {
        this->age = age;
        this->name = name;
    }
    bool operator==(person&p2)
    {
	return this-age == p2.age && this->name == p2.name;
    }
    int age;
    string name;

}
void test()
{
 person p1(10,"lucy");
    person p2(10,"bob");
    if (p1 == p2) //p1.operator==(p2)
    {
        cout << "p1 == p2" << endl;
    }
}
```

+ 函数调用符号（）重载，又名函数对象

```cpp
class myadd
{
public:
    int add(int a,int b)
    {
        return a+b;
    }
    int operator()(int x, int y)
    {
        return x + y;
    }
};

void test()
{
    myadd p;
    p(3,4); //p.opertor()(3,4)
	//或
    cout << myadd()(3,4)<< endl; //匿名对象 调用 operator（3，4）函数
}
```

+ 总结

=，[]，()，和->操作智能通过成员函数进行重载

<< 和>>只能通过全局函数配合友元函数进行重载

不要重载&&，||，不然短路规则可能会实现不了

所有的一元运算符建议使用成员函数重载

### 智能指针

定义一个对象，保存另外对象的地址，等出了函数自动释放内存。

```cpp
class smartpointer{
    public:
    Smartpointer(person *p1)
    {
        this-> p = p1;
    }
	~Smartpointer()
    {
        delete p;
    }
    person *p
}

void test(){
    person *p = new person(10);
    Smartpointer sp(p);
    cout << sp->age<<endl; //可以修改 sp.operator->()，让他返回 p 的地址
} //这样运行完程序就会自动释放 p

```

### 继承

子类（派生类）

```cpp
class Animal
{
public:
    int age;
protected:
    int b;
private:
    int c;
};

class Dog :public Animal
{
    public:
    int tail_len;
    void show(){
        //子类函数不能访问父类的 private 成员
    }
    /*
    public:
    int age;
protected:
    int b;
private:
    int c;
    */
} //公有继承方式，基类中是什么控制权限，继承到子类中也是什么控制权限

class B : protected Animal
{
public:
        /* 保护继承，将父类中共有的变成保护的，子类不能访问父类的 private 成员
protected:
 int age;
    int b;
private:
    int c;
    */
}
class C :private Animal
{
    public:
    int d;
            /* 私有继承，将父类中所有成员变成私有的，子类不能访问父类的 private 成员，看继承之前的权限

private:
    int c;
     int age;
    int b;
    */
}
```

+ 对象构造和析构调用原则

+ 创建子类对象时候，必须先创建出父类的对象，需要调用父类的构造函数

  ```cpp
  class son:public Base
  {
  public:
  son(int id, int age, string name): Base(age,name)
  {
  this->id = id;
  }
  int id;
  }
  //系统会先调用父类的构造函数，再是子类的构造函数。先析构
  //如果子类和父类又同名的成员变量或者函数，父类的成员会被隐藏，访问的是子类的成员
  ```

+ 继承中的静态成员特性，如果子类与父类有同名的静态成员变量或函数，父类的同名的静态成员变量或函数会被隐藏。

+ 多继承

```
class C:public A, public B
{}
```

+ 菱形继承和虚继承 

```cpp
class animal
{
    public:
    int age;
};
class sheep:virtual public animal //虚继承
{
public:
int id;
}; 
class camel:virtual public animal
{
public:
    int camel_num;
};

class shenshou:public sheep,public camel //这边的 age 只会从 animal 继承
{
public:
    int a;
}
void test()
{
    shenshou p;
    p.age = 100;  
}
```

+ cl /d1 reportSingleClassLayoutanimal file.cpp

虚继承后，类中储存 Virtural base ptr, 指向 VBtable

### 多态

一种接口多种形态

+ 父类有虚函数
+ 必须发生继承
+ 子类重写虚函数，参数，函数名要一致
+ 父类的指针或引用指向子类的对象

静态多态：编译时，调用的地址已经绑定（静态联编）

动态多态：运行时才确定需要调用的地址

```cpp
class animal{}
class dog:public animal{}
//如果两个类发生了继承，父类和子类编译器会自动转换
void dowork(animal &obj){
    obj.speak(); //地址早绑定
}
void test(){
    animal p1;
    dowork(p1);
    dog p2;
    dowork(p2);
}
//如果：
class animal{
    public:
    virtual void speak(){}        
}    
class dog:public animal{
    public :
        void speak(){} //重写虚函数，函数名一致
}
void dowork(animal &ogj){
    
    obj.speak(); //地址晚绑定，动态多态
}
```

+ 开发时 对源码的修改是关闭的 对扩展是开放的
+ 多态实现增添计算机功能

```cpp
class cal{
public:
    virtual int mycal(int a, int b){
        return 0;
    } 
};
class add: public cal
{
    public:
    int mycal(int a, int b){
        return a+b;
    }
};
int do_work(int a, int b, cal &obj){
    return obj.mycal(a,b);
}
void test(){
    add p1;
    do_work(3,4,p1);
}
```

+ 抽象基类和纯虚函数

有纯虚函数的类 叫做抽象类 抽象类不能实例出例子， 子类继承了抽象类，如果子类没有重写虚函数，那么这个子类也是抽象类

```cpp
virtual int mycal(int a, int b) = 0; //纯虚函数
```

### 虚析构函数

```cpp
void test(){
animal *p = new dog;
    p->speak();
        delete p; //这边调用的是父类的析构函数
}
```

如果再父类析构函数加上 virtual，变成虚析构函数，再调用基类的析构函数之前会先调用子类的析构函数。

+ 纯虚析构：虚析构函数等于 0。同纯虚函数

### 重写 重载 重定义

重载：

+ 函数名相同
+ 作用域
+ 桉树的各户，顺序，类型不一致
+ const 也可以称为重载的条件

重定义

+  发生继承
+ 子类和父类有同名的函数和变量父类中的同名的变量和函数都会被隐藏

重写

+ 父类中有虚函数
+ 发生了继承
+ 子类重写了虚函数，函数名返回值参数一致





