---
title: Java 基本
date: 2021-11-26
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- 计算机语言
mathjax: true
toc: true
comments: 笔记
---

> 常用 API 与语法 [官网 API 链接](https://docs.oracle.com/en/java/javase/17/docs/api/index.html)

<!--more-->

> Java 关键词：jvm java 虚拟机，应用服务器，高可用，高性能，高并发，web 开发，hadoop 大数据领域，android 手机端，可移植性（write once, run anywhere），高性能，分布式，动态性，javaSE（桌面，控制台），javaEE（web，服务器）

### JDK,JRE,JVM

java development kit - java, javac 等开发工具
java runtime environment 运行程序的环境
java virtual machine 运行程序的地方，不同的平台上有不同的 JVM 使得 java 可以做到一次编译，处处可用。

## 安装

跟着 [JDK 下载页面](https://www.oracle.com/java/technologies/downloads/) 操作即可。企业中通常使用 JDK8（最早的 LTS 版本）。安装后 bin 下的 java 为执行文件，javac 为编译文件

 **UBUNTU18.04+上安装 JDK11** 

`apt-get install default-jdk` 安装好后，`jave -version` 检查。
处理多个 java 版本问题：`update-java-alternatives --list` 显示已安装的版本，
`sudo update-java-alternatives --jre --set <java11name>` 指定 java 版本

 **工具使用 IDEA** 

文件架构：project （淘宝网站）- module（购物车） - package - class

快捷键：

> ctrl + D 复制当前行到下一行
> ctrl + X 删除所在行
> ctrl + ALT + L 格式化代码
> ALT +  SHIFT + up/down arrow 上下移动代码

导入模块：新建模块后，将 src 下的文件复制到新模块内

将 `.class` 文件拖放到 idea 界面上可以查看到反编译后的代码。或者使用 `javap xxx.class`

## helloword

编写源代码文件 `HelloWorld.java` HelloWorld 是公共的, 应在名为 HelloWorld.java 的文件中声明。编译 `javac HelloWorld.java`。运行 `java HelloWorld`；JDK 11 开始支持直接运行 java 源文件 `java HelloWorld.java`

## 语法

### 基础

#### 文档注释

```java
/**文档注释，可以自动被加入到 app 说明中
*/
```

 #### 进制

`0B` 二进制`0` 八进制`0x`十六进制                                                                                                                                                                                                                                                                                                                                                                                              

#### 类型转换

+ 小范围类型自动转换为大范围类型（自动补位）。`(int)3.14` 强制转换（舍弃高位）。
+ byte 表达式中作为 int 计算

#### 运算符

```java
System.out.println(a%10 + "你好" + b); // 'char' + int 为运算，"字符串" + int 为拼接
&& //短路与，|| 短路或
```

#### 输入

```java
import java.util.Scanner; // 导包并不需要自己操作，通过工具统一导入更方便
Scanner ss = new Scanner(System.in);
int a = ss.nextInt();  // String s = ss.next();
```

#### 控制语句

```java
if (){}
for (int i=0;i<1;i++){}  // idea 快捷键 collection.fori 
while (){}
do {} while()
```

#### 随机数

```
Random r = new Random();
int a = r.nextInt(10);
```

#### 数组

`int[] arr1 = arr2; ` arr1 与 arr2 指向同一块区域

```java
String[] name = new String[]{"zxc","zxc"} // 同 String name[]
int[] arr = new int[3];  // 
name[0] = "new"  // name.length
```

#### 方法

```java
public static int name(int[] a,int b){return 0;} // 参数 a 以引用形式工作
```

### 类

实际开发中，建议在对应的 `car.java` 文件中编写对应类。java 存在自动垃圾回收期，自动清理没有对应指针的内存区域。

```java
public class car{ // 对象存放在堆内存中，栈内存中储存对象的地址。
    String name;  //属性存放于对象中
    public void method(){}  //方法存放于方法区，对象中储存方法的地址。
}
car a=new car()
car b=a; // a,b 指向同一个对象
```

#### 构造

提供有参构造方法的话，需要手动写一个无参构造器才能使用无参构造。

```java
public class car{
    public Car(String name, double price){
        this.name=name;
        this.price=price;
    } // 提供有参构造的话
}
Car mycar = new Car("baoma",3.1);
```

#### 封装

封装是一种规范。赋值与取值通常使用 `setter` 与 `getter` 来访问。IDEA 中自动生成代码：右键代码- generate - getter and setter - 选择变量。通常不在 setter 与 getter 中对变量进行限制，而在交互端对输入变量进行控制。

```java
public class car{
    private String name;
    public String getName(){return this.name;}
    public void setName(String name){this.name = name;}
}
```

#### javaBean

成员变量使用 private 修饰；提供每一个成员变量对应的 setXxx() / getXxx()；必须提供一个无参构造器。

#### static

 **static 变量** 

变量只在内存中储存一份，储存于堆内存中的静态变量区，可以被共享、访问、修改。类中的静态成员变量建议使用 `类名.静态成员变量` 访问。

```java
public class User{
    public static int number;  //创建
}
User.number; // 类中静态变量，推荐使用类名访问
```

 **static 方法** 

无需访问到成员对象变量的，建议采用静态方法

```java
public class User{
public static void getMax(int a, int b){}
}
getMax();  // 可以直接调用静态方法，或者使用 类.静态方法
```

 **静态代码块** 

与类一起加载，自动触发，优先执行，只加载一次

```java
public class Test{
    static{
        // 静态代码块
    }
    {
        // 构造代码块，相当于写在构造函数类，用的比较少。
    }
}
```

 **饿汉单例设计模式。** 

在用类获取对象的时候，对象已经提前创建好了。

+ 构造器私有；定义一个静态变量储存一个对象。

```java
public class Car {
    public static Car mycar=new Car();
    private Car(){}
}  // Car.mycar 访问单例对象
```

 **懒汉单例设计模式** 

需要用到的时候再创建对象。

+ 构造器私有；定义一个静态变量存储一个对象；定义一个返回单例对象的方法；

```java
public class Car {
    private static Car mycar;
    private Car(){}
    public static Car getMycar() {
        if (mycar == null){
            mycar = new Car();
        }
        return mycar;
    }
}  //  Car car1 =  Car.getMycar(); 获取对象
```

#### 继承

```java
public class NewCar extends Car {}  //super.method() 调用父类方法
```

子类不能继承父类构造器
子类不能访问父类的私有方法、属性
一个类只能继承一个直接父类
不支持多继承，但是支持多层继承
私有方法、静态方法不能被重写

```java
@override
public void run(){} //重写方法时，访问权限应该大于父类方法
```

子类构造器默认先访问父类中的无参构造器，再执行自己。可以主动调用父类的有参构造器。

```java
public class NewCar extends Car {
    public Newcar(int price){
        super(price);
    }
} 
```

#### 包

不同包下的类，需要导包才能使用

```java
import com.xxxx.Car;
```

#### 权限修饰符

修饰符 - 可访问权限：

+ private - 同一类中
+ 不写 - 同一包下类中
+ protected - 同一个包下的类中，其他包下的子类
+ public - 任何地方

#### final

修饰方法 ，表示不能被重写；修饰变量，表示只能复制一次；修饰类，表示不能被继承

```java
final class Cat{}
public final void run(final doucle price){}  //无法在内部复制
public static final int age;  // 基本变量不能改变
public static final int[] arr;  // 不能改变指向地址，但可以改变地址所在值
```

#### 常量

`public static final` 修饰常量。常量建议使用全大写

#### 枚举类

多例模式；可以使用枚举做信息标志。

```java
public enum Move {
    UP, DOWN, LEFT, RIGHT
}  // Move.UP
```

#### 抽象类

某个父类实现了类中的基本方法（框架），具体方法细节在子类中实现。

```java
public abstract class Move {
    public abstract void run();
}

public class MM extends Move{
    @Override
    public void run() {//子类一定要完成重写}
}
```

#### 接口

JDK 1.8 之后，接口的成员只有常量和抽象方法。接口不能创建对象，需要用实现类创建对象。

```java
public interface Move {
    String name = "123";
    void run();  // 前面自动省略 public abstract 
}

public class MM implements Move{
    @Override
    public void run() {}
}  // 实现类可以实现多个接口
```

一个接口可以继承多个接口（没有冲突的前提下）。

接口新增方法：

```java
default void run(){}
static void run(){} // 通过接口.XXX 调用
private void run(){} //仅接口中调用，jdk9 以上支持
```

#### 多态

```java
Animal a=new Dog();
Animal a2=new Cat();
a.name;  // 实际提取 Animal 的属性
a.run(); // 实际调用 Dog 和 Cat 的方法
```

多态下不能访问独有功能，需要进行类型转换。

```java
Animal a2=new Cat();
if (a2 ubstabceif Cat){Cat c = (Cat)a2;}  // 强制类型转换
```

#### 内部类

 **静态内部类** 

可以直接访问外部公开成员，与普通类没区别，只是定义在类内部。

```java
public class outer{
	public static class inner{}
}  // outer.inner i = new outer.inner();
```

 **成员内部类** 

可以直接访外部类静态成员、外部类实例成员

```java
public class outer{
	public class inner{}
}  
outer.inner i = new outer().inner();
```

 **匿名内部类** 

方便创建子类对象，简化代码编写。可以作为形参进行传递。

```java
Animal a = new Animal(){
    public void run(){}
};  // Animal 为抽象类
```

#### 包装类

8 种基本数据类型对应的引用类型，如 `double` 对 `Double`，`int` 对 `Interger` ，`char` 对 `Chaacter`

自动装箱：`int a=0; Interger a2 = a;` 自动拆箱 `int b = a2;`

类型转换：`String rs = Interger.toString(123);` ，`Int a = Interger.valueOf("123");`

### Collection、MAP API

#### String

不可变字符串类型，修改的本质是改变指向目标。栈内存储存 string 的地址，字符串储存与字符串常量区。

 **创建** 

```java
String name = "abc";  // 指向常量池中"abc"所在地址。
String name =new String(chars)  // char[] chars={'a','b'} 非双引号出来的对象储存于堆内存
String name =new String(bytes)  // byte[] bytes={97,98}
```

```java
name.equals()  // 字符串比较不使用 ==
name.equalsIgnoreCase() 
    
name.length();
name.charAt();
char[] arr1 = name.toCharArray();
String name2 = name.substring(0,3); // 截取，包前不包后
String r3 = name.replace("pattern","new"); //不会修改 name
name.contains("subword")
name.startswith();
name.split("substr");
```

#### ArrayList

```java
ArrayList<object> l1 = new ArrayList<>();
l1.add("java");  //向末尾添加
l1.add(1,"java");  //插入元素，相应位置元素后移
ArrayList<Interger> l2 = new ArrayList<>();  // JDK 1.7 开始，泛型后面可以不写
```

```java
l1.get(2);  //索引
l1.size();
l1.remove(2);  //删除索引 2 变量，并返回对应值
l1.remove("apple");  //删除值为 apple 的第一次出现的对象，成功返回 True
l1.set(0, "new_value");  // 修改索引 0 的值，返回修改前的值
```

#### object

```java
class.toString() // 默认打印的是地址，重写快捷操作：ToString 回车自动补全。
class.equals()  // 默认比较地址是否相同，重写快捷方式 equals 回车自动补全。
```

#### StringBuilder

字符串拼接效率高，最终结果还是要用 String 接

```java
StringBuilder s = new StringBuilder("123");
String a = s.append("1").reverse().append("2").toString();
```

#### Math

`Math.abs()` `.ceil()` ;`.floor()`;`.pow(2,3)`;`.round()`;`.random() // 0.0- 1.0`

#### System

`System.exit(0)` 以 0 终止当前运行的 java 虚拟机; 
`long time = System.currentTimeMillis()`; 

#### BIgDecimal

解决浮点型计算精度失真问题，直接使用 `new BigDecimal(3.14)` 仍会有精度问题。建议使用：`BigDecimal r2 = BigDecimal.valueOf(3.14);` 
`double a = r2.doubleValue();`
`r2.divide(r3, 2, RoundingMode.HALF_UP)` 解决无限循环小数问题。 

#### Date

此刻时间： `Date d = new Date();`
获取毫秒值，做性能分析： `long time = d.getTime();`
时间毫秒值转日期： `Date d2 = new Date(time + 10000);` 或使用 `d.setTime(time)`

#### SimpleDateFormat

```java
SimpleDateFormat sd = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss"); 
Date d = sd.parse("2021-12-26 22:23:35");  // 从字符串获取日期
String s = sd.format(d)  // 可以格式化日期或时间毫秒值
```

#### Calendar

```java
Calendar rightnow = Calendar.getInstance();
int year = rightnow.get(Calendar.YEAR);  // 有 date of year 等信息
rightnow.add(Calendar.MINUTE, 50);  // 50 分钟后的时间，直接修改 `rightnow`
Date d = rightnow.getTime();
```

#### LocalDate、LocalTime、LocalDateTime

以上三个 API 类似
`LocalDate d = LocalDate.now();`
`LocalDate d = LocalDate.of(2020, 2, 23);`
`d.getDayOfYear()`

time 相关 API 可以转换为 `LocalDate`： `t1.toLocalDate()`

#### Instant

世界标准时间：`Instant instant = Instant.now();` 

#### DateTimeFormatter

```java
DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd");
LocalDate d2 = LocalDate.parse("2020-11-11",dtf);
```

#### period

提供时间段的信息 `Period d = Period.between(localdate1, localdate2);`

#### Duration

类似 Period：`Duration.between(date1, date2)`

#### Arrays

`int[] arr = {1,2,3};` 转字符串：`Arrays.toString(arr);`

排序：`Arrays.sort(arr);` 直接修改 `arr`，使用比较器只支持引用类型的比较：

```java
Move[] moves = new Move[3];
for (int i=0;i<3;i++){
    moves[i] = new Move(12+2*i);
}
Arrays.sort(moves, new Comparator<Move>() {
    @Override
    public int compare(Move o1, Move o2) {
        return o2.age - o1.age;  //降序
        // return Double.compare(o1.price, o2.price)
    }
});
// Arrays.sort(moves,(Move o1, Move o2)-> o2.age - o1.age);

System.out.println(Arrays.toString(moves));
```

二分搜索：`Arrays.binarySearch(arr, target);` 返回索引值

 #### Collection

集合中智能储存引用类型数据 ArrayList, LinkedList, HashSet, 

`arr.add()` , `.isEmpty() `,  `.clear()`,  ` .size() ` , `.contains()` 包含某元素 , `.remove("123")` 默认删除第一个匹配到的值, `.addAll(a)` 添加 `a`  集合中所有元素。

集合迭代器：`Iterator<Integer> i = arr.iterator(); ` ，查看是否有下一个 `i.hasNext()`， 进入下一个元素位置并返回当前元素值`int b = i.next();` 也可以采用 `for (Integer integer : arr) {}`；`arr.forEach(s->System.out.println(s));`

 **list** 

可用的有 arraylist（动态数组） 与 linkedlist（双链表）；常见方法有`list.add(2,"item)`, `list.remove(idx, "item)`, `list.get(2,"item")`, `List<> a = list.sublist(beg,end)`

`LinkedList<int> stack = new LinkedList<>();`; 

`stack.addFirst();` = `stack.push();`,
`stack.getFirst();`
`stack.removeFirst();`= `stack.pop();`

并发修改问题（边遍历，边修改）：使用迭代器的 `it.remove();` ；或使用 for 指针循环。

 **set** 

`Set<int> sets = new HashSet<>();` 无序，不重复，无索引。JDK 7 前，哈希表由数组 + 链表组成，初始化生成一个数组，使用哈希值模数组的大小作为插入位置算法，链表解决冲突问题。JDK8 开始，哈希表底层采用数组+链表+红黑树实现，当链表长度超过 8 时候，自动更换为红黑树。
`LinkedHashSet<>();` 有序，不重复，无索引 。底层依旧使用哈希表，知识每个元素额外多了一个双链表的机制记录储存的顺序。

`xxxx.hashCode()` 可以计算哈希值。set 集合基于哈希值去重复。set 中的类想要达到去重效果的话，需要重写 `hashCode()` 方法和 `equals()` 方法。

`TreeSet<>()` 不重复，无索引，可排序；字符串通过 ASCII 排序。自定义对象通过实现 comparable 接口实现排序。优先使用 TreeSet 提供的 comparable 比较器，再使用类中的比较器。 

```java
public class Card implements Comparable<Card>{
public int compareTo(Card o){
return this.weight- o.weight >= 0 ? 1: -1;
}  // 类中实现比较器
}
```

#### Collections 工具

```java
List<Integer> arr = new ArrayList<>();
Collections.addAll(arr, 1,2,3,4,5);
Collections.shuffle(arr);  //shuffle(List<?> list)
Collections.sort(arr);  // sort(List<?> list, Comparator<? super T>) 对自定义类排序，需要提供比较器
```

#### Map

键值类集合

map 集合 API 有：`.put(key, value);`, `clear();`, `get(key);` 不存在返回 `null`, `remove(key);` 返回对应键的值, `.isEmpty();`, `containsKey(key);` , `.containsValue(value);` , 获取全部键的集合：`Set<K> k = maps.keySet();`, `Collection<V> v = maps.values()`, `maps.putAll(map2);` 将 `map2` 的元素复制到 `map` 中。将 Map 中的值 + 1 `maps.put(key,maps.get(key)+1);`

遍历：

```java
for (String key:maps.keySet()){maps.get(key);}

Set<Map.Entry<String , Integer>> entries = maps.entrySet();
for (Map.Entry<String ,Integer> entry:entries){
    String s = entry.getKey();
    int v = entry.getValue();
}
maps.forEach((k, v)-> System.out.println(k + v));  // BiConsumer


```

`HashMap(); ` 底层为哈希表，采用键计算哈希值。`LinkedHashMap` 有序，不重复，无索引。底层为哈希表 + 储存顺序的双链表。`TreeMap` 只能对键排序，原理与 `TreeSet` 相同。

#### Properties

系统配置信息读写，格式为 字符串=字符串 的键值对

```java
Properties properties = new Properties();
properties.setProperty("admin","123456");
properties.store(new FileWriter("./test"),"comment");  // 写

Properties p2 = new Properties();
p2.load(new FileReader("./test")); // 读 p2.getProperty
```

### 拓展

#### re

```java
string.matches("pattern"); //如果匹配到，返回 true
String rs = "123asdfasdf123asd";
String re = "\\d\\d";
Pattern pattern = Pattern.compile(re);
Matcher matcher = pattern.matcher(rs);
while (matcher.find()){
    String r = matcher.group();
    System.out.println(r);
}
```

#### lambda

简化函数接口中只有一个抽象方法的匿名内部类。

```java
public interface Outer{
    void run();
}
Outer o1 = ()->System.out.println("234");
o1.run();
```

#### 泛型

jdk 5 引入的特性，可以在编译阶段约束数据类型。形式为 `<>` 如 `List<>`。定义时候使用 `E,T,K,V`

泛型类：`public class MyArrayList<T>{}` , 
泛型方法的定义：`public <T> void show(T[] arr){} ` 可以接收任意类型的数组  
泛型接口：

```java
public interface Outer<E>{
    void run(E e);
}
public class Move implements Outer<String>{
    @Override
    public void run(String s) {}
}
```

通配符，在函数使用的时候提供多类型支持

```java
public void go(ArrayList<?> cars){}  // 支持所有类型
public void go(ArrayList<? extends Car> cars){}  // 支持 Car 及其以下的类
public void go(ArrayList<? super Car> cars){}  //支持 Car 及其以上的类
```

#### 可变参数

可以传递多个参数，一个方法只能使用最多一个可变参数，可变参数必须放在最后面.

```java
public static void sum(int...nums){ 
    for (int i:nums){System.out.println(i);}  //方法中作为数组使用
}
```

#### 不可变集合

`List, map， Set` 等加 `of` ，如：  `List<Double> l = List.of(1.2,2.2);`。 

#### Stream 流

流只能遍历一次

```java
l.stream().filter(s -> s > 1.0).forEach(s-> System.out.println(s));
```

获取：Collection 使用 `Stream<Double> s = list.stream()`, `maps.keySet().stream()`, `maps.entrySet().stream()`, 数组使用 `Arrays.stream(str[])` 或使用 `Stream.of()`
中间方法：`filter(lambda)`, `.limit(3)`, `.skip(2)`, `.map(lambda)`, `.distinct()` `.max(lambda)`
合并流： `Stream.concat(stream1, stream2)`
终结方法：`.foreach()`, `.count()` 等不反回流的方法。
收集 stream 流到集合或数组中：`List<Double> d = s.collect(Collectors.toList());` 得到的集合可变，或者 JDK 16 开始直接 `s.toList()`, `s.toArray()` 得到不可变的集合。

#### 异常处理

Throwable = Error 不可处理的系统/硬件异常 +  Exception 代码异常

 **编译时异常**  - 继承 Exception 的异常或者其子类

+ 如 `ParseException` 等
+ `throws` 处理：`throws ParseException, 异常 2, 异常 3..`  或 `throws Exception`

 **运行时异常**  - RuntimeException

+ 如 `ArrayIndexOutOfBounds`, `NullPointer` , `ClassCast`, `Arithmetic`等
+ 处理：

```java
try{...
}catch (Exception e){
    e.printStackTrace();
}
```

 **自定义异常**  

使用 `throw new` 跑出，通常情况下使用 `RuntimeException`，若使用编译时异常，调用对应方法时候需要 `throws` 处理编译异常。

```java
public class MyException extends Exception{
    public MyException(){}
    public MyException(String message) {
        super(message);
    }
}
public static void run() throws MyException {
    throw new MyException("message");
}
```

#### 日志

日志实现框架 `Log4j`, `JUL`, `Logback` 等

 **Logback** [官网](https://logback.qos.ch/)

[slf4j-api](https://www.slf4j.org/): 日志规范

Logback-core：基础模块
logback-classic：log4j 的一个改良版，同时完整的实现了 slf4j API

快速入门

+ 项目下建立 lib 文件夹，放入下载好的 slf4j, logback-core, logback-classic 三个 jar 包，并添加到项目依赖中。idea 中右键 - add as library 

+ 将 logback 核心配置文件 logback.xml 直接拷贝到 src 目录下。来源：[黑马程序员-java 基础]()

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <!--
        CONSOLE ：表示当前的日志信息是可以输出到控制台的。
    -->
    <appender name="CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
        <!--输出流对象 默认 System.out 改为 System.err-->
        <target>System.out</target>
        <encoder>
            <!--格式化输出：%d 表示日期，%thread 表示线程名，%-5level：级别从左显示 5 个字符宽度
                %msg：日志消息，%n 是换行符-->
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%-5level]  %c [%thread] : %msg%n</pattern>
        </encoder>
    </appender>

    <!-- File 是输出的方向通向文件的 -->
    <appender name="FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n</pattern>
            <charset>utf-8</charset>
        </encoder>
        <!--日志输出路径-->
        <file>./code/my_data.log</file>
        <!--指定日志文件拆分和压缩规则-->
        <rollingPolicy
                class="ch.qos.logback.core.rolling.SizeAndTimeBasedRollingPolicy">
            <!--通过指定压缩文件名称，来确定分割文件方式-->
            <fileNamePattern>./code/my_data-data2-%d{yyyy-MMdd}.log%i.gz</fileNamePattern>
            <!--文件拆分大小-->
            <maxFileSize>1MB</maxFileSize>
        </rollingPolicy>
    </appender>

    <!--
    level:用来设置打印级别，大小写无关：TRACE, DEBUG, INFO, WARN, ERROR, ALL 和 OFF
   ， 默认 debug
    <root>可以包含零个或多个<appender-ref>元素，标识这个输出位置将会被本日志级别控制。
    -->
    <root level="ALL">
        <!--只打印配置了的类型-->
        <appender-ref ref="CONSOLE"/>
        <appender-ref ref="FILE" />
    </root>
</configuration>
```

+ 创建 logback 日志对象

```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public static Logger LOGGER =  LoggerFactory.getLogger("Test.class");
LOGGER.debug("this is debug");  // .info() .error()
```

 **日志级别** 

TRACE < DEBUG < INFO < WARN < ERROR；默认 DEBUG。只输出不低于设定级别的日志。`OFF` 为全部关闭，`ALL` 为全部开启。

#### FILE

创建对象：`File f = new File("/mnt/together/nut/MITB/email_key.txt");` 支持绝对与相对路径

查询方法：`f.length()` 返回字节大小, `f.exists()`, `isDirectory()`, `isFIle()`, `getAbsolutePath()`, `getName()`, `getPath()`, `lastModified()` 返回时间毫秒值, `File[] files = f.listFiles();` 

修改方法：`createNewFile()`基本上不用, `mkdir()` 只能创建一级文件夹, `mkdirs()`,  `delete()` 删除文件或者空文件夹

#### 字符集

ASCII： 128 位字符信息。

GBK（ANSI）： 兼容 ASCII 码表，windows 中文系统默认码表，包含几万个汉字。一个中文以两个字节表示。

Unicode：统一码，万国码。UTF-8 一个中文一般以三个字节表示，也兼容 ASCII 编码。

所有编码里，英文都是一个字符，兼容 ASCII。

编码：`byte[] bytes = name.getBytes();`  默认 UTF-8 编码
解码：`String name1 = new String(bytes, "GBK");`

### IO 流

#### 字节流

 **InputStream** 

```java
InputStream is = new FileInputStream("./test"); //is.read() 默认读一个字节
byte[] buffer = new byte[3];
int b = is.read(buffer);  // 读取 3 个字节到 buffer 中，b 为 buffer 长度，b==-1 为读取完毕
String rs = new String(buffer,0,b); // 解码

byte[] b1 = is.readAllBytes();
System.out.println(new String(b1));
```

 **OutputStream** 

```java
OutputStream os = new FileOutputStream("./test");  // 追加的话，在后面多加参数 true 
byte[] b = "你好".getBytes(StandardCharsets.UTF_8);
os.write(b);  // 写入桶
os.write('a');  // 写入 1 个字节到缓存
os.flush(); // 刷新缓存，存入数据。
os.close(); // 写完关闭
```

拷贝文件

```java
InputStream is = new FileInputStream("./test");
OutputStream os = new FileOutputStream("./1");
byte[] buffer = new byte[1024];

int len;
while ((len = is.read(buffer)) != -1){
    os.write(buffer,0, len);
    os.flush();
}
os.close();
is.close();
```

优化资源释放：在将`.close()` 放在`try{}catch{}finally{.close()}` 中。关闭前需要进行非空校验

```java
InputStream is = null;
OutputStream os = null;

finally{
    if (os != null)os.close
}
```

JDK 7 后可以自动关闭资源（实现了 Closeable 接口）：

```java
try(
    InputStream is = new FileInputStream("./test");  // 放置需要自动关闭的资源
    OutputStream os = new FileOutputStream("./1");
)
{...}catch(...){...}
```

####  **字符流** 

 **Reader** 

```java
Reader fr = new FileReader("./test");
int code = fr.read();  // System.out.println((char)code);  code == -1 表示结束

char[] buffer = new char[10];
int code = fr.read(buffer);  // new String(buffer,0,code);
```

 **Writer** 

```java
Writer fw = new FileWriter("./test");  // 默认覆盖源文件，添加 true 参数改为增添数据
fw.write(123);  // .write('a'); .write("可以写字符串") 
fw.write("写入部分字 123",0,5);  // 写入 "写入部分字"
fw.flush();
```

#### 缓冲流

`BufferedInputStream`, `BufferedOutputStream` 使用较多。缓冲流速度在桶小（约 1kb）情况下快很多，原始流使用（8kb）桶性能有相对的提高。

```java
InputStream is = new FileInputStream("./test");
InputStream bis = new BufferedInputStream(is);  // 使用方法与 InputStream（被包装方法）相同
```

`BufferedWriter`, `BufferedReader` 缓冲字符输入流新增按行读取

```java
Reader f = new FileReader("./test");
BufferedReader fr = new BufferedReader(f);
String line = fr.readLine();  // while (line != null)
```

#### 转换流

 **InputStreamReader** 

```java
InputStream is = new FileInputStream("./test");
Reader isr = new InputStreamReader(is, "GBK");
BufferedReader br = new BufferedReader(isr);
```

 **OutputStreamReader** 

```java
OutputStream os = new FileOutputStream("./test");
Writer ofw = new OutputStreamWriter(os, "GBK");
BufferedWriter bw = new BufferedWriter(ofw);
```

#### 序列化对象

 **对象序列化** 

```java
OutputStream os = new FileOutputStream("./test");
ObjectOutputStream oss = new ObjectOutputStream(os);
oss.writeObject(a);  
```

对象要序列化，需要实现 Serializable 接口。对类中不需要序列化的属性，使用 `private transient String password;` 序列化需要申明版本号 `private static final long serialVersionID = 1;`  确保序列化前后数据版本一致。

 **反序列化** 

```java
ObjectInputStream ois = new ObjectInputStream(new FileInputStream("./test"));
List<Integer> b = (List<Integer>) ois.readObject();
```

#### 打印流

PrintStream 更高效的写数据到文件，PrintWriter 使用方法、功能相似。

```java
PrintStream p = new PrintStream("./test");  // 相当于 new PrintStream(new FileOutputStream("./test"))
p.println(97);  // 支持打印可视化的 List 等
```

重定向，将控制台输出输出到文件：`System.setOut(p);`

#### commons-io

[官网下载](https://commons.apache.org/proper/commons-io/download_io.cgi) commons-io-2.11.0-bin 文件

复制：`IOUtils.copy(new FileInputStream("./test"),new FileOutputStream("./new_test"));`
删除：`FileUtils.delete(new File("./test"));`
等

### 多线程

#### Thread

编写简单，不利于扩展。

```java
class MyThread extends Thread// {重写 run 方法}
MyThread m = new MyThread();
m.start();  //启动线程，不要把主线程任务放到子线程前。
```

`m.setName` 设置线程名，主线程名称为 main；`m.getName()` 获得当前进程名称；`Thread m2 = new Thread.currentThread()` 。可以重写 `MyThread` 的有参构造器，调用父类 `Thread`  的构造器修改线程名字：`public void MyThread(String n){super(n);}`。在建立对象时候起名字： `Thread t = new Thread(m,"线程名字");` 实际应用中，通常使用线程默认名字。

让当前线程休眠 3 秒：`Thread.sleep(3000);` 

#### Runable

可以继承其他类，利于拓展。线程执行后没有返回结果。

```java
class MyRunable implements Runnable // {重写 run 方法}
Runnable m = new MyRunable();
Thread t = new Thread(m);
```

#### callable

```java
class MyCallable implements Callable<String> {
    @Override
    public String call() throws Exception {
        return "";
    }
}

Callable<String> callable = new MyCallable();
FutureTask<String> f = new FutureTask<>(callable);
Thread t = new Thread(f);
t.start();
System.out.println(f.get());  // 通过 .get()  获得线程执行完毕后的结果。
```

#### 线程同步

同步代码块 `synchronized(锁对象){被锁代码块}` 建议使用共享资源作为锁对象，对实例方法，可以使用对象 `this` 。对静态方法，可以使用 `类名.class`。

同步方法 `public synchronized void method(){}` 默认使用 `this` 作为锁对象。同步方法需要方法高度面向对象。

#### Lock

```java
class MyCallable  {
    private final Lock lock = new ReentrantLock();
    public void run(){
        lock.lock();
        try{
            System.out.println("code");
        }
        finally {
            lock.unlock();
        }
    }    
}
```

#### 线程池

临时线程当任务队列满了，并且核心线程都在忙时候创建。当临时线程、任务对流、核心线程都满了，开始拒绝任务。

```java
public ThreadPoolExecutor(int corePoolSize,
 int maximumPoolSize,
 long keepAliveTime,
 TimeUnit unit,
 BlockingQueue<Runnable> workQueue,
 ThreadFactory threadFactory,
 RejectedExecutionHandler handler)
```

自定义线程池 - runable

```java
ExecutorService pool = new ThreadPoolExecutor(3,5,6,TimeUnit.SECONDS,
new ArrayBlockingQueue<>(5),
Executors.defaultThreadFactory(),
new ThreadPoolExecutor.AbortPolicy());

Runnable r = new MyRunnable();

pool.execute(r); // 该例中，第 9 个线程开始启用临时线程
pool.submit(new MyCallable()).get())  // callable
pool.shutdownNow();  //立即关闭线程池，数据可能丢失
pool.shutdown();  // 执行后关掉，一般不会去关线程池
```

建议使用自定义的线程池。`ExecutorService pool = Executors.newFixedThreadPool(3);`  任务队列长度不受限制，可能出现 OOM 错误。` Executors.newScheduledThreadPool` 创建允许创建线程数量不受限，大量线程可能导致 OOM。

#### 定时器

相关 API：`Timer`, `ScheduledExecutorService`。后者使用线程池，相对比较安全。

```java
ScheduledExecutorService pool = Executors.newScheduledThreadPool(3);
pool.scheduleAtFixedRate(new TimerTask() {
    @Override
    public void run() {
        System.out.println("task");
    }
}, 0,2,TimeUnit.SECONDS);
```

线程的 6 中状态：New, Runnable, Teminated, Blocked, Waiting（等待唤醒）, Timed Waiting

## 技术概述

### 网络通讯

[计算机网络笔记](http://wujiawen.xyz/2021/04/02/TCPIP1/)

 **IP 操作 API：InetAddress** 

提供域名方式获取`InetAddress ip2 = InetAddress.getByName("wujiawen.xyz");`
基本方法：`ip2.getHostAddress()`, `ip2.getHostName()`, `ip2.isReachable(5000)`

 **UDP** 

接收

```java
DatagramSocket socket = new DatagramSocket(8888);
byte[] buffer = new byte[1024 * 64];
DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
socket.receive(packet);
packet.getSocketAddress(); // packet.getPort()
```

发送

```java
DatagramSocket socket = new DatagramSocket();
byte[] buffer = "message".getBytes(StandardCharsets.UTF_8);
DatagramPacket packet = new DatagramPacket(buffer, buffer.length, InetAddress.getLocalHost(),8888);
socket.send(packet);
```

广播发送使用 `DatagramPacket packet = new DatagramPacket(buffer, buffer.length, InetAddress.getLocalHost(),8888);`；组播使用 `MulticastSocket`

 **TCP** 

`java.net.Socket` 底层使用 TCP 协议。服务器一个线程只能建立一个客户端的通信。

客户端 - 发送消息

```java
Socket socket = new Socket("127.0.0.1",7777);
OutputStream os = socket.getOutputStream();
Scanner sc = new Scanner(System.in);
PrintStream ps = new PrintStream(os);
while (true){
    String msg = sc.nextLine();
    ps.println(msg);
    ps.flush();

}
```

服务端 - 接收消息

```java
ServerSocket serverSocket = new ServerSocket(7777);

while (true){
    Socket socket = serverSocket.accept();
    System.out.println(socket.getRemoteSocketAddress() + "上线了");

    new ServerThread(socket).start();
}

// ServerThread
InputStream is = socket.getInputStream();
BufferedReader br = new BufferedReader(new InputStreamReader(is));

String msg;
if ( (msg = br.readLine()) != null ){
    System.out.println(socket.getRemoteSocketAddress());
    System.out.println(msg);
```

需要对架构进行优化，如使用线程池等；若要实现客户端的发送与接收功能，可以在服务器端使用 `List` 或 `Map` 储存客户端 `socket` ，再检索 `Map` 发送消息。[BS 框架概述视频](https://www.bilibili.com/video/BV1Cv411372m?p=185)

### 单元测试

针对最小的功能单元编写测试代码，即对 Java 的方法进行测试。通常使用 JUnit 开源测试框架进行测试，可以做到一键测试全部方法，生成测试报告，单元测试中的某个方法失败不影响其他测试等。

#### JUnit

IDEA 通常整合了 Junit 框架。如果没有整合，需要导入 `junit-xx.jar`，`hamcrest-core-xx.jar` 。编写以下的测试类，在 IDEA 中可以直接右键导航栏 -> `run all test`，

```java
public class test {
    @Test
    public void test1(){
        Mycall call = new Mycall();
        int result = call.demo1();
        Assert.assertEquals(result + "should be 1",1,result);
    }
    
    @Test
    public void test2(){
        Mycall call = new Mycall();
        call.run();
    }
}

class Mycall{
    public int demo1(){return 1;}
    public void run(){System.out.println(10 / 0);}
}
```

JUnit 4： `@Before`，`@After` `@BeforeClass` `@ AfterClass`

### 反射

 **获取类对象使用：** 

`Class c = Class.forName("xyz.wujiawen.hello2.test");`
`Class c3 = test.class;
test t = new test();` , `Class c2 = t.getClass();`

 **提取类中构造器对象：** 

公开构造器：`Constructor[] constructors = c.getConstructors();` 
全部构造器：`Constructor[] constructors = c.getDeclaredConstructors();`
无参构造器：`Constructor constructor = c.getConstructor();`
通过参数传递，获取有参构造器：` c.getDeclaredConstructor(int.class);`

 **通过构造器创建对象：** 

遇到私有构造器，打开权限：`constructor.setAccessible(true);`
得到构造器对象：`test t = (test) constructor.newInstance();`

 **获取成员变量：** 

获取全部：`Field[] fields = c.getDeclaredFields();` 取值：``fields[0].getName()`, `fields[0].getType()`
根据名称获取：`Field field = c.getDeclaredField("a");` 
给 `test t` 赋值`field.set(t,18);` 
取值：`field.get(t)`

 **获取方法对象：** 

全部方法：`Method[] methods = c.getDeclaredMethods();` 获取单个方法用 `getDeclaredMethod()`  传入方法名获取对应方法。
出发方法：`method.invoke(s,"args");`

 **发过编译阶段为集合添加数据** 

```java
ArrayList<Integer> a = new ArrayList<>();
Class c = a.getClass();
Method add = c.getDeclaredMethod("add", Object.class);
add.invoke(a,"args");
```

也可以使用 `ArrayList b = a;  b.add("others")`

提供通用框架，支持保存所有对象信息：

```java
try{
    PrintStream ps = new PrintStream(new FileOutputStream("./test.txt"),true);
    Class c = obj.getClass();
    ps.println(">>>" + c.getSimpleName() + ">>>");

    Field[] fields = c.getDeclaredFields();
    for (Field field:fields){
        field.setAccessible(true);
        String name = field.getName();
        String value = "" + field.get(obj);
        ps.println(name + "=" + value);
    }
}catch (Exception e){
    e.printStackTrace();
}
```

### 注解

自定义注解：`public @interface test {String name(); int age() default 12;}` 
有多个属性未提供默认值的情况下，注解属性名称必须提供：`@test(name="fillin",age=1)`

元注解：`@Target` 约束自定义注解的使用地方，`@Target({ElementType.METHOD,ElementType.FIELD})`
只能注解方法与成员变量；`@Retention` 申明注解生命周期：`@Retention(RetentionPolicy.RUNTIME)` 在运行时仍然生效。

注解解析：

```java
@Test
public void parseClass() throws NoSuchMethodException {
    Class c = bookstore.class;
    Method m = c.getDeclaredMethod("run");

    if(c.isAnnotationPresent(test.class)){
        test t = (test) c.getDeclaredAnnotation(test.class);  
        // m.getDeclaredAnnotation
        System.out.println(t.name());
    }
}
```

注解应用场景 - 有注解的方法才执行：遍历所有方法，使用注解解析检测该方法是否有加注解。使用反射触发被注解方法。

### XML

纯文本，默认使用 UTF-8 可嵌套，可用浏览器查看。常用与数据传输、软件配置等。

抬头声明 `<?xml version="1.0" encoding="UTF-8" ?>` 用于识别 xml 文件。
标签格式：`<name id=1></name>`，必须且只能存在一个本标签
注释：`<!-- comment -->`  
特殊字符：小于`&lt;`大于 ` &gt;` 与 `&&` 使用：` &amp;&amp`。字符数字区：`<![CDATA[select * from ...]]>`

 **文档约束：** 

DTD：编写 DTD 文档，后缀必须是 .dtd；将编写的 XML 文件导入到 XML 中 `<!DOCTYPE XX SYSTEM "data.dtd">` ；不能约束具体的数据类型，可约束 XML 文件的编写。

schema：本身是 XML 文件；编写 schema 约束文档，后缀 .xsd；导入 schema 文档 

```java
<书架 xmlns="http://www.itcase.cn"
    xmlns:xsi="http://xxxxxxxxx"
        xsi:schemaLocation="xxxxxxx data.xsd"
    >
</书架>
```

 **XML 解析：** 

dom4j [官网](https://dom4j.github.io/) 解析文件大致格式为：`Document{Element 标签{Attribute:Text}}`。 `Element`, `Attribute` 和 `Text` 均为 Node 对象。

`SAXReader saxReader = new SAXReader();` 将文件置于 src 文件夹读取：  `InputStream is = Dom4app.class.getResourceAsStream("/filename.xml");` 。解析文件：`Document document = saxReader.read(is);`  获取文本等`document.getText();`

 **Xpath** 

使用 Xpath，带入 Dom4j 与 jaxen.jar。通过 Dom4j 获取 Document 文件，利用 Xpath 完成选取 XML 文档元素节点。
通过 Xpath 检索：`List<Node> nodes = document.selectNodes("xpath")`，xpath 路径如：
绝对路径：`/根元素/子元素/属性名/等等`
遍历路径下全部子路径：`路径//目标元素`
锁定元素：`//@id` ，查询 name 元素包含 id 属性的 `//name[@id=8]`
`Element ele = (Element) node; `



