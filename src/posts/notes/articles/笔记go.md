---
title: Go 基础
date: 2021-12-12
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- 计算机语言
mathjax: true
toc: true
comments: 笔记
---

> 对于高性能分布式系统领域而言，Go 语言无疑比大多数其它语言有着更高的开发效率。它提供了海量并行的支持，这对于游戏服务端的开发而言是再好不过了。
> 关键词：快速、并行。
>
> [中文 golang 文档](https://studygolang.com/pkgdoc), [Go 文档](https://go.dev/doc/)

<!--more-->

## 安装

[官方教程](https://go.dev/doc/install)：下载 `go1.17.6.linux-amd64.tar.gz` ，解压到 `/usr/local` 。配置环境路径 `export PATH=$PATH:/usr/local/go/bin`

IDE：[GoLand](https://www.jetbrains.com/go/)， LiteIDE 等 

## 基础语法

`hello.go` 文件架构。源码内容必须使用 `UTF-8` 格式编写。必须在源文件中非注释的第一行指明这个文件属于哪个包，运行程序必须有一个 main 包，且一个工程文件夹只能有一个 main 包

```go
package main  
import "fmt"  

func main() {  // {不能放单独一行
   /* 这是我的第一个简单的程序 */
   fmt.Println("Hello, World!")
}
```

直接运行代码 `go run hello.go`。使用 `go build hello.go` 生成 二进制文件。

#### 变量

声明变量 `var a int`，变量声明后必须使用。
赋值 `a=10`
声明并赋值：`var c = 1` 或使用自动推导类型：`c, b := 30, 20` 不需要声明。Go 支持多重赋值：`c,b=b,c`，同时支持匿名变量 `c,_ = run()`
多个全局变量可以一起声明：

```go
var(
a int
b bool
)
```

#### 数据类型

整型 `%d`：`int16`, `uint16`
浮点型 `%f`：`float32`, `float64`, `complex64`, `complex128`
布尔型：`bool`
字符串 `%s`：`string`
字符 `%c`：`uint8`或`byte` 代表 ASCII 字符，`rune`  代表 UTF-8 字符
其他：`Pointer`, `Channel`, `interface`, `struct`, `Map`

`bool` 不支持类型转换，类型转换使用：`int(a)`

类型别名：`type bigint int64` 给 `int64` 取别名

#### 常量

定义格式：`const a int = 10` 其中类型可省略。
支持表达式定义：

```go
const(
	a = "abc"
    b = len(a)
    c = unsafe.Sizeof(a)
)
```

第一个 iota 等于 0，每当 iota 在新的一行被使用时，它的值都会自动加 1。同一行的 iota 值相同

```go
const (
    a = iota   //0
    b          //1 使用上一行表达式
    c          //2
    d = "ha"   //独立值，iota += 1
    e          //"ha"   iota += 1
    f = 100    //iota +=1
    g          //100  iota +=1
    h = iota   //7,恢复计数
    i          //8
)
```

#### 运算符

位运算符：`&` 与， `|` 非， `^` 异或，`<<`  左移 
支持赋值运算符：`c |= 2`
其他：`&` 取地址，`*` 取值

#### 逻辑语句

 **条件语句** 

支持 1 个初始化语句，初始化和判断以分号分割 `if c:= 1;c == 1 {}`

```go
if c == 1 {
} else if c == 2 {  // else 需要在 } 后面
} else {
}
```

 **循环语句** 

并没有 while 语句

```go
for d:=1;d<5;d++{}
for true {}
for i, val := range str{}  // 使用迭代器 i 为 索引，val 为值
```

支持 `break`,  `continue`, `goto`

```go
goto label
label: statement   //跳转到改行执行
```

#### 数组

数组声明：`var name [10] int` 可以用 `...` 代替长度
初始化：`var a = [2]float32{1.2,1.1}` 或 `b := [...]int{1,2}`
索引或修改：`b[1] = 0`
向函数传递数组使用：`void fun(param []int)`
多维数组：`var c [2][2][2]int`
初始化多维数组：`c:=[2][2]int{{0,1},{1,1}}`

获取命令行参数：`list := os.Args`，如`go run hello.go 12 34` 命令， list 中储存 `[命令执行文件地址 12 34]`

#### 指针

`var a *int` 整型指针。
空指针判断：`a == nil`，必须有合法指向才可以操做指针：`p = &a`, ` *p = 100`
可以使用 new 申请可操作空间 `var p *int`, `p = new(int)`, `*p = 6`。或者使用 `q := new(int)`

指针数组：`var ptr [4]*int`，数组指针需要取值 `*ptr[4]` 使用。

```go
a := [2]int{1,1}
var ptr [2]*int
ptr[0] = &a[0]
```

#### 结构体

定义结构体

```go
type Books struct{
	title string	
    author string
}
```

初始化：`var b Books = Books{xx,xx}`
可以查看或修改结构体信息：`b.title = "123"`

 **结构体指针**  
指针有合法指向后，才能才做成员。需要定义普通结构体后，再建立指针指向结构体。
初始化：`var struct_ptr *Books`
指向结构体：`struct_ptr = &p` 或 `struct_ptr := new(Books) `
调用及修改成员：`struct_ptr.title`
函数传参：`func a(book *Book)`

结构体可以进行 `==` 比较。同类型结构体可以相互赋值。

#### 可见性

如果想使用别的包的函数、结构体类型、结构体成员，那么变量名称必须大写开头，可见。如果是小写，只能在同一个包里使用。

#### 切片

可使用 make 初始化：`var a []int = make([]int,len)`
可以对数组进行截取接片 `s := a[0:2]`
空切片为 `a == nil`

#### 集合

map 通过 hash 实现，是无序的键值对集合

`var mymap map[string]string` 声明变量，默认 map 是 `nil`
`mymap := make(map[string]string) `需要初始化才能存放键值对。
`mymap["123"]="23"`
`b, c := a["123"]` 若不存在键值，c 为 false，b 为空。
使用 `range` 遍历的是键值：

```go
for country := range countryCapitalMap {
        fmt.Println(country, "首都是", countryCapitalMap [country])
    }
```

删除键值：`delete(countryCapitalMap, "France")`
map 作为函数参数传递：`fun test(m map[int]int)` 。本质是值传递

####  **defer** 

`defer print(xxx)` 在 main 函数结束前执行语句。多个 defer 下，根据先进后出执行，部分 defer 语句发生什么错误，其他 defer 语句仍可以执行。defer 的匿名函数会先传递参数。

#### 工作区

导入包 `import xx` 必须使用，否则编译不过。导入包起别名：`import name "fmt"`

1、创建 `greeting` 文件夹，`go mod init example.com/greetings ` 初始化模块，模块名称建议为 `公司名/包名`。文件夹中的所有源文件需写入 `package greeting`。 **如果使用别的包的函数，包名中的函数必须首字母大写。** 
2、创建 `main` 文件夹，`go mod init example.com/main` 初始化同上，创建 `main.go` 并编写。
3、`go mod edit -replace example.com/greetings=../greetings` 指定依赖包的路径。
4、`go mod tidy` 生成 `example/main` 的依赖信息。完成后 `main` 文件夹下的 `go.mod` 内容为：

```go
module example.com/main

go 1.17

replace example.com/greetings => ../greetings

require example.com/greetings v0.0.0-00010101000000-000000000000
```

导入包时，首先执行 `func init(){}` 函数。调用某个包的 `init` 函数，但不调用包下其他函数：`import _ "fmt"`
同包 `package` 下，使用其他文件函数，直接调用函数名。

### 函数

函数定义：

```go
func max(num1 int, num2 int) int {
    return result
}
```

支持返回多个值：`func swap(x, y string) (string, string){}`

以上为通过  **值传递** ，在函数中处理的为单数副本。使用 **引用传递** 可修改值。

```go
func swap(x *int, y *int) {
   var temp int
   temp = *x    /* 保持 x 地址上的值 */
   *x = *y      /* 将 y 值赋给 x */
   *y = temp    /* 将 temp 值赋给 y */
}
```

不定参数通过迭代或者索引使用，传递 `args` 类型为数组

```go
func myfun(args ...int) {  
	for i, val := range args {
		println(i, val)
	}
}
```

 `myfun2(args...)` 传递全部 args 参数，传递部分参数：`myfun2(args[:2]...)`

#### 闭包

```go
a := 1
f1 := func() {
    print(a)
}
```

闭包中的函数可以捕获到外部定义的变量。只要闭包函数还在使用这个变量，他就会一直存在。 如：

```go
func getSequence() func() int {
   i:=0
   return func() int {
      i+=1
     return i  
   }
}
```

#### 方法

为结构体定义方法：`func (c Cicle) getArea() float64{return c.r * c.r *3.14}` 
使用：`var c1 Circle` , `c1.getArea()`

#### 函数类型

定义函数类型 `type FuncType func(int,int) int`, `var test FuncType `

#### 回调

传入对应的函数对象，实现不同功能

```go
type FuncType func(int,int) int
func Calc(a, b int, myfun FuncType) int {
	return myfun(a, b)
}
```

### 面向对象编程

封装通过方法实现，继承通过匿名字段实现，多态通过接口实现

#### 匿名组合

 **结构体匿名字段** 类似于继承，实现了代码复用：

```go
type Person struct {
    name string
    age  int
}

type a struct {
    Person  
    num int
}
var c a = a{Person{"1", 1}, 1}
d := a{Person{"1", 1}, 1}
```

只有类型，没有名字。匿名字段继承了 Person 所有的成员。`fmt.Printf("%+v", d)` 使用格式化输出详细的结构体信息，结果：`{Person:{name:1 age:1} num:1}` 。`%T` 输出变量类别， `%p` 输出变量地址。
结构体中的结构体可以当做整体进行复制：`d.Person = Person{"1",1}`
对于同名字段，默认采用就近原则：优先操作本作用于字段。

对于指针类型的匿名字段：`type a struct{*Person}` 初始化使用地址：`d := a{&Person{"1", 1}, 1}` 

#### 方法

类似于面向对象的封装。`func (temp Long) fun_name(){}`  
为结构体添加方法：

```go
func (temp Person) PrintInfo(){print(temp.name)}

p:= Person("name")
p.PrintInfo()
```

如果要修改结构体内容，使用 `func (p * Person) Setinfo(){}` 。其中接受者类型本身不能为指针。如 `person = &Person` `func (p person)Setinfo(){}` 会出错。
方法不支持重载。

相同变量，通过指针或者普通变量定义的方法均可调用。

结构体匿名字段可以继承方法。

#### 接口

```go
type Phone interface {
    call()  //只放函数声明
}

type NokiaPhone struct {
}

func (nokiaPhone NokiaPhone) call() {
    fmt.Println("I am Nokia, I can call you!")
}
func main() {
    var phone Phone

    phone = new(NokiaPhone)
    phone.call()
}
```

多态可以用于函数中，将接口作为参数传入：

```go
func Makecall(p Phone){p.call()}
```

接口继承：

```go
type Smartphone interface {
	Phone
	sing()
}
```

超集可以转换为子集：

```go
func main() {
	var phone Phone       // 子集
	var sphone Smartphone // 超集
	phone = sphone
	phone = new(NokiaPhone)
	//phone.sing()
	phone.call()
}
```

空接口可以保存任意类型的值：`var i interface{} = 1`，定义三个空接口：`i := make([]interface{},3)`
判断其中值是否为某类型 `value, ok = data.(int)`，data 为 `int` 时为 `true`

### 异常处理

#### error

error 是一个接口类型

```go
packages errors

type error interface { 
Error() string
}
```

返回一个错误信息 `err1 = errors.New("math: square root of negative number")`

或者使用 `err1 := fmt.Errorf("error message)`

#### panic

比 error 错误更致命，会导致程序崩溃。
显示调用：`panic("message")`
空指针、数组越界会导致 panic

#### recover

设置 recover

```go
func test(x int) {
	defer func() {
		if err := recover(); err != nil {
			print("new error")
			fmt.Println(err)
		}
	}()
	var a [2]int
	a[x] = 0
}
```

### 字符串

`import strings`
`Contains(s string, substr string) bool` 
拼接`strings.Join(sl []string, s string) string`
`strings.Index(s string, substr string) int`
重复字符串 n 次：`strings.Repeat(s string, n int) string`
以 s2 拆分字符：`strings.Split(s string, s2 string)  []string`
去掉两头的字符：`strings.Trim(s, s2 string) string`
去掉前后空格，并按空格分隔字符串：`strings.Fields(s string) []string`

#### strconv

字符串转换使用 `strconv`
常用的有 
append：`strconv.AppendInt(slice, 1234, 10)`以 10 进制添加字符串。
Format：`strconv.FormatFloat(f 3.14, 'f', -1, 64)`  或整型转字符串 `strconv.Itoa(666)`
字符串转整型：`strconv.Atoi("567")`

#### 正则表达式

通过 regexp 包使用，实现了 RE2 语法。 大部分与 python 语法相似。效率没有 strings 包高，但是功能强大。

```go
a := "123 143 45 4"
pattern := `1\d3`  // 此处使用`` 非 ""
reg1 := regexp.MustCompile(pattern)
if reg1 == nil {
    fmt.Println("error message")
    return
}
result := reg1.FindAllStringSubmatch(a, -1)
fmt.Println(result)
//[[123] [143]]
```

#### json 

通过结构体生成 json：

```go
type IT struct {
	// 成员变量名必须大写
	Age  int `json:"age"` //二次编码，输出时属性名为小写。
	Num  int `json:"-"`   //不会输出
	Name string
	Isok bool `json:",string"`
}

func main() {
	i := IT{1, 1, "123",true}
	buf, err := json.Marshal(i)
    // buf 为切片类型
	if err != nil {
		fmt.Println("err = ", err)
	} else {
		fmt.Println(string(buf))
	}
}
```

格式化输出使用：`buf, err := json.MarshalIndent(i, "空符号", "缩进符号") `

通过 `map` 生成 json：

```go
a := make(map[string]interface{}, 4)
a["name"] = "123"
a["member"] = []string{"123", "apple"}
a["num"] = 34
a["isok"] = true
buf, err := json.Marshal(a)
```

解析 json 到结构体：

```go
var temp IT
jsonBuf := `{"isok":true,"age":123,"Name":"123","num":34}
`  
err := json.Unmarshal([]byte(jsonBuf), &temp)  // 需要结构体地址传递
if err != nil {
    fmt.Println("err = ", err)
} else {
    fmt.Println(temp)
}
```

jsonBuf 需要使用未解析字符串，json 文件需要与结构体中的 json 命名规则对应。

json 到 map：

```go
a := make(map[string]interface{}, 4)
err := json.Unmarshal([]byte(jsonBuf), &a)
```

取值需要使用 `c := a["Name"]` 直接提取，使用`va c string`, `c = a["Name"]` 存在错误：`cannot use a["Name"] (type interface {}) as type string in assignment:`

### 文件操作

标准输出设备（屏幕）为 `os.Stdout`，
标准输入：`fmt.Scan(&input)`
写入文件：Create 会抹掉文件原有内容：`f, err = os.Create(path string)`；使用后关闭文件 `f.Close()`；
写入：`n, err := f.writeString(s string)` ；`n == len(s)`
读取文件：`f, err = os.Open(path string)` ；
每次读取一行：

```GO
r := bufio.NewReader(f)
for {
    text, _, err := r.ReadLine()
    if err == io.EOF {
        break
    }
    fmt.Println(string(text))
}
```

`info, err = os.Stat(filename string)` 获取文件属性，包括大小，名字等。

### 并发

go 语言从语言层面你支持了并发

#### goroutine

开启一个新的运行期线程 goroutine 来执行函数：`go fun(...)`。
主线程退出后，子线程也会退出。

#### runtime

`import runtime`
`runtime.Gosched()` 先让出时间，等别的协程执行完毕后再接下去执行。
`runtime.Goexit()` 终止当前协程。
设置 CPU 核数：`runtime.GOMAXPROS(4) `

####  **channel** 
`ch := make(chan int)` 声明一个通道
`ch <- v` 把 v 发送到 ch 通道
`v := <- ch` 从 ch 接收数据，并进入阻塞直到数据接收完毕。
默认通道不带缓冲区，发送端发送数据，接收端必须接收数据。 **当通道中有未读取的数据并且达到容量上限时，发送端会进入阻塞。当通道中没有数据时，接收端会进入阻塞。** 
带缓冲区的通道：`ch := make(chan int, 100)` 。此处的容量： `cap(c) == 100`
关闭：`close(ch)`，关闭 channel 后，再发数据会导致 panic，但是可以继续从 channel 接收数据。
`num, ok := <-ch` 如果 `ok == true` 说明管道没有关闭
使用 range 遍历 channel，在结束时会自动跳出循环。`for num := range ch{}`

单向 channel：`var writeCh chan<- int = ch` 将双向管道改为单向。`writeCh <- 6`；只读：`var readCh <-chan int = ch`，`<-readCh`。单向无法转换为双向。
单向 channel 作为函数传参使用：

```go
func p(out chan<- int){
out <- 1
}
func c(in <-chan int){
    print(in)
}
ch :=make(chan int)
// 调用 p(ch) c(ch)
```

### API

#### Timer

`timer := time.NewTimer(time.Second)` 一秒后往 `time` 通道写入当前时间内容（仅写入一次），`t := time.C` 从 `time` 通道接收。

`timer := time.NewTricker(time.Second)` 以一秒中为周期，持续往 `time.C` 通道写入内容。

定时器停止：`timer.Stop()`
定时器重置时间：`timer.Reset(time.Second)`

#### Select

可以实现几秒后没有操作就退出程序：

```go
select {
    case x <- ch:
    print(x)
    case <-time.After(5*time.Second):
    	break
}
```

执行条件满足的语句，当多个语句满足条件时，随机顺序执行所有满足的语句。

### 网络编程

#### net

`import net`
`listener, err :=  net.Listen("tcp","127.0.0.1:8000")`

服务器初始化：`listener, err := net.Listen("tcp", "127.0.0.1:8000")`
服务器接收：`conn, err := listener.Accept()`，通过 byte 数组接收`buf := make([]byte, 1024)`，`n, err1 := conn.Read(buf)` 返回 `n` 为数据长度；
服务器发送：`conn.Write([]byte("123"))`

客户端初始化：`conn, err := net.Dial("tcp", "127.0.0.1:8000")`
客户端发送：`conn.Write([]byte("this is a meaage"))`；
客户端接收服务器信息：`n, err1 := conn.Read(buf)`

使用结束后关闭：`listener.Close()`，`cnn.Close()`

#### HTTP

`import net/http`

服务器网页返回信息：

```go
func myHandler(w http.ResponseWriter, r *http.Request) {
	 w.Write([]byte("hello"))
}
```

设置 `/go` 路径的返回函数：`http.HandleFunc("/go", myHandler)`
设置服务器地址：`http.ListenAndServe("127.0.0.1:8000", nil)`

客户端：
`resp, err := http.Get("http://www.baidu.com")`
获取请求信息：`resp.Status`, 
读取网页：

```go
buf := make([]byte, 4*1024) 
var tmp string
for {
    n,err := resp.Body.Rea(buf)
    if n == 0{
        fmt.Println("read end")
    	break
    }
    tmp += string(buf[:n])
}
resp.Body
```



