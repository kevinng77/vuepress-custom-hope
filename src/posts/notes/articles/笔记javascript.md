---
title: JavaScript 学习笔记
date: 2022-05-11
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- 计算机语言
mathjax: true
toc: true
comments: 笔记
---

## 概述

支持对 CSS/HTML 的静态、动态操作，可以对浏览器时间做出响应，实现网页与用户的互动；前段与后端的交互等。

## 基础语法

引入 js 的位置没有限制，但推荐统一放`<body>` 或 `<head> `部分中，或者同时存在于两个部分中。保持页面规范。

```html
<script>
    window.alert("hello world");
</script>

<script src="js/fistjs.js"></script>
```

#### 调试

google chrome 中可以直接对 js 进行调试。`ctrl + c` 打开开发者页面，在 `console` 中直接运行 js 代码。

#### 数据类型

js 中一般固定值称为字面量，`var 变量 = 字面量`。[类型转换](https://www.runoob.com/js/js-type-conversion.html)

 **值类型：** 

+  **数字：** `123`, `3.14`, `NaN`, `Infinity`, `1e-5`；js 只有一种数字类型。

将字符串转换为数字：`parseInt()`，`parseFloat()` 或者直接使用一元运算符 `+`， `var y = "5"; var x = + y;`

+  **字符串：** `var s = 'text' ` 或 `"text"` 

常用方法：`s[0]`；`s.length`；`toLowerCase()`；`indexOf()`；`fromCharCode()` 将 Unicode 转换为字符串；`search()` 正则匹配。见下面正则。`substring(beg,end)` 截取子字符串。
不推荐使用 String 创建对象：`var s = new String("123")` 
字符串模板：`price:${s}` 此处使用 \` 号包含，而非引号！！ 

+  **布尔值：** `true`、`false`

+  **空：** `Null` 为 object 对象，当使用完一个比较大的对象需要释放内存时，把他设置成 `null`

+  **未定义：** `Undefined`

 **引用数据类型（对象类型）：** 

+  **数组（Array）：** `[1,2,3,4]`

`slice(beg, end)` 截取；`.push()`,`.pop()`, `.sort()`, `.reverse()`, `arr.join("-")` 链接的方式和 python 相反；`.shift()` 删除头部，`.unshift()` 在头部插入

定义数组时，最后一个值后面不要添加逗号

```js
var colors = [5, 6, 7,]; //这样数组的长度可能为 3 也可能为 4。
```

+  **对象（Object）：** `var per = {key:"value",key3:123}`

定义对象，最后不能添加逗号。
对象内容能够直接复制：`per.name="123"`

访问对象`per.key` 或 `per['key']` ，对象中也可以储存函数方法 `func: function(){}`，并且调用 `per.func()`；对象中的函数能够使用 `this.var` 来访问到对象内的其他变量。

在浏览器中，`this` 默认为 window 对象

+  **map：**  

```js
var map = new Map([['tom',100],['jack',90],['tim',80]]);
var name = map.get('tom');//通过 key 获取 value
map.delete('tom');//删除元素
map.set('kate',70);
```

+  **set：** 

```js
var set = new Set([3,2,1]);
set.add(5); set.delete(3);  
set.has(1); 
set.size;
```

+  **函数：** `function func(a){return a;}` 定义一个函数

ES6 新增了箭头函数：`const x = (x, y) => x * y;` 即 `(参数 1, 参数 2, …, 参数 N) => { 函数声明 }`

+  **正则：** `var patt = /runoob/i `其中 `i` 为修饰符，这里表示搜索不区分大小写；`g` 执行全局匹配，匹配所有内容，而非匹配第一个后停止；`m` 执行多行匹配

字符串搜索：`s.search(patt)`；字符替换`s.replace("")`
常见的匹配符号：`[a-A0-9]`，`\d` 数字，`\b` 单词边界

+  **日期：** `data = new Date();`

获取时间：`.getDate()`, `getDay()`, `getFullYear()`，同理还有 `Hours`, `Minutes`, `Month`, `Seconds`。
将日期转换为 `time = data.getTime()` 后进行日期大小比较。
日期转字符串：`data.toLocaleString()`

+  **变量：**  `var a = 1`；赋值使用 `=`；等价与 `var a; a=1;` 对于 `var x,y,z=1;` 其中 `x,y` 不会被赋值。

注意：局部变量可以使用 `let`  来声明，防止更改到外部变量的值。如果在循环体内使用 `for(var i=0;i<4;i++){}` 那么循环结束后，`i` 的值将为 4

声明新的变量时候也可以用 `new` 关键词声明类型，如`var x = new Number;`
全局变量在页面关闭后销毁；局部变量在函数执行完毕后销毁；
js 中拥有动态类型，相同变量可用作不同类型。
 **声明提升：** JavaScript 中，函数及变量的声明都将被提升到函数的最顶部。因此代码中可以先使用变量，而后在其后面声明。
 **严格模式：**  在文件顶部加入`use strict`后，js 的语法要求会更严格。包括上述声明提升将不生效。不允许删除变量、函数；不允许使用转移字符等

逻辑运算符：`&&`, `||`, `!` 非操作
比较运算符：`==` 判断值相等；`===` 判断类型相等，值也一样；`!=`, `>`, `<`

#### 事件

HTML 页面事件包括按钮点击 `onclick`、页面完成加载 `onload`、HTML 元素改变 `onchange` ，鼠标移动到元素上 `onmouseover`，鼠标从一个 HTML 元素上移开 `onmouseout` 等。更多事件参考：[JavaScript 参考手册 - HTML DOM 事件](https://www.runoob.com/jsref/dom-obj-event.html)
html 中可以添加事件属性，使用 js 来对事件做出反应，js 函数用单/双引号添加，格式为：

```html
<some-HTML-element some-event='JavaScript 代码'>
```

如

```html
<button onclick="getElementById('demo').innerHTML=Date()">现在的时间是?</button>
```

事件中的 `this` 指向接收事件的 HTML 元素

### 运算符

常用的运算：`++`, `--`, `%` 取模, `/`, `*` ；支持自运算符如 `/=`, `%=`。数字与字符串相加的话，数字会先被转化为字符串。

### 条件语句

`voteable=(age<18)?"年龄太小":"年龄已达到";`

```js
if (time<10)
{}
else if (time>=10 && time<20)
{}
else
{}
```

### 循环

```js
for (let i=0;i<cars.length;i++){ }
for (x in person){}
while (i<5){    i++;}  // while 和 for(;i<5;){} 很像
do
{i++;}
while (i<5);
```

循环中可以 `break` 或者 `continue`

### 输出

`window.alert()` 弹窗
`console.log()` 输出到浏览器控制台
`debugger` 在开发者模式下打断点调试。

`document.write()` 在当前位置写入 HTML 内容。

`document.getElementByID("demo").innerHTML=` 在选中的 HTML 元素中写入或修改内容。

#### 错误语句

```js
try {
    if(x == "")  throw "值为空";    //异常的抛出
} catch(err) {
   message.innerHTML = "错误: " + err;    //异常的捕获与处理
} finally {
    ...    //结束处理
}
```

### 面向对象

class 关键字在 ES6 之后加入了。

```js
class Site {
  constructor(name) {
    this.sitename = name;
  }
  present() {
    return '我喜欢' + this.sitename;
  }
}
 
class Runoob extends Site {  //继承
  constructor(name, age) {
    super(name);  // 使用 super 调用父类构造方式
    this.age = age;
  }
  show() {
    return this.present() + ', 它创建了 ' + this.age + ' 年。';
  }
  get s_name() {  // getter 和 setter 用于操作类中属性
    return this.sitename;
  }
  set s_name(x) {
    this.sitename = x;
  }
}
 
let noob = new Runoob("菜鸟教程", 5);
document.getElementById("demo").innerHTML = noob.show();
```

### promise 对象

使用 `fetch` 等异步函数 （`async function(){}`） 后，会返回一个 promise 对象。 promise 对象可以使用 `.then()` 或者 `.catch()` 等，来对对象本身进行进一步操作。

```js
fetch(url).then(
resp=>resp.json()).catch(...)
```

值得注意的是，promise 对象使用 `.then` 或者 `.catch` 之后，返回的还是一个 `promise` 对象。但我们可以在 `.then`， `.catch` 中对 promise 中储存的 PromiseResult 进行处理。

## 操作 BOM

+  **window** 

浏览器对象，如获取浏览器大小：

```js
var w=window.innerWidth
|| document.documentElement.clientWidth
|| document.body.clientWidth;
 
var h=window.innerHeight
|| document.documentElement.clientHeight
|| document.body.clientHeight;
```

调整窗口大小 `window.resizeTo()` 等

+  **Screen**  ：`screen.availWidth`, `screen.availHeight `

+  **location：** 当前页面的 URL 信息

```js
host: "www.baidu.com"
href: "https://www.baidu.com/"
protocol: "https"
reloadLf reload() //刷新网页
// 设置新的地址
location.assign('新的网站')  // 可以用来设置新的网站跳转，获得类似流氓插件或网站的效果。
```

+  **document：** 

获取 DOM 节点：[css 选择器](https://www.runoob.com/cssref/css-selectors.html) ；推荐使用 jQuery ，而非原生 js 的 document 选择器。

```js
//对应 css 选择器
var h1 = doucment.getElementByTagName('h1');
var p1 = doucment.getElementById('p1');
var p2 = doucment.getElementByClassName('p2');
var father = doucment.getElementById('father');
//获取父节点下所有的子节点
var childrens = father.children;
```

对节点可以进行插入、删除、更新等操作；[DOM 常用命令](https://www.runoob.com/js/js-htmldom.html)

表单

```js
var x = document.forms["myForm"]["fname"].value;
```

其中 HTML 代码为：

```html
<form name="myForm" action="demo_form.php" onsubmit="return validateForm()" method="post">
名字: <input type="text" name="fname">
<input type="submit" value="提交">
</form>
```

在 js 代码块中可添加算法，对表单内容进行验证。

+ DOM 分配事件：

```html
<script>
document.getElementById("myBtn").onclick=function(){displayDate()};
</script>
```

+ EventListener：像元素添加事件反应

```js
element.addEventListener("mouseover", myFunction);
element.addEventListener("click", mySecondFunction);
element.addEventListener("mouseout", myThirdFunction);
```

+ 计时器

让一个函数一直运行，间隔 `m` 毫秒 `var myVar = window.setInterval("func",m);`
停止一个 Interval 的运行`window.clearInterval(myVar)`

## jQuery

[下载 jquery](https://www.bootcdn.cn/jquery/)

```html
<script src="https://cdn.bootcdn.net/ajax/libs/jquery/3.6.0/jquery.js"></script>
```

jQuery 基础语法：`$(selector).action()` ，`$` 代表使用 jQuery，`(selector)` 选择元素

如 `$(document).ready(function(){执行代码})`：等待 DOM 结构加载完毕后，执行函数

选择元素时采用了 Xpath 和 CSS 选择器语法的组合：

```js
$('p').click(); // document.getElementsByTagName();
$('#id').click(); // document.getElementById();
$('.class1').click(); // document.getElementsByClassName();
```

### 事件

```js
$("p").click(function(){
  $(this).hide(); // 其中 this 指向被选中的 p 元素
});
```

`hover()`

```js
$("#p1").hover(
    function(){
        alert("你进入了 p1!");
    },
    function(){
        alert("拜拜! 现在你离开了 p1!");
    }
);
```

### 操作

修改文本

```js
$('#test-ul li[name=python]').text();    //获得值
$('#test-ul li[name=python]').text('设置值');//设置值
$('#test-ul').html();    //获得值
$('#test-ul').html('<strong>123</strong>');  //设置值
```

修改元素属性

```js
$("#runoob").attr("href","http://www.runoob.com/jquery");  // 添加任意属性

$("h1,h2,p").addClass("blue");  // 添加 class 属性
$("h1,h2,p").removeClass("blue");
$("h1,h2,p").toggleClass("blue");  // 对类属性切换，有变无，无变有

$("p").css("background-color","yellow"); // 修改元素 CSS
```

添加元素，如在选中的 body 节点中添加元素：

```js
function appendText(){
    var txt1="<p>文本-1。</p>";              // 使用 HTML 标签创建文本
    var txt2=$("<p></p>").text("文本-2。");  // 使用 jQuery 创建文本
    var txt3=document.createElement("p");
    txt3.innerHTML="文本-3。";               // 使用 DOM 创建文本 text with DOM
    $("body").append(txt1,txt2,txt3);        // 尾部添加；头部添加使用 prepend()
}
```

后面添加元素：`$("img").after("在后面添加文本");` ；`.before()`前面添加

发送请求：

```js
$("button").click(function(){
  $.get("demo_test.php",function(data,status){
    alert("数据: " + data + "\n 状态: " + status);
  });
});

$("button").click(function(){
    $.post("/try/ajax/demo_test_post.php",
    {
        name:"菜鸟教程",
        url:"http://www.runoob.com"
    },
    function(data,status){
        alert("数据: \n" + data + "\n 状态: " + status);
    });
});
```

## 其他

### javascript 库

jQuery：用的很多的 js 库，简化了 DOM 操作。

React：体术虚拟 DOM，提升渲染效率；使用复杂，需要额外学习 jsx 语言

Vue：模块化开发+虚拟 DOM

Axios：前段通信框架

### 前段框架

Ant Design

ElementUI

Bootstrap：成熟的框架

AmazeUI：HTML5 跨屏前段框架

后端技术：NODEJS