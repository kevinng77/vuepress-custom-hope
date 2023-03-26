---
title: D3 学习笔记
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

D3 为一个 JavaScript 函数库，是数据可视化工具，他能够灵活、简单地通过 web 端展示数据分析的结果。

[d3 中文文档 新](https://github.com/xswei/d3js_doc/blob/master/API_Reference/API.md) ， [d3 中文 V3 之前](https://github.com/d3/d3/wiki/API--%E4%B8%AD%E6%96%87%E6%89%8B%E5%86%8C), [旧版本 d3 画 bar chart](https://vegibit.com/create-a-bar-chart-with-d3-javascript/)， [D3 绘制动态表图](https://observablehq.com/@d3/learn-d3-interaction?collection=@d3/learn-d3)

## 导入

直接在 html 中导入 d3 来源，如：

```html
<script src="https://d3js.org/d3.v7.min.js"></script>
```

当然，也可以导入其他来源的 d3 包。

## 数据输入

[官方指南](https://observablehq.com/@observablehq/introduction-to-data) ；以下以 CSV 为例，同理可以导入 json, text, tsv 等格式

#### DOM 导入

直接定义成变量，`primes = [2, 3, 5, 7, 11, 13]`；检索：`primes[0]` 

对于 CSV 格式的可以这样导入：`data = d3.csv.parse(d3.select("pre#data").text()); ` 通过 `data[i]` 访问第 `i` 行，每行元素以键值对形式储存。新版本的 d3 可以直接 `d3.csvParse`

```html
<pre id="data">
name,mpg,cyl,disp,hp,drat,wt,qsec,vs,am,gear,carb
Mazda RX4,21,6,160,110,3.9,2.62,16.46,0,1,4,4
Mazda RX4 Wag,21,6,160,110,3.9,2.875,17.02,0,1,4,4
Datsun 710,22.8,4,108,93,3.85,2.32,18.61,1,1,4,1
Hornet 4 Drive,21.4,6,258,110,3.08,3.215,19.44,1,0,3,1
Hornet Sportabout,18.7,8,360,175,3.15,3.44,17.02,0,0,3,2
Valiant,18.1,6,225,105,2.76,3.46,20.22,1,0,3,1
Duster 360,14.3,8,360,245,3.21,3.57,15.84,0,0,3,4
Merc 240D,24.4,4,146.7,62,3.69,3.19,20,1,0,4,2
</pre>
```

从文件读取：

```javascript
flare = FileAttachment("flare.csv").csv()
flareGist = d3.csv("https://gist.githubusercontent.com/mbostock/4063570/raw/11847750012dfe5351ee1eb290d2a254a67051d0/flare.csv")
```

从 API 导入：

```javascript
client = DatabaseClient("Baby Names") // sample client for demonstration purposes

names = client.query(`SELECT name, gender, year, SUM(number) AS number
FROM names
WHERE year > ?
GROUP BY name, gender, year`, [1920])
```

## 数据处理

[d3 数组操作](https://github.com/d3/d3/wiki/%E6%95%B0%E7%BB%84#d3_zip) 

导入文件时候可以为每行进行格式：[d3 时间处理](https://github.com/xswei/d3-time-format/blob/master/README.md#locale_parse)

```javascript
const parseDate = d3.utcParse("%Y-%m-%d");
var data = d3.csvParse(string, function(d) {
  return {
    year: parseDate(d.year), // lowercase and convert "Year" to Date
    make: d.Make, // lowercase
    model: d.Model, // lowercase
    length: +d.Length // lowercase and convert "Length" to number
  };
});
```

d3 解析 CSV 文件后，返回 `object` 对象的形式为： `[{key:value},{}]`，每行数据以逗号隔开。可以通过 `.map(func())` 对每行数据批量处理。

#### 时间处理

旧版本时间处理，可能需要导入对应的时间库：`<script src="https://d3js.org/d3-time-format.v3.min.js"></script>`

```js
var parseTime = d3.timeParse("%Y-%m");
year_and_month = parseTime(d.FinancialYear); //返回 date 对象
```

#### 数组统计

常用：min, max, sum, mean, variance, deviation

排序：`arr.ascending()` 默认使用数值，也可以自定义比较器：

```js
function ascending(a, b) {
  return a < b ? -1 : a > b ? 1 : a >= b ? 0 : NaN;  //
}
```

二分搜索：`d3.bisectLeft(arr, x)` 返回小于 x 的最大数的位置

#### 数据分组

[网友笔记](https://blog.csdn.net/gdp12315_gu/article/details/51721988) 

```js
console.log(data[0], typeof data);
var d_nestByYear = d3.nest()
                    .key(d => d.year)   // 根据 year 属性分组
					.key(d => d.month)  // 再根据 month 分组
                    .entries(data);  // 应用在 data 数据集上

```

将 `.entries` 替换为 `.map`，得到的数据形式不同，但结构大同小异

也可以对数组进行聚合操作，聚合后的数据通过 `d.values` 访问：

```js
var d_nestByYear = d3.nest()
    .key(d => d.year)
    .rollup(function(leaves){
        return {"length":leaves.length,"mean_score":d3.mean(leaves,d=>d.ConditionScore)}   // 聚合多个类别
    })
    .entries(data);
```

## 语法笔记

### HTML 元素选择与操作 

关于本部分的 [更多 API](https://github.com/d3/d3/wiki/%E9%80%89%E6%8B%A9%E5%99%A8)

选中展示位置：

+ `d3.select()` 返回第一个匹配的元素

+ `d3.selectAll()` 返回所有匹配的元素

对于选中的对象，我们可以用 `.style()`，`.text()` 等方式为其 html 添加内容，如：我们 html 中存在 circle 对象：

```html
<circle cx="40" cy="60" r="10"></circle>
```

d3 通过以下方式更改 `circle` 对应的 html 内容：

```javascript
var circles = d3.select('body').selectAll('circle');  //也可以不用 select body
circles.style("fill", "steelblue");
```

以上操作相当于：

```html
<circle cx="40" cy="60" r="10" style="fill: steelblue;"></circle>
```

除了 `style` 还支持：

+ `circles.data(data)`：将一个数组绑定到选择集上，并且数据各项值分别与选择集的各元素对应
+ `circles.attr("r", function(d) { return Math.sqrt(d); })`：修改元素已有属性，此处 `r` 的值将设置为 `sqrt(data)`，需要提前绑定数据；
+ `circles.enter().append('circle')`：在选择集后面添加元素

添加删除元素：

```javascript
var body = d3.select("body");
body.append("div");  //在一个元素内添加一个元素
body.remove();  // 删除元素
```

元素与数据配对：假设我们目前有 3 个 `circle`；但绑定的数据为 4 个，那么可以采用 `enter()` 来选中多出的元素。反之，使用 `exit()`。

```javascript
var data2 = [32, 57, 112, 293]
var circles = body.selectAll('circle').data(data2);  // 选择 body 元素下的所有 circle 元素
var circleEnter = circles.enter().append("circle");  // 在元素集中添加一个额外的 circle 需要使用 enter 占位；
circleEnter.attr("cy", 60);  // append 函数返回添加的元素对象，可以直接修改他的属性

var data3 = [32, 57]
var circles = svg.selectAll("circle").data(data3);
circles.exit().remove();  // 删除没有数据绑定的元素
```

### SVG 相关函数

SVG 可缩放 **矢量** 图形，用于回执可视化的图形；其使用 xml 来定义图形，大部分的浏览器都支持 SVG，可以将 SVG 文本直接嵌入到 HTML 中显示。SVG 对象中的元素改变时，其图形也会改变，因此可以进行动态的图形化展示。[svg 基础语法|菜鸟教程](https://www.runoob.com/svg/svg-tutorial.html)

通过 D3 对 SVG 内容进行操作，先获得 SVG 画布。

+ `d3.select("body").append("svg")`：添加新画布
+ `d3.select("svg")`：选择已有画布

#### 柱状图案例

以下以 d3 旧版本代码做演示，绘制柱状图：

```javascript
var margin = { top: 50, right: 10, bottom: 30, left: 50 }
var width = 500 - margin.left - margin.right,
    height = 250 - margin.top - margin.bottom,
    barWidth = 30,
    barOffset = 10;
```

### d3 比例尺

[d3 官方示例](https://observablehq.com/@d3/learn-d3-scales?collection=@d3/learn-d3) 

比例尺可以看做一个函数，他将变量映射到画布的位置上。我们需要传入变量的域 `domain` 和画布的范围 `range`。以下先定义比例尺，如何使用参考绘图与坐标轴部分。

```js
var sampleData = d_nestByYear.map(d => d.values.length);
// y 值对应 [45, 543, 123,..., 500]
var sampleYear = d_nestByYear.map(d => d.key);
// x 值对应年份 [2000, 2001, ...]

var yScale = d3.scale.linear()  // d3 线性比例尺
    .domain([0, d3.max(sampleData)])  // 输入 y 的域。即 [0, 500] 区间
.range([height, 0])  // 输入 y 在画布上的范围，即最顶端 0，到最底端 height.
var xScale = d3.scale.ordinal()  // 离散型
    .domain(sampleYear)  // 此处 x 的域就是 sampleYear array 本身。
    .rangeBands([0, width])  
```

### 绘图

在绘图时，应用比例尺来来决定柱状图的位置 `x`, `y`, `height`。比例尺自动根据画布大小调整表格大小和位置。

如 `.attr('height', d=>yScale(d.values.length))` ；若 `d.values.length` 为 `[10,20,30]` ，比例尺对应画布`range` 为 `[0,200]`。那么 `yScale(d.values.length) == [0,100,200]`

```js
var q1 = d3.select("#question1")
	.append("svg")  // svg 矢量图
    .attr('width', width + margin.left + margin.right)  // 用 margin 留出坐标轴的空间
    .attr('height', height +  margin.top + margin.bottom)
    .selectAll("rect").data(d_nestByYear)
    .enter().append('rect')
    .style({"fill":"steelblue"})
    .attr('width', barWidth)
    .attr('height', d => height - yScale(d.values.length))
    .attr('x', (d, i) => 2 + margin.left + xScale(d.key))
	.attr('y', d => margin.top + yScale(d.values.length))
	// y=0 时为顶部，因此做个变换，然表格翻转
```

添加坐标轴

```javascript
var xAxis = d3.svg.axis()
    .scale(xScale)
    .orient('bottom')  // 位于底部
    .tickValues(sampleYear)  // 坐标轴上显示的标签

var xTicks = d3.select("#question1 svg").append("g").call(xAxis)
xTicks.attr('transform', "translate(" + margin.left + "," + (height + margin.top) + ")")
    .selectAll('path').style({ fill: 'none', stroke: "#000" })

var yAxis = d3.svg.axis()
    .scale(yScale)
    .orient('left')
    .ticks(10)

var yTicks = d3.select("#question1 svg").append("g").call(yAxis)
yTicks.attr('transform', "translate(" + margin.left + "," + margin.top + ")")
    .selectAll('path').style
```

添加其他标签，如标题、节点信息等，可以通过 `<text>` 节点来添加，并设计对应的 `x,y` 属性来控制位置，如添加节点信息：

```js
var numbers = q1.append("g")
    .style({'fill':'white','font-size':font_size})
    .selectAll("text").data(d_nestByYear)
    .enter().append('text').text(d=>d.values.length)
    .attr('x', (d, i) => 4 + margin.left + xScale(d.key))
    .attr('y',  d => margin.top + yScale(d.values.length)+font_size)
```

### 折线图案例

[D3 V3](https://bl.ocks.org/mbostock/3310323) [D3 V4+](https://observablehq.com/@d3/learn-d3-shapes?collection=@d3/learn-d3)

D3 中提供 shape 函数，包括 `line`, `arc` 等等。利用他们能够轻松地绘制折线图、饼图等形状。以下以折线图为案例，坐标轴添加方式同上。

在选中`svg` 画布后，绑定数据到我们的画布上，这边使用 `.datum()`。

```js
var q2 = d3.select("#question2").append("svg").datum(data)
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom);
```

构造 `line` 类别，他是一个折线图数据生成器。 line 中保存画布上的坐标数据 `(x,y)` 。因此需要用比例尺进行转换。

```js
var line = d3.svg.line()
    .x(d=> xScale(d.key))  // 映射原数据到画布上的坐标
    .y(d => yScale(d.values.mean_score))
```

生成折线图时，为了更好地调整 `margin` 位置，可以考虑将所有线和点信息放在一个 `g` 元素下。

```js
var q2_line = q2.append("g")               
                .attr('transform', `translate(${margin.left},${margin.top})`) 

q2_line.append("path")  // 生成折线图
    .attr("class", "line")
    .attr("d", line);

q2_line.selectAll(".dot")  // 绘制图上对应的点
    .data(d_nestByYear)
    .enter().append("circle")
    .style({'fill': 'white', stroke: 'steelblue'})
    .attr("class", "dot")
    .attr("cx", line.x())
    .attr("cy", line.y())
    .attr("r", 3.5);
```

### 多色散点图案例

[散点图案例](https://bl.ocks.org/mbostock/3887118)

散点图的添加方式与上一节这项图中的散点添加方式一样。使用 `svg.dot.circle` 进行添加。以下案例为散点添加了不同色彩。

```js
var color = d3.scale.category10();

q3.selectAll(".dot")
    .data(flat_data3)
    .enter().append("circle")
    .attr("class", "dot")
    .attr("r", 5)
    .attr("cx", d=>xScale(d.Manufacturer))
    .attr("cy", d=>yScale(d.mean_score))
    .style("fill", function(d) { return color(d.VehicleType); });
```

以上方式可以对三维数组进行可视化，此处的 `flat_data3` 为 `[{Manufacturer:xx,mean_score:xx,VehicleType:xx}]`。除了坐标轴外，需要使用 legend 对色彩提供信息：

```js
var legend = q3.append("g")
    .selectAll(".legend")
    .data(color.domain())
    .enter().append("g")
    .attr("class", "legend")
    .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

legend.append("rect")
    .attr("x", width - font_size)
    .attr("width", font_size)
    .attr("height",font_size)
    .style("fill", color);

legend.append("text")
    .attr("x", width - 20)
    .attr("y", 9)
    .style("text-anchor", "end")
    .text(function(d) { return d; });
```

