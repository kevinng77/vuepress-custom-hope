---
title: 客户端开发 | figma 基础使用
date: 2023-10-03
author: Kevin 吴嘉文
tag:
- webfront
category:
- 知识笔记
---



# 资源

- [UI 设计模板|figma 社区](https://www.figma.com/community)

+ [mobile UI 设计|figma 社区](https://www.figma.com/community/tag/mobile/free/popular)

可以考虑的工具或模板：

- [手机组件|Mobile UI kit](https://www.figma.com/community/file/836596421863073964)

- [mobile UI 设计|figma 社区](https://www.figma.com/community/tag/mobile/free/popular) 中各种 ios ui 组件
- [一个 flow 流程设计的案例](https://www.figma.com/community/file/848318135747364351/wireframes-for-mobile-ui-design)

插件：

+ [UXPilot](https://www.figma.com/community/plugin/1224384489638070064/uxpilot-ai-color-gradient-ai) - AI 配色助手

```python
As an expert in UI design, I would like you to develop a collection of Pastel Colors that adhere to the specified criteria and the details provided in the product description below.

Criteria:

1. Limit the design to a maximum of 4 colors.
2. Integrate gradient colors into the color palette.
3. Display the HTML code for each color.

Product Description:

1. The application under development serves as a functional tool exclusively for Personal Watercraft (PWC) users.
2. The PWC logo has been attached for reference purposes. Please consider it while creating the color scheme.
```

+ [Mini Magic Moodboard and Art Generator](https://www.figma.com/community/plugin/1231006635261421541/mini-magic-moodboard-and-art-generator) - 根据颜色生成背景涂鸦



# 技巧

auto layout（容器根据内部内容自动更换长宽，元素间距及排版）:

+ 文字与形状同时选择，而后点击 autolayout；
+ 当容器内有多个 content 时，可以设置 content 之间的内间距。

玻璃拟态（磨砂玻璃的效果）：

+ 添加矩形，设置 background blur + 投影 + stride 渐变 + 透明度 

快捷蒙版：

+ 同 PS 当中的快捷蒙版，sigma 中可以可以 Frame 来限定元素展现的范围。

色彩提取：

+ 快捷键 `i`

## Document Style

style 可以看成变量，大致使用逻辑为：

1. 在 figma 页面定义好你的 `主题色`，可以将`主题色`先设置为红色。
2. 而后让你的元素使用 `主题色`，这时候你的元素会显示红色。
3. 修改`主题色`为绿色后，所有使用`主题色` 的元素都会显示为绿色。

如果要配置 style 颜色。如下图，点击 Libraries 中的 添加符号 +。而后添加 style 颜色。

![主题色示例图](/assets/img/figma/image-20231002192857056.png =x400)

当然除了添加颜色为 style，也可以为 style 设置 effects，字体风格等。

## Components

创建组件后，组件可以通过复制粘贴实现复用。要批量编辑组件时，可以通过修改父组件，来对全部字组件进行样式修改。

1. 选择几个元素，点击花瓣按钮 create component。父元素显示为花瓣按钮，子元素显示为矩形。

2. 对于每一个子组件，可以在右侧点击 frame 里的 go to main componment 来跳转到父组件。

![image-20231002195208790](/assets/img/figma/image-20231002195208790.png =x100)

::: tip

通常我们会将所有的父组件单独放在一页中。我们可以在左侧 asset 当中查看并使用组件。

![Asset 截图](/assets/img/figma/image-20231002195535274.png =x300)

:::

## 交互和 prototype

### prototype 基础

该功能可以模拟实现手机的页面跳转效果。

1. 点击元素，然后点击右边栏的 prototype。元素会出现 + 号。

![image-20231002200014724](/assets/img/figma/image-20231002200014724.png =200x200)

2. 点击  + 号选择链接对象。之后点击连接线，可以设置触发条件（如 on tap 等）和动画特效（如 css 中的 transition）。

![flow 示例图](/assets/img/figma/image-20231002200342490.png =x400)

3. 完成后点 左上角的 flow 播放键，可以进入到模拟手机页面
4. 当然，interations 也可以选择 nav back。

::: tip

1. figma 一般用于制作简单的页面跳转，如果需要高级特效，可以考虑使用 protopie 等其他工具。
2.  **每一个页面都可以是一个 flow 的 start 流程，我们可以将不同的业务线设置成不同的 flow，方面我们 debug。**  点击页面，然后在右侧 Prototype 选择点击 Flow start point，就可以添加新的 flow 了。

:::

### 遮罩与提示框

1. 创建新页面 “遮罩”，设置 fill 为 none。
2. 创建 prototype 链接，选择样式为 open overlap
3. 在 “遮罩” 页面添加对话框，消息框等。可以通过设置透明度来实现。

### 页面滚动

前面介绍到，当一个 frame 中的 content 不在 frame 里时，超出 frame 范围的 content 会被隐藏掉。

+ 我们可以在 frame 的 prototype 当中选择 scroll 选项（设置水平滑动或者垂直滑动），来定义进行 flow 时候，页面是否可以进行滑动。
+ 可以在 prototype 中设置页内跳转。比如返回顶部特效。

## 网页设计

推荐尺寸：1280 * 720

### 栅格

栅格是一些辅助线，在设计网页时候起到参照作用。

Layout Grid 中可以选择配置栅格，比如选择 layout grid 类型为 columns，然后设置相关参数：

![栅格示例图](/assets/img/figma/image-20231002231140182.png =x400)

### 网页动态响应

Design 中可以设置 constraints，来对不同尺寸的页面进行大小配置。这样不同大小的设备，都可以有好的视觉效果。

![Constraints 示例图 x200](/assets/img/figma/image-20231002232932098.png)

以上设置，不论页面多大，该元素总是居中。Constraints 也可以设置 scale 等其他方案。

