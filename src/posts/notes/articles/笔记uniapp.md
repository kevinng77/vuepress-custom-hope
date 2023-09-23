---
title: 客户端开发（一）| uni-app 项目环境开发
date: 2023-08-27
author: Kevin 吴嘉文
tag:
- webfront
category:
- 知识笔记
---

::: tip

感谢黑马程序员提供的完善资料。该文章基于黑马程序员教材做了少量改动。

:::

*uni*-*app* 是一个使用 Vue.js 开发所有前端应用的框架，开发者编写一套代码，可发布到 iOS、Android、Web（响应式）、以及各种小程序、快应用等多个平台。

> 本来以为，开发手机 app，或者微信小程序，还需要另外学技术。有了 uni-app 后，很开心，前几年学的 VUE，现在用得上了，这是在暗示着创业顺利嘛哈哈哈

## 创建 uni-app 项目方式

 **uni-app 支持两种方式创建项目：** 

1. 通过 HBuilderX 创建

2. 通过命令行创建（推荐）

## HBuilderX 创建 uni-app 项目

### 创建步骤

 **1.下载安装 HbuilderX 编辑器** 

在官网 https://www.dcloud.io/ 下载对应编辑器。

 **2.通过 HbuilderX 创建 uni-app vue3 项目** 

![通过 HbuilderX 创建 uni-app vue3 项目](/assets/img/uniapp/image-20230827194022024.png)

 **3.安装 uni-app vue3 编译器插件** 

![安装 uni-app vue3 编译器插件](/assets/img/uniapp/uniapp_picture_3.png)

 **4.编译成微信小程序端代码** 

![编译成微信小程序端代码](/assets/img/uniapp/uniapp_picture_4.png)

 **5.开启服务端口** 

如果遇到打开微信小程序开发工具，在设置，安全下，将服务端口打开。

![开启服务端口](/assets/img/uniapp/uniapp_picture_5.png)

开启编译成小程序代码，可以在微信开发者工具上看到转换之后的代码。在 Hbuildex 上进行修改，并保存后，可以随时同步在微信开发者工具上。 **小技巧分享：模拟器窗口分离和置顶** 

![模拟器窗口分离和置顶](/assets/img/uniapp/uniapp_picture_6.png)

 **Hbuildex 和 微信开发者工具 关系** 

![Hbuildex 和 微信开发者工具 关系](/assets/img/uniapp/uniapp_picture_7.png)

::: tip

 **Hbuildex**  和  **uni-app**  都属于 [DCloud](https://dcloud.io) 公司的产品。

:::

## pages.json 和 tabBar 案例

### 目录结构

我们先来认识 uni-app 项目的目录结构。

```shell
├─pages            业务页面文件存放的目录
│  └─index
│     └─index.vue  index 页面
├─static           存放应用引用的本地静态资源的目录(注意：静态资源只能存放于此)
├─unpackage        非工程代码，一般存放运行或发行的编译结果
├─index.html       H5 端页面
├─main.js          Vue 初始化入口文件
├─App.vue          配置 App 全局样式、监听应用生命周期
├─pages.json        **配置页面路由、导航栏、tabBar 等页面类信息** 
├─manifest.json     **配置 appid** 、应用名称、logo、版本等打包信息
└─uni.scss         uni-app 内置的常用样式变量
```

### 解读 pages.json

用于配置页面路由、导航栏、tabBar 等页面类信息

### 案例练习

 **效果预览** 
![案例练习](/assets/img/uniapp/uniapp_case_1.png)

 **参考代码** 

::: tip

在定义 page 的时候，最后避免使用 `index.vue` 作为页面文件名称。减少小程序编译时可能出现的问题。

:::

```json
{
  // 页面路由
  "pages": [
    {
      "path": "pages/index/index",
      // 页面样式配置
      "style": {
        "navigationBarTitleText": "首页"
      }
    },
    {
      "path": "pages/my/my",
      "style": {
        "navigationBarTitleText": "我的"
      }
    }
  ],
  // 全局样式配置
  "globalStyle": {
    "navigationBarTextStyle": "white",
    "navigationBarTitleText": "uni-app",
    "navigationBarBackgroundColor": "#27BA9B",
    "backgroundColor": "#F8F8F8"
  },
  // tabBar 配置
  "tabBar": {
    "selectedColor": "#27BA9B",
    "list": [
      {
        "pagePath": "pages/index/index",
        "text": "首页",
        "iconPath": "static/tabs/home_default.png",
        "selectedIconPath": "static/tabs/home_selected.png"
      },
      {
        "pagePath": "pages/my/my",
        "text": "我的",
        "iconPath": "static/tabs/user_default.png",
        "selectedIconPath": "static/tabs/user_selected.png"
      }
    ]
  }
}
```

## uni-app 和原生小程序开发区别

### 主要区别

uni-app 项目每个页面是一个 `.vue` 文件，数据绑定及事件处理同 `Vue.js` 规范：

1. 属性绑定 `src="{ { url }}"` 升级成 `:src="url"`

2. 事件绑定 `bindtap="eventName"` 升级成 `@tap="eventName"`， **支持（）传参** 

3. 支持 Vue 常用 **指令**  `v-for`、`v-if`、`v-show`、`v-model` 等

### 其他区别补充

1. 调用接口能力， **建议** 前缀 `wx` 替换为 `uni` ，养成好习惯，这样支持多端开发。[uni_app api](https://uniapp.dcloud.net.cn/api/) 文档。
2. `<style></style>` 样式不需要写 `scoped`
3. 生命周期分为三部分：应用生命周期(小程序)，页面生命周期(小程序)，组件生命周期(Vue)

### 案例练习

 **效果预览** 
![案例练习](/assets/img/uniapp/uniapp_case_2.png)

 **主要功能** 

1.  滑动轮播图
2.  点击大图预览

 **参考代码** 

```vue
<template>
  <swiper class="banner" indicator-dots circular :autoplay="false">
    <swiper-item v-for="item in pictures" :key="item.id">
      <image @tap="onPreviewImage(item.url)" :src="item.url"></image>
    </swiper-item>
  </swiper>
</template>

<script>
export default {
  data() {
    return {
      // 轮播图数据
      pictures: [
        {
          id: '1',
          url: 'https://pcapi-xiaotuxian-front-devtest.itheima.net/miniapp/uploads/goods_preview_1.jpg',
        },
        {
          id: '2',
          url: 'https://pcapi-xiaotuxian-front-devtest.itheima.net/miniapp/uploads/goods_preview_2.jpg',
        },
        {
          id: '3',
          url: 'https://pcapi-xiaotuxian-front-devtest.itheima.net/miniapp/uploads/goods_preview_3.jpg',
        },
        {
          id: '4',
          url: 'https://pcapi-xiaotuxian-front-devtest.itheima.net/miniapp/uploads/goods_preview_4.jpg',
        },
        {
          id: '5',
          url: 'https://pcapi-xiaotuxian-front-devtest.itheima.net/miniapp/uploads/goods_preview_5.jpg',
        },
      ],
    }
  },
  methods: {
    onPreviewImage(url) {
      // 大图预览
      uni.previewImage({
        urls: this.pictures.map((v) => v.url),
        current: url,
      })
    },
  },
}
</script>

<style>
.banner,
.banner image {
  width: 750rpx;
  height: 750rpx;
}
</style>
```

## 命令行创建 uni-app 项目

 **优势** 

通过命令行创建 uni-app 项目， **不必依赖 HBuilderX** ，TypeScript 类型支持友好。

 **命令行创建**   **uni-app**   **项目：** （需要先安装好 vue-cli，请参考 vue 相关教程）

vue3 + ts 版

```shell
# npx degit dcloudio/uni-preset-vue#vite-ts 项目名称
npx degit dcloudio/uni-preset-vue#vite-ts kevin_app
```

创建其他版本可查看：[uni-app 官网](https://uniapp.dcloud.net.cn/quickstart-cli.html)

::: tip

如果遇到网络问题，尝试

- 在设备或路由器的网络设置中增加 DNS（如：8.8.8.8）
- 在设备中增加固定的 hosts（如：140.82.113.4 github.com）

:::



### 编译和运行 uni-app 项目

发布运行 uniapp 使用

```bash
npm run dev:%PLATFORM%
npm run build:%PLATFORM%
```

`%PLATFORM%` 可替换为 `mp-weixin`, `h5`, `app-plus` 等应用。以微信小程序为例：

1. 安装依赖 `npm install`
2. 编译成微信小程序 `npm dev:mp-weixin`

3. 运行后，生成 `dist/dev/mp-weixin` 文件夹。打开微信小程序，导入微信开发者工具

::: tip

在 `manifest.json` 文件添加小程序 `appid` 方便真机预览。

如果使用 windows + WSL2 开发，可以将 mp-weixin 文件夹软连接到 windows 本地目录下，如：

```bash
 ln -s /mnt/d/小程序项目/mp-weixin ./dist/dev/mp-weixin
```

如果直接用微信开发者工具，导入 wsl 内的 dist 文件，会出现报错。

:::

## 为什么使用 VS Code？

## 用 VS Code 开发 uni-app 项目

### 为什么选择 VS Code？

- VS Code 对  **TS 类型支持友好** ，前端开发者 **熟悉的编辑器**  👍
- HbuilderX 对 TS 类型支持暂不完善，期待官方完善 👀

### 用 VS Code 开发配置

- 安装 uni-app 插件
  -  **uni-create-view**  ：快速创建 uni-app 页面：右键 page，点击 new uni-app page
  -  **uni-helper uni-app**  ：代码提示；自动补全等
  -  **uniapp 小程序扩展**  ：鼠标悬停查文档
- TS 类型校验
  - 安装类型声明文件 `pnpm i -D @types/wechat-miniprogram @uni-helper/uni-app-types`
  - 配置 `tsconfig.json`
- JSON 注释问题
  - 设置文件关联，把 `manifest.json` 和 `pages.json` 设置为 `jsonc`

```diff
// tsconfig.json
{
  "extends": "@vue/tsconfig/tsconfig.json",
  "compilerOptions": {
    "sourceMap": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    },
    "lib": ["esnext", "dom"],
    "types": [
      "@dcloudio/types",
+      "@types/wechat-miniprogram",
+      "@uni-helper/uni-app-types"
    ]
  },
  "include": ["src/**/*.ts", "src/**/*.d.ts", "src/**/*.tsx", "src/**/*.vue"]
}
```

注意：原配置 `experimentalRuntimeMode` 现无需添加。

## 开发工具回顾

选择自己习惯的编辑器开发 uni-app 项目即可。

 **VS Code 和 微信开发者工具 关系** 
![VS Code 和 微信开发者工具 关系](/assets/img/uniapp/uniapp_picture_8.png)

 **HbuilderX 和 微信开发者工具 关系** 
![HbuilderX 和 微信开发者工具 关系](/assets/img/uniapp/uniapp_picture_7.png)

## 用 VS Code 开发课后练习

使用 `VS Code` 编辑器写代码，实现 tabBar 案例 + 轮播图案例。

温馨提示：`VS Code` 可通过快捷键 `Ctrl + i` 唤起代码提示。