---
title: 客户端开发 | 微信小程序笔记
date: 2023-08-27
author: Kevin 吴嘉文
tag:
- webfront
category:
- 知识笔记
---

# 小程序开发指南

[官方文档](https://developers.weixin.qq.com/miniprogram/dev/devtools/devtools.html)，[官方小程序开发文档](https://developers.weixin.qq.com/miniprogram/dev/framework/)

小程序的主要开发语言是 JavaScript ，小程序的开发同普通的网页开发相比有很大的相似性。对于前端开发者而言，从网页开发迁移到小程序的开发成本并不高，但是二者还是有些许区别的。

+ 前段开发非常熟悉的一些库，在小程序开发中是用不了的，如 jQuery 等。

+ 开发 IOS 和 Android 小程序的运行环境有一点点不同 

[小程序 API](https://developers.weixin.qq.com/miniprogram/dev/api/) - 微信包装好的，如之父，跳转，数据缓存等 API。

[小程序组件](https://developers.weixin.qq.com/miniprogram/dev/component/) - 类似 vue 中的组件，可以直接调用

## 小程序开发入门

1. 下载开发者工具

[微信开发者工具下载](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)

2. 创建小程序
3. 选择对应的 APPID（注册好小程序之后会给）

![image-20230827140557736](/assets/img/xiaochengxu/image-20230827140557736.png)

点击确认后进入编辑页面，官方会为你创建好程序的模板：

![image-20230827164718808](/assets/img/xiaochengxu/image-20230827164718808.png)

3. 小程序界面介绍

## 开发者应用文件架构说明

####  **JSON 配置文件** 

配置文件储存在 `*.json` 当中

`app.json` 是当前小程序的全局配置（[文档 - 小程序的配置 app.json](https://developers.weixin.qq.com/miniprogram/dev/framework/config.html)），包括了小程序的所有页面路径、界面表现、网络超时时间、底部 tab 等。如

```json
{
  "pages":[
    "pages/index/index",  // 告诉小程序你的启动文件是哪个
    "pages/logs/logs"
  ],  
  "window":{ //定义小程序所有页面的顶部背景颜色，文字颜色定义等
    "backgroundTextStyle":"light",
    "navigationBarBackgroundColor": "#fff",
    "navigationBarTitleText": "Weixin",
    "navigationBarTextStyle":"black"
  }
}
```

`project.config.json` 保存开发者工具的配置。

`page.json` 可以配置每个小程序页面的属性，如顶部颜色、是否允许下拉刷新等等。[页面配置](https://developers.weixin.qq.com/miniprogram/dev/framework/config.html#%E9%A1%B5%E9%9D%A2%E9%85%8D%E7%BD%AE)

#### WXML 模板

是微信中的 HTML 语法，例如

```html
<view class="container">
  <view class="userinfo">
    <button wx:if="{{!hasUserInfo && canIUse}}"> 获取头像昵称 </button>
    <block wx:else>
      <image src="{{userInfo.avatarUrl}}" background-size="cover"></image>
      <text class="userinfo-nickname">{{userInfo.nickName}}</text>
    </block>
  </view>
  <view class="usermotto">
    <text class="user-motto">{{motto}}</text>
  </view>
</view>
```

WXML 知识 MVVM 开发模式，（类似 Vue）。WXML 中也包装好了其他的组件： [小程序的能力](https://developers.weixin.qq.com/miniprogram/dev/framework/quickstart/framework.html)。

#### WXSS

微信中的 CSS。更详细的文档可以参考 [WXSS](https://developers.weixin.qq.com/miniprogram/dev/framework/view/wxss.html) 

#### JS 逻辑交互

微信中的逻辑交互类似 vue，在 WXML 中定义 

```html
<view>{{ msg }}</view>
<button bindtap="clickMe">点击我</button>
```

而后在 page 对应的 `*.js` 文件中提供方法。

```js
Page({
   data: {
     // 这里配饰 data，类似 vue  
   },
    // 公开的函数方法直接配置，无需像 vue 那样写在 method 里面。
  clickMe: function() {
    this.setData({ msg: "Hello World" })
  }
})
```

::: details js 配置示例

```js
const app = getApp()

Page({
  data: {
    motto: 'Hello World',
    userInfo: {},
    hasUserInfo: false,
    canIUse: wx.canIUse('button.open-type.getUserInfo'),
    canIUseGetUserProfile: false,
    canIUseOpenData: false // 如需尝试获取用户信息可改为 false
  },
  // 事件处理函数
  bindViewTap() {
    wx.navigateTo({
      url: '../logs/logs'
    })
  },
  onLoad() {
    if (wx.getUserProfile) {
      this.setData({
        canIUseGetUserProfile: true
      })
    }
  },
  getUserProfile(e) {
    // 推荐使用 wx.getUserProfile 获取用户信息，开发者每次通过该接口获取用户个人信息均需用户确认，开发者妥善保管用户快速填写的头像昵称，避免重复弹窗
    wx.getUserProfile({
      desc: '展示用户信息', // 声明获取用户个人信息后的用途，后续会展示在弹窗中，请谨慎填写
      success: (res) => {
        console.log(res)
        this.setData({
          userInfo: res.userInfo,
          hasUserInfo: true
        })
      }
    })
  },
  getUserInfo(e) {
    // 不推荐使用 getUserInfo 获取用户信息，预计自 2021 年 4 月 13 日起，getUserInfo 将不再弹出弹窗，并直接返回匿名的用户个人信息
    console.log(e)
    this.setData({
      userInfo: e.detail.userInfo,
      hasUserInfo: true
    })
  }
})
```

:::

## 小程序 API

#### 组件

类似 vue 开发，小程序提供了各种组件。可以在 WXML 中直接调用他们。[小程序组件](https://developers.weixin.qq.com/miniprogram/dev/component/) - 类似 vue 中的组件，可以直接调用

#### API

微信小程序提供了很多现成的 js 方法让我们使用，包括获取用户信息、本地存储、微信支付等。

[小程序 API](https://developers.weixin.qq.com/miniprogram/dev/api/) - 微信包装好的，如之父，跳转，数据缓存等 API。

## 版本和发布

-  **开发版本** ：使用开发者工具，可将代码上传到开发版本中。 开发版本只保留每人最新的一份上传的代码。
  点击提交审核，可将代码提交审核。开发版本可删除，不影响线上版本和审核中版本的代码。

-  **体验版本** ：可以选择某个开发版本作为体验版，并且选取一份体验版。（点击开发者小程序中的上传，然后让管理员将该版本设置为体验版本。）

在 管理-版本管理 中，可以找到开发版本，并选择对应开发版本为体验版本。

-  **审核版本** ：只能有一份代码处于审核中。有审核结果后可以发布到线上，也可直接重新提交审核，覆盖原审核版本。

在 管理-版本管理 中，可以找到开发版本，点击提交审核。提交审核规则需要符合 [小程序平台运营规范](https://developers.weixin.qq.com/miniprogram/product/) 以及 

-  **线上版本** ：线上所有用户使用的代码版本，该版本代码在新版本代码发布后被覆盖更新。

## 运营数据

登录 [小程序管理后台](https://mp.weixin.qq.com/) - 数据分析；或使用 小程序数据助手。[官方运营数据介绍](https://developers.weixin.qq.com/miniprogram/analysis/#%E5%B8%B8%E8%A7%84%E5%88%86%E6%9E%90)

## 发送请求

微信发送请求有不少限制，[微信小程序网络使用说明](https://developers.weixin.qq.com/miniprogram/dev/framework/ability/network.html)。

+ 每个微信小程序需要事先设置通讯域名，小程序 **只可以跟指定的域名进行网络通信** 。包括普通 HTTPS 请求、上传文件、下载文件 和 WebSocket 通信。
+ 只能往 `https://域名:port` 发送请求，并且要求域名经过 ICP 备案，这意味着我们必须使用国内域名。
+ 

## 备注

+ 小程序主体迁移问题 

- [ ] 不确定个人主体是否可以迁移到企业。企业与个人税收及福利应该差很多。

+ AIGC 服务备案问题

- [ ] LLM 备案似乎是必须得了，那对于 AI 绘画需要备案嘛？在哪备案？
- [ ] 微信小程序发送 API request 只能发给域名，TCP API 那边的备案是肯定要的了。那小程序这边需要其他备案嘛？

+ 小程序规范

- [ ] 不要展示用户的任何信息，即便是设置了同意获取用户数据，尽可能只显示用户头像。

- [x] 了解企业域名备案，域名不提供网站，但是提供 tcp 服务时的备案是否与 web 网站备案流程一致？

+ 收费功能实现问题

- [x] 小程序设置中设置对公打款账号，可用于收费
- [x] 代码中如何设置？wx app 提供了 api，直接调用就行

+ 用户登录问题

- [ ] 用户 session 如何控制？确认 uniapp 中用户 session 实现方式，以及 wx 中 session 实现。

- 小程序也是用 npm build，然后放置在云服务器上的吗？

- [x] 似乎是的，通过 uni-app 的方案，似乎可以判断小程序采用了类似 web-UI 服务的搭载方式

