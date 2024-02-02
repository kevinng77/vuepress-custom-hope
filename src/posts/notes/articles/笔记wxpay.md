---
title: 微信小程序支付
date: 2023-12-08
author: Kevin 吴嘉文
category:
- 知识笔记
---

## 流程要点

::: info

以下提到的微信 API 请求的 base URL 为 `https://api.mch.weixin.qq.com`，如 `GET /v3/certificates` 表示 `GET https://api.mch.weixin.qq.com/v3/certificates `。

微信支付 API v3 要求商户对请求进行签名，微信支付会在收到请求后进行签名的验证。如果签名验证不通过，微信支付 API v3 将会拒绝处理请求，并返回 401 Unauthorized。制作签名参考[这里](https://pay.weixin.qq.com/docs/merchant/development/interface-rules/signature-generation.html)。

:::

用户发起支付时，大致的支付流程如下：

### 1. 用户提交订单

比如点击付款按钮；该操作相当于以下业务流程图中的【步骤 1】。

-  **【示例图步骤 2】** 前端发送 order id 等信息到后端。

### 2. 后台处理订单信息

该部分开发指南有：[官方指南](https://pay.weixin.qq.com/wiki/doc/apiv3/open/pay/chapter2_8_2.shtml)，或[开源 wechatpayv3](https://github.com/minibear2021/wechatpayv3?tab=readme-ov-file)，[微信支付接口文档](https://pay.weixin.qq.com/docs/merchant/apis/mini-program-payment/query-by-out-trade-no.html)（主要参考），[接口文档 2](https://pay.weixin.qq.com/wiki/doc/apiv3/apis/chapter3_5_1.shtml)（主要参考）

-  **【示例图步骤 3】** 后端接受到用户请求，提取用户 openid 等信息，获取商品信息，生成订单。
- 必要时，获取并更新平台证书（`GET /v3/certificates`），这一步可以参考[这里](https://pay.weixin.qq.com/docs/merchant/development/interface-rules/signature-generation.html)，当然如果使用开源方案，比如[wechatpayv3](https://github.com/minibear2021/wechatpayv3?tab=readme-ov-file)，我们是不需要手动去更新平台证书的。

### 3. 后台进行小程序下单

-  **【示例图步骤 4】** `【POST】/v3/pay/transactions/jsapi`。除了商户信息外，需要提交商户订单号（`out_trade_no`），用户信息，商品信息到微信 API。参考[这里](https://pay.weixin.qq.com/docs/merchant/apis/mini-program-payment/mini-prepay.html)。
-  **【示例图步骤 6】** 后台会接收到微信发过来的 `prepay_id`。

-  **【示例图步骤 7】** 根据 prepay id 生成下单参数和签名信息，官方参考代码在[这里](https://pay.weixin.qq.com/docs/merchant/apis/mini-program-payment/mini-transfer-payment.html)。wechatpay 参考实现：

  ```python
  prepay_id = result.get('prepay_id')
  timestamp = str(int(time.time()))
  noncestr = str(uuid.uuid4()).replace('-', '')
  package = 'prepay_id=' + prepay_id
  sign = wxpay.sign(data=[APPID, timestamp, noncestr, package]) # 使用 RSAwithSHA256 或 HMAC_256 算法计算签名值供调起支付时使用
  signtype = 'RSA'   
  ```

-  **【示例图步骤 8】** 将结果发送到前端。

### 4. 用户在小程序进行支付

-  **【示例图步骤 9】** 前端调用 `wx.requestPayment` ，可以配置回调函数，当用户取消支付或者支付成功后，调用相关函数，参考[这里](https://developers.weixin.qq.com/miniprogram/dev/api/payment/wx.requestPayment.html)。

-  **【示例图步骤 10】** 用户窗口弹出微信支付界面，显示订单信息，并询问是否确认支付。

  ![image-20240202220819325](/assets/img/wxpay/image-20240202220819325.png =x300)

-  **【示例图步骤 13】** 用户点击确认支付后，会弹出输入密码，或者指纹支付界面。

  ![image-20240202221101943](/assets/img/wxpay/image-20240202221101943.png =x300)

### 5. 后台订单处理

- 在【示例图步骤 4】中，我们填写了 `notify_url` 及异步接收微信支付结果通知的回调地址。
-  **【示例图步骤 16】** ，在用户输入密码，完成支付之后，后台定义好的 `/notify_url` 对应的 api 接收到微信发送过来的信息，显示用户支付的情况。关于如何写这个接受函数，可以参考[官方接口文档](https://pay.weixin.qq.com/docs/merchant/apis/mini-program-payment/payment-notice.html)和[开源实现](https://github.com/minibear2021/wechatpayv3/blob/1.2.43/examples/server/examples.py#L197)。（不过怎么测试都出不来用户取消支付的通知。）
-  **【示例图步骤 18】**  在 `notify_url` 中，我们需要返回接受结果给微信，具体回复格式查看[官方接口文档](https://pay.weixin.qq.com/docs/merchant/apis/mini-program-payment/payment-notice.html#%E9%80%9A%E7%9F%A5%E5%BA%94%E7%AD%94)。

### 6. 用户收到支付结果

-  **【示例图步骤 19】**  用户界面跳出如下结果提示，和弹窗通知：

![image-20240202222832792](/assets/img/wxpay/image-20240202222832792.png =x400)

### 7. 用户小程序状态变更

-  **【示例图步骤 20】**  前端发送请求到后端，查询支付结果。
-  **【示例图步骤 21】**  后端调用 API，`【GET】/v3/pay/transactions/out-trade-no/{out_trade_no}`，查询对应的订单状态，并进行处理。比如，如果订单状态为 `成功`，那么，处理结果可能是将用户加入到会员列表当中等等。
-  **【示例图步骤 23】**  后端将消息发送到前端进行展示，比如用户的界面上，显示了 `会员` 图标。 

#### 示例图

参考[微信小程序开发指引](https://pay.weixin.qq.com/wiki/doc/apiv3/open/pay/chapter2_8_2.shtml)实际前后端发生的流程图如下：

![业务流程图](https://pay.weixin.qq.com/wiki/doc/apiv3/assets/assets/img/pay/wechatpay/6_2.png)





## 操作对应

### 前端

以下以 黑马程序员前端项目 uniapp 小兔鲜儿微信小程序项目视频教程 中的示例代码，对应上面的业务逻辑图进行梳理。

```tsx
// 订单支付
const onOrderPay = async () => {
  if (import.meta.env.DEV) {
    // 开发环境模拟支付
    await getPayMockAPI({ orderId: query.id })
  } else {
    // #ifdef MP-WEIXIN
    // 1. 进入小程序下单； 8. 获取小程序支付参数
    const res = await getPayWxPayMiniPayAPI({ orderId: query.id })
    
    // 9. 调用 wx.requestPayment 发起微信支付
    await wx.requestPayment(res.result)
    // #endif
  }
  // 关闭当前页，再跳转支付结果页
  uni.redirectTo({ url: `/pagesOrder/payment/payment?id=${query.id}` })
}
```

`wx.requestPayment` 可以配置回调函数，当用户取消支付或者支付成功后，调用相关函数，参考[这里](https://developers.weixin.qq.com/miniprogram/dev/api/payment/wx.requestPayment.html)。

### 后端

参考 [微信支付 API v3 Python SDK](https://github.com/minibear2021/wechatpayv3?tab=readme-ov-file)，[微信支付接口文档](https://pay.weixin.qq.com/docs/merchant/apis/mini-program-payment/query-by-out-trade-no.html)，官方[微信支付说明文档](https://pay.weixin.qq.com/wiki/doc/apiv3/open/pay/chapter2_8_0.shtml) 等。

1. 请求下单支付：

往 `/v3/pay/transactions/jsapi` 发送请求，[接口文档](https://pay.weixin.qq.com/docs/merchant/apis/mini-program-payment/mini-prepay.html)。

简洁版的示例为：

```bash
curl -X POST \
  https://api.mch.weixin.qq.com/v3/pay/transactions/jsapi \
  -H "Authorization: WECHATPAY2-SHA256-RSA2048 mchid=\"1900000001\",..." \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "appid" : "wxd678efh567hg6787",
    "mchid" : "1230000109",
    "description" : "Image 形象店-深圳腾大-QQ 公仔",
    "out_trade_no" : "1217752501201407033233368018",
    "notify_url" : " https://www.weixin.qq.com/wxpay/pay.php",
    "amount" : {
      "total" : 100,
      "currency" : "CNY"
    },
    "payer" : {
      "openid" : "oUpF8uMuAJO_M2pxb1Q9zNjWeS6o\t"
    }
  }'

```

-  **appid**  公众号 ID
-  **mchid**  直连商户号
-  **description**   商品描
-  **out_trade_no**  商户系统内部订单号，可以用来查询订单。
-  **notify_url**   异步接收微信支付结果通知的回调地址
-  **amount**  金额，单位 分
-  **payer**  支付者，需要提供 openid

除了 body 外，微信的接口还需要每个商家进行签名验证，请参考签名[验证文档](https://pay.weixin.qq.com/docs/merchant/development/interface-rules/signature-generation.html)，生成签名信息。



## 申请证书

[小程序支付接入前准备](https://pay.weixin.qq.com/wiki/doc/apiv3/open/pay/chapter2_8_1.shtml)

[ **APIv3 证书与密钥使用说明** ](https://pay.weixin.qq.com/wiki/doc/apiv3/wechatpay/wechatpay3_0.shtml)



## 资源

https://github.com/minibear2021/wechatpayv3?tab=readme-ov-file

[微信支付官方文档 - 接入准备](https://pay.weixin.qq.com/wiki/doc/apiv3/open/pay/chapter2_8_1.shtml)