---
title: RASA 回忆录（二） - RASA 自定义通道/储存
date: 2022-09-25
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
---

RASA 感觉像是时代的眼泪了，自动 LLM 火了之后，类似 RASA 一类的传统 NLP 对话架构变得冷门，不少技术似乎也被摒弃。但如今不少厂家在实现对话机器人、智能服务等应用时，却也会用到这类型的框架。市面上提供类似 RASA 服务的 low code 平台也不少，如 Nuance, Omilia, Kore.ai 等。

该系列文章主要围绕 RASA 源码，整理落地部署 RASA 类似系统的一些笔记。

本文围绕 RASA 的源码架构，针对 RASA 自定义部署方式，RASA 代码拆解功能展开。

参考代码：

https://github.com/kevinng77/rasa_example/tree/master/examples/4_unpack_rasa

### 安装环境

RASA 采用 poetry 进行包管理，可以直接 `poetry install` 来安装对应依赖。如果没有 poetry 的话，就：

```bash
pip install -r requirements.txt
```

默认情况下官方推荐使用 `rasa run` 启动 nlu 以及 action 模块。

Entrypoint 对应的主函数在 `rasa/__main__.py` 中，因此我们以模块的方式运行它就行了。这也是 poetry 中触发 rasa 的方法。至于如何用 rasa 直接开启服务，可以查看 poetry python 包管理工具。

执行 `rasa train` 等价于 `python -m rasa train`；`rasa run actions` 相当于 `python -m rasa run actions`。

因此，如果我们只希望用到 rasa 当中的某个模块而非整个 rasa 模块，就可以避开其复杂的 sanic router，采用传统 python 的方式运行其中对应的函数了。

## 自定义 Channel

https://github.com/kevinng77/rasa_example/tree/master/examples/5_custom_channel

自定义 channel 可以用于修改和用户交互的方式，或进行一些输出预处理等。比如开源的 RASA 并不支持语音功能，那么我们可以在自定义 Channel 当中加入 ASR 和 TTS 模块实现语音文字转换。

首先需要自定义 Input channel 模块，重写 `blueprint` 等模块，代码请参考 

https://github.com/kevinng77/rasa_example/blob/master/examples/5_custom_channel/mychannel/myrest.py#L22

而后根据需求对 Output Channel 的 `sent_text_message`, `sent_image_url`也进行重写，[示例代码](https://github.com/kevinng77/rasa_example/blob/master/examples/5_custom_channel/mychannel/myrest.py#L196)。

https://github.com/kevinng77/rasa_example/blob/master/examples/5_custom_channel/mychannel/myrest.py#L196

最后在 `credentials.yml` 中配置使用我们定义的 Channel：

```yaml
mychannel.myrest.RestInput:
```

配置完成后启动整个 rasa 服务：

```python
rasa train
rasa run actions &
rasa run --enable-api
```

打开一个新的终端 B 输入：

```
curl -X POST http://localhost:5005/webhooks/rest/mywebhook \
-H "Content-Type: application/json" \
-d '{  "sender": "change_to_your_name",   "message": "Find me some good venues"}' 
```

你会在终端 A 中看到：

```
This is a custom webhook made by me.
>>> you could see the following information: sender_id change_to_your_name,
 text Find me some good venues
>>> the custom channel is used for custom front-end interaction, if you want to save
middleware information, please use custom trackerstore, see next example
```

我们主要将 channel 自定义在了 `mychannel` 中，并在 `RestInput.blueprint()` 中修改了 app 的 router。

### WebUI

对于使用 RASA 搭建 WebUI，可以考虑使用 widget + socket IO 的方式实现。

https://github.com/kevinng77/rasa_example/tree/master/examples/8_wedget

1. 在 credentials 中添加设置 socketio 通道:

```yaml
socketio:
  user_message_evt: user_uttered
  bot_message_evt: bot_uttered
  session_persistence: true
```

2. 启动 RASA 服务

```bash
rasa train
rasa run actions &
rasa run --enable-api --cors '*'
```

3. 在任意 HTML 文件中添加（记得修改你的 websocket 为 `http://<ip>:<port>/socket.io`）:

```html
<div id="rasa-chat-widget" data-websocket-url="http://40.76.242.139:5005/socket.io"></div>
<script src="https://unpkg.com/@rasahq/rasa-chat" type="application/javascript"></script>
```

而后打开该网页，就能看到 UI 右下角的对话框了。

## 配置 Redis 储存

使用 redis tracker store 记录 session 还有 tracker。

https://github.com/kevinng77/rasa_example/tree/master/examples/6_redis_store

配置 redis 需要同时开启 rasa 和 redis 服务，为了方便 debug 及查看 redis 中的数据，我们同时部署 redis 和 redisinsight，编写以下 docker compose 文件：

```yaml
version: "3.0"

services:
  redis:
    image: redis
    container_name: redis
    ports:
      - 127.0.0.1:6379:6379
    volumes:
      - $PWD/redis_data:/data
      - $PWD/myredis.conf:/etc/redis/redis.conf
    command:
      redis-server --save 60 1 --requirepass 777777

  redisinsight:  
    image: redislabs/redisinsight
    container_name: redisinsight
    network_mode: "host"
```

而后启动服务 `docker compose up -d`

我们在 `endpoints.yml` 中配置好应用 redis 的一些参数:

```yaml
rest:
```

而后启动服务：

```bash
rasa run actions &
rasa run --enable-api
```

打开新终端尝试发送一个 POST 到服务器：

```bash
curl -X POST http://localhost:5005/webhooks/rest/webhook \
-H "Content-Type: application/json" \
-d '{  "sender": "change_to",   "message": "Find me some good venues"}'
```

执行发送后，rasa 在处理上面这个请求时，会将 tracker 记录在 redis 中。我们打开 `your_ip:8001` 查看，如下填写，其中密码是 777777：

![img](https://pic2.zhimg.com/80/v2-072028dcd5bbc1a80429c67e9f71684d_1440w.webp)

在 browser 中我们可以查看到 redis 数据库中的记录：

![img](https://pic4.zhimg.com/80/v2-a1363bfce70bbbfd31cb7393b7a68fa3_1440w.webp)

完整的 session 数据：

```json
{
  "events": [
	// 此处省略一些 session 内容
    {
      "event": "user",
      "timestamp": 1679306228.4886646,
      "metadata": {
        "model_id": "e1105180adce4701ac369a4b0ef90c8b"
      },
      "text": "Find me some good venues",
      "parse_data": {
        "intent": {
          "name": "search_venues",
          "confidence": 0.9929251670837402
        },
        "entities": [],
        "text": "Find me some good venues",
        "message_id": "e75148a4a0ad4f8cb1168d716ea43479",
        "metadata": {},
        "text_tokens": [
          [ 0, 4 ],
          [ 5, 7 ],
          [ 8, 12]
        ],
        "intent_ranking": [
          {
            "name": "greet",
            "confidence": 0.00012646465620491654
          },
          {
            "name": "thankyou",
            "confidence": 1.6760051948949695e-05
          }
        ]
      },
      "input_channel": "rest",
      "message_id": "e75148a4a0ad4f8cb1168d716ea43479"
    },
    {
      "event": "user_featurization",
      "timestamp": 1679306228.5121474,
      "metadata": {
        "model_id": "e1105180adce4701ac369a4b0ef90c8b"
      },
      "use_text_for_featurization": false
    },
    {
      "event": "action",
      "timestamp": 1679306228.5121644,
      "metadata": {
        "model_id": "e1105180adce4701ac369a4b0ef90c8b"
      },
      "name": "action_search_venues",
      "policy": "MemoizationPolicy",
      "confidence": 1.0,
      "action_text": null,
      "hide_rule_turn": false
    },
    {
      "event": "bot",
      "timestamp": 1679306228.5122068,
      "metadata": {
        "model_id": "e1105180adce4701ac369a4b0ef90c8b"
      },
      "text": "here are some venues I found",
      "data": {
        "elements": null,
        "quick_replies": null,
        "buttons": null,
        "attachment": null,
        "image": null,
        "custom": null
      }
    },
    {
      "event": "slot",
      "timestamp": 1679306228.5122104,
      "metadata": {
        "model_id": "e1105180adce4701ac369a4b0ef90c8b"
      },
      "name": "venues",
      "value": [
        {
          "name": "Big Arena",
          "reviews": 4.5
        },
        {
          "name": "Rock Cellar",
          "reviews": 5.0
        }
      ]
    },
    {
      "event": "action",
      "timestamp": 1679306228.516813,
      "metadata": {
        "model_id": "e1105180adce4701ac369a4b0ef90c8b"
      },
      "name": "action_listen",
      "policy": "MemoizationPolicy",
      "confidence": 1.0,
      "action_text": null,
      "hide_rule_turn": false
    }
  ],
  "name": "change_to"
}
```



其中比较重要的包括：`intent_ranking` ，entity 等数据都有。

## 小结

1. RASA 采用 sanic 搭建 API 服务，NLU 和 Dialog 模块分别由两个 API 服务组成（Kore.ai, Nuance 等平台也是采用该方案）。

2. RASA 中对轮对话方案是通过将用户历史的 intent, message, slot 等储存在服务器中实现的。（用户只想后端发送 message 和 session_id）。可以参考目前大多数 GPT 客户端多轮对话的方案，将历史对话信息储存在用户客户端的 Session Storage 或者 Local Storage 中，来环节本地储存的压力。

3. RASA 免费版支持的 Channel 较少，我们可以自定义 Channel 实现语音对话等功能。对于 Web UI 搭建的话，RASA 提供的  Widget 还是挺不错的。