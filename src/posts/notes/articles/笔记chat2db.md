---
title: Chat2db 体验心得
date: 2023-06-20
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
- AIGC
---

## 前言

最近在 github trending 看到了 [Chat2DB](https://link.zhihu.com/?target=https%3A//github.com/alibaba/Chat2DB/blob/main/README_CN.md)，碰巧个人正在做 Text2SQL, SQL2Text 等相关工作，因此便尝试了一下。Chat2DB 包含 AI 智能助手、团队协作、支持链接多种数据库等特性，本文仅覆盖  **AI 智能助手**  相关的分享。

总体感觉，Chat2DB 在数据库操作平台上嵌入了 AI 交互功能，但用户与 AI 的交互方式还有待改善。目前支持的功能并不够惊艳，以使用 MySQL 为例，如果用户手动在 ChatGPT 上提问 SQL 相关问题，然后将生成的代码内容粘贴到 MySQL 上执行，那么使用体验基本与 Chat2DB 差不多。


## Chat2db 示例

以下通过官方提供的 [docker](https://github.com/alibaba/Chat2DB/blob/main/README_CN.md) 进行安装，而后将一个 sqlite3 文件 `test.db` 共享到容器中进行测试。

```bash
docker run -ti -v ./test.db:/app/test.db --name=chat2db -p 10824:10824 chat2db/chat2db:latest
```

启动容器后，打开 `localhost:10824` 就能看到 Chat2DB 的 web UI。在 UI 界面选择添加以下 SQLite 数据源。

Chat2DB 默认使用 OPENAI 的 API，基于测试目的，笔者使用 [WizardLM](https://github.com/nlpxucan/WizardLM) 及 [FastChat](https://github.com/lm-sys/FastChat) 在本地的 5001 端口部署了 [OpenAI-compatible RESTful API](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/openai_api_server.py) 。下文中的 AI 回复内容均来自于  [WizardLM-13B](https://github.com/nlpxucan/WizardLM) 。

### AI 配置

在搭建好 [API 服务](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/openai_api_server.py) 后，在 Chat2DB 自定义 AI 界面进行 AI 来源配置。其中 Api Host 输入 `本地服务 ip:端口` 即可：

![image-20230620213720429](https://pic1.zhimg.com/80/v2-8e74baa2832990864f808fbdcbb938e8_1440w.webp)

 [OpenAI-compatible RESTful API](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/openai_api_server.py) 提供了 Text-to-embeddings，Text-completions，Chat-Completions 等多种 API。通过 FastChat 的 API 日志发现， Chat2DB 连接 OpenAI API 时，几乎所有的 AI 功能均使用了 Chat Completions 服务，即系统会向 `http://api_host/v1/chat/completions`发送请求，不同 AI 功能发送的请求内容不同。

![Chat2DB 的五种 AI 功能](https://pic4.zhimg.com/80/v2-a6b194ad7e2f278b148920674f0dda6f_1440w.webp)

以下为 Chat2DB 的五种 AI 功能： chatbot，自然语言转 SQL，SQL 解释，SQL 优化，SQL 转换 的示例以及请求内容。

#### chatbot

1. 示例：该功能类似 chatgpt 的闲聊，同一窗口下 **支持多轮对话** 。如下图 `--- BEGIN ---` 下面的语句为用户输入，`--- Chat 机器人 ---` 之后内容为 AI 回复。

![示例提问 记住：一斤苹果的价格是 5 块钱; 一斤苹果多少钱？](https://pic1.zhimg.com/80/v2-700fa2c394d53b8b4dbe887b2ea8e11c_1440w.webp)

2. 以下为 Chat2DB 向 OpenAI API 发送的请求内容。对于多轮对话功能，通过在发送的 messages 参数中进行拼接历史对话实现：

```json
{"model": "gpt-3.5-turbo", 
 "messages": [{"role": "user", "content": "记住：一斤苹果的价格是 5 块钱。"}, 
              {"role": "user", "content": "一斤苹果多少钱？"}], 

 "temperature": 0.2, 
 "top_p": 1.0,
 "n": 1, 
 "max_tokens": 2048, 
 "stop": None, 
 "stream": True, 
"presence_penalty": 0.0, 
 "frequency_penalty": 0.0, 
 "user": None}
```

#### 自然语言转 SQL

1. 示例提问 `这个表有多少行？`， `2012 年的记录有多少条？`：

除了 chatbot 功能之外，目前自然语言转 SQL，SQL 解释等其他功能都不支持多轮对话。

![提问： 这个表有多少行？ 2012 年的记录有多少条？](https://pic3.zhimg.com/80/v2-91aad580b6e5bd0b487c29d5ed077ac6_1440w.webp)

2. 该功能会使用 prompt template 处理用户的提问，如问 `2012 年的记录有多少条？` 时，表的结构信息会被添加到 prompt 中，而后发送给 LLM，以下是完整的请求内容：

```json
{'model': 'gpt-3.5-turbo', 
 'messages': [{'role': 'user', 'content': ' 请根据以下 table properties 和 SQL input 将自然语言转换成 SQL 查询. \n\n SQLITE SQL tables, with their properties:\n\n my_table(index, date, type, locale, locale_name, description, transferred)\n\n\n SQL input: 2012 年的记录有多少条？'}], 
 'temperature': 0.2, 'top_p': 1.0, 'n': 1, 'max_tokens': 2048, 'stop': None, 'stream': True, 'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'user': None}

```

#### SQL 解释

1. 示例：针对 `SELECT COUNT(*) FROM my_table WHERE date BETWEEN '2012-01-01' AND '2012-12-31';"` 进行解释，并添加附加信息 `这条语句有涉及到对数据库的删除和修改吗？`：

关于附加信息：SQL 解释、SQL 优化和 SQL 转换都支持用户在对话时提供额外信息。

![相关图片](https://pic4.zhimg.com/80/v2-1eddf66237b4fcf1b3c17be72d7f9d9f_1440w.webp )

下图为 SQL 解释对话示例，原始用户输入为 `SELECT COUNT(*) FROM my_table WHERE date BETWEEN '2012-01-01' AND '2012-12-31';"`，其他附加信息为 `这条语句有涉及到对数据库的删除和修改吗？`

![示例：SQL 解释功能](https://pic3.zhimg.com/80/v2-d4c4905b37a24f0590018db11134a3f6_1440w.webp)

2. prompt template：用户提供的其他附加信息会被添加到 prompt 当中，参考上面示例，填充完 SQL 解释模板后的请求为：

```json
{'model': 'gpt-3.5-turbo', 
 'messages': [{'role': 'user', 'content': " 请根据以下 SQL input 解释 SQL. 这条语句有涉及到对数据库的删除和修改吗？\n\n SQL input: SELECT COUNT(*) FROM my_table WHERE date BETWEEN '2012-01-01' AND '2012-12-31';"}], 
 'temperature': 0.2, 'top_p': 1.0, 'n': 1, 'max_tokens': 2048, 'stop': None, 'stream': True, 'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'user': None}
```

#### SQL 优化

1. 示例：针对 `SELECT COUNT(*) FROM my_table WHERE date BETWEEN '2012-01-01' AND '2012-12-31';"` 进行优化。其他附加信息为 `这行有什么优化的空间吗？`

![SQL 优化示例](https://pic3.zhimg.com/80/v2-e3c1ae638b61581aea1560bbabe64002_1440w.webp)

2. 该功能同样会使用 prompt template 填充，完整请求内容如下：

```json
{'model': 'gpt-3.5-turbo', 
 'messages': [{'role': 'user', 'content': " 请根据以下 SQL input 提供优化建议. 这行有什么优化的空间吗？\n\n SQL input: SELECT COUNT(*) FROM my_table WHERE date BETWEEN '2012-01-01' AND '2012-12-31';"}], 
 'temperature': 0.2, 'top_p': 1.0, 'n': 1, 'max_tokens': 2048, 'stop': None, 'stream': True, 'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'user': None}

```

#### SQL 转换

1. 示例：我们选择将选中的 SQL 转换为 `python pandas`

![SQL 转换示例](https://pic3.zhimg.com/80/v2-228c3f8ddb859203564fb6b47026bd3e_1440w.webp)

2. prompt template 及发送的请求内容：

```json
{'model': 'gpt-3.5-turbo', 
 'messages': [{'role': 'user', 'content': " 请根据以下 SQL input 进行 SQL 转换. \n\n SQL input: SELECT COUNT(*) FROM my_table WHERE date BETWEEN '2012-01-01' AND '2012-12-31';\n\n 目标 SQL 类型: python pandas"}], 
 'temperature': 0.2, 'top_p': 1.0, 'n': 1, 'max_tokens': 2048, 'stop': None, 'stream': True, 'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'user': None}

```

## 一点心得

1. 目前 Chat2DB 似乎只支持对单个表格的 query。如果想要对多个表格进行操作，需要自己写 prompt。

2. Chat2DB 发送的请求中仅包含了表格名称和表属性，因此向 OPENAI 发送什么样的数据是可控的。不像 Langchain 中的 SQL Agent 等，会在不经意间将表格数据发送给 OPENAI API。

3. Chat2DB 的使用体验在很大程度上依赖于 LLM 的能力。然而，对于 Text2SQL 任务， 笔者测试了 GPT-4 API，Google PaLM2 API 以及其他开源 LLM 模型， 它们的发挥都不够很稳定，潜在的问题包括：

   -  **生成的 SQL 语句不能执行；** 通常回复的 SQL 语法是正确的，但是因为列不存在、表不存在等情况导致 SQL 执行异常。

   -  **生成的 SQL 语句与问题无关；** 有少数情况下，LLM 会对数据进行不必要的求和、取平均等操作。

   导致以上问题的因素可能有：

   - 表属性太多，无法将全部列名及对应表述都添加到 prompt 中；
   - 用户提问不明确；
   - 表格数据本身就是杂乱无章的，表格属性（如列名称等）不直观，非常抽象；

4. 回到文章开头提到的点，Chat2DB 中 AI 的交互功能，几乎就是数据库操作平台 + ChatGPT Web UI ，不同的是 Chat2DB 让你能够在数据库操作平台上直接对 AI 进行提问，省去了复制粘贴的麻烦，同时为你提供了几个简易的 prompt template。
5. 总感觉自己漏掉了某些重大功能，若 Chat2DB 仅有上文提到的 5 种 SQL 相关的 AI 交互功能，似乎不太值 7 千个 star？期待 [官方公布](https://github.com/alibaba/Chat2DB/blob/main/README_CN.md) 6 月底即将发布的重大更新！

 





