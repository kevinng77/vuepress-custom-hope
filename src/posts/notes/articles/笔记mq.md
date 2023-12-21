---
title: RabbitMQ
date: 2023-11-18
author: Kevin 吴嘉文
category:
- 知识笔记
---



# Message Queue

MQ 选型

当消息量不大时 RabbitMQ 效果挺不错。但 RabbitMQ 的 erlang 导致他 debug 成本高。相对的 Kafka，rocketMQ 的分布式+支持堆积消息优势性大。但似乎 rabbitMQ 更容易部署。

# RabbitMQ

## RabbitMQ 安装

参考[官网](https://www.rabbitmq.com/getstarted.html)

```bash
docker run -it --rm --name rabbitmq -e RABBITMQ_DEFAULT_USER=kevin -e RABBITMQ_DEFAULT_PASS=777777 -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

访问 UI 管理界面：

```bash
http://192.168.1.104:15672/
```

python 客户端安装：

```bash
python -m pip install pika --upgrade
```

## 快速开始

### 基础使用

最基础的消息队列架构为：

![img](https://www.rabbitmq.com/assets/img/tutorials/python-one-overall.png)

其中由生产者、队列、消费者组成。

基础生产者需要：

1. 建立链接

```python
import pika, sys, os

# 建立链接
    credentials = pika.PlainCredentials(username='kevin', password='777777')
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', credentials=credentials))
    channel = connection.channel()
```

2. 声明使用的 queue，[更详细的 queue 指南参考官网](https://www.rabbitmq.com/queues.html)

```python
    channel.queue_declare(queue='hello',
                        durable=False,  # 队列里面的消息是否化磁盘) 默认况消息存储在内存中
                        exclusive=False,  # 该队列是否只供一个消费者进行消费 是否进行消息共享，true 可以多个消费者消费 false:只能一个消费者消费
                        auto_delete=False  # 是否自动删除 最后一个消费者端开连接以后 该队一句是否自动删除 true 自动删除 false 不自动删除
                        )
```

3. 发送消息：

```python
# 通过 exchange 将消息发送到 queue
# 因为是 hello world 教程，我们在这里直接将消息发送到队列中。
# If we send a message to non-existing location, RabbitMQ will just drop the message.
channel.basic_publish(exchange='',
                      routing_key='hello',  # The queue name needs to be specified here
                      body='Hello World!')
```

4. 接收端

::: details 基础消费者

```python
#!/usr/bin/env python
import pika, sys, os

def main():
	# 建立链接等操作与 sender 相同

    def callback(ch, method, properties, body):
        print(f" [x] Received {body}")
        
        
    channel.basic_consume(queue='hello',
                        auto_ack=True,
                        on_message_callback=callback)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()
```

:::

### work queue

![img](https://www.rabbitmq.com/assets/img/tutorials/python-two.png)

多个工作者轮询接收并处理消息，每个消息准确地被处理一次。

#### 消息丢失处理

1. [message acknowledgement](https://www.rabbitmq.com/confirms.html)

在使用 RabbitMQ 时，如果工作进程在任务完成前终止，可能会导致消息丢失。为防止这种情况，RabbitMQ 提供了消息确认机制。消费者处理完消息后，需要向 RabbitMQ 发送确认（ack）。如果消费者在未发送确认的情况下死亡，RabbitMQ 会重新排队这些消息，并可能分配给其他在线消费者。

默认情况下，消息确认是手动的。在之前的示例中，我们通过设置 `auto_ack=True` 来关闭此功能，但现在应该移除此标志，并在任务完成后发送确认。以下是一个示例代码：

```python
def callback(ch, method, properties, body):
    print(f" [x] Received {body.decode()}")
    time.sleep(body.count(b'.'))
    print(" [x] Done")
    ch.basic_ack(delivery_tag = method.delivery_tag)

channel.basic_consume(queue='hello', on_message_callback=callback)
```

这样即使在处理消息时强行终止工作进程，也不会丢失消息。确认必须通过接收消息的同一通道发送，否则会导致协议异常。

2.  **消息持久化** 

为防止 RabbitMQ 服务器停止时任务丢失，需将队列和消息标记为持久化。首先，声明队列为持久化：

```python
channel.queue_declare(queue='hello', durable=True)
```

如果已有同名非 **持久化队列** ，需使用不同名称，例如 `task_queue`：

```python
channel.queue_declare(queue='task_queue', durable=True)
```

此更改需同时应用于生产者和消费者代码。接下来，将 **消息标记为持久化** ，设置 `delivery_mode` 为 `pika.spec.PERSISTENT_DELIVERY_MODE`：

```python
channel.basic_publish(exchange='',
                      routing_key="task_queue",
                      body=message,
                      properties=pika.BasicProperties(
                         delivery_mode = pika.spec.PERSISTENT_DELIVERY_MODE
                      ))
```

注意，即使消息标记为持久化，也无法完全保证不丢失，因为 RabbitMQ 可能未立即将消息保存到磁盘。如需更强保证，可使用发布者确认机制。

3.  **轮询策略** 

换句话说，直到工作者处理并确认了前一个消息之前，不要向其分发新消息。

```python
# worker 中定义
channel.basic_qos(prefetch_count=1)
```

## 交换机 Exchange

### 发布订阅模式（fanout 交换机）

![img](https://www.rabbitmq.com/assets/img/tutorials/exchanges.png)



RabbitMQ 的核心：生产者不直接向队列发送消息，而是只能发送到交换机。

交换机类型包括  `direct`, `topic`, `headers` and `fanout`。我们关注  `fanout` 类型（用于广播），创建一个名为 `logs` 的交换机：

```python
channel.exchange_declare(exchange='logs', exchange_type='fanout')
```

#### 发送端

扇出交换机将消息广播到所有队列。

我们之前使用的是默认交换机（用空字符串 "" 表示），它将消息路由到指定名称的队列。现在，我们改为使用命名的交换机 `logs`：

```python
channel.basic_publish(exchange='logs', routing_key='', body=message)
```

使用发送确认：

```python
# Turn on delivery confirmations
channel.confirm_delivery()

try:
    channel.basic_publish(exchange='test',
                          routing_key='test',
                          body='Hello World!',
                          properties=pika.BasicProperties(content_type='text/plain',
                                                          delivery_mode=pika.DeliveryMode.Transient)):
    print('Message publish was confirmed')
except pika.exceptions.UnroutableError:
    print('Message could not be confirmed')
```

#### 接收端

在接收端，需要定义一个 queue 来绑定到对应的 exchange 上。

1. 定义一个临时队列

```bash
result = channel.queue_declare(queue='', exclusive=True)
```

2. 绑定 queue 和 exchange

```python
channel.queue_bind(exchange='logs',
                   queue=result.method.queue)
```

### Routing 路由（direct 交换机）

![img](https://www.rabbitmq.com/assets/img/tutorials/python-four.png)

使用 direct 模式时，队列可以和交换机绑定。该模式下，可以自由指定哪些消息要去到哪个队列。

::: tip

可以想象交换机为一个 mapping，其中储存了 `routing_key:queue` 的键值对。

:::

#### 通道配置

```python
channel.exchange_declare(exchange='direct_logs', exchange_type='direct')
```

#### 发送端

比如往 `direct_logs` 交换机发送 `info` 类型的信息。

```python
channel.basic_publish(
    exchange='direct_logs', routing_key="info", body=message)
```

#### 接收端

首先定义 exchange 和 queue：

```python
channel.exchange_declare(exchange='direct_logs', exchange_type='direct')
result = channel.queue_declare(queue='', exclusive=True)
queue_name = result.method.queue
```

绑定 exchange 和 queue，让这个 queue 接受 `info` 类型的信息：

```python
channel.queue_bind(
        exchange='direct_logs', queue=queue_name, routing_key="info")
```

一个队列可以和交换机绑定多个 `routing_key`。绑定多个 `routing_key` 需要多次调用 `channel.queue_bind`。

### 主题模式（topic 交换机）

![](https://www.rabbitmq.com/assets/img/tutorials/python-five.png)

相对于 direct 交换机模式，一个 queue 可以用不同的 tag 来命名，比如一个 queue 可以命名为 `<celerity>.<color>.<species>`，其中包含三种 tag。topic 模式让我们能够使用 `*`（匹配任意 1 个 tag）, `#`（匹配 0+ 个任意的 tag） 对 tag 进行匹配。

比如发布到 `lazy.orange.rabbit` 的消息会被主题 `*.orange.*` 接收到。

#### 通道配置

```python
channel.exchange_declare(exchange='topic_logs', exchange_type='topic')
```

#### 发送方

```python
routing_key = "lazy.orange.rabbit"
channel.basic_publish(
    exchange='topic_logs', routing_key=routing_key, body=message)
```

#### 接收端

```python
# 声明任意的队列
result = channel.queue_declare('', exclusive=True)
queue_name = result.method.queue

# 绑定队列到对应的 binding_key
binding_key = "*.orange.*"
channel.queue_bind(
        exchange='topic_logs', queue=queue_name, routing_key=binding_key)
```

### RPC

![](https://www.rabbitmq.com/assets/img/tutorials/python-six.png)

当接收方的 worker 需要执行 remote server 的代码，并等待执行结果时，我们需要用到 RPC。

进行远程处理时，需要明确的指出：

+ 对 remote 消息进行 error handle 处理（比如长时间未回复）
+ 明确区分 worker 当中的本地执行任务和远程执行任务。
+ 请做好详细的文档注释

#### 通道构建

```python
# 服务端定义主通道
channel.queue_declare(queue='rpc_queue')
```

#### 服务端

在这个机器上，我们可能需要运行一些耗时的任务。当接收到 client 发出的消息后，开始任务的工作，而后将工作结果 `publish` 到回复队列中去。

```python
def on_request(ch, method, props, body):
    n = int(body)
    response = fib(n)  # run your slow tasks here
    import time 
    time.sleep(10)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,  # 特定回复通道
                     properties=pika.BasicProperties(correlation_id = \
                                                         props.correlation_id),
                     # 这个执行结果对应的 UUID
                     body=str(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)



channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='rpc_queue', on_message_callback=on_request)

channel.start_consuming()
```

#### 客户端



```python
# 客户端定义回复消息通道
result = self.channel.queue_declare(queue='', exclusive=True)
self.callback_queue = result.method.queue

# 回复通道也需要定义好 on_response 函数，用来接受回复答案。
self.channel.basic_consume(
    queue=self.callback_queue,
    on_message_callback=self.on_response,
    auto_ack=True)

self.response = None
self.corr_id = None

def on_response(self, ch, method, props, body):
    if self.corr_id == props.correlation_id:
        self.response = body

def call(self, n):
    self.response = None
    
    # 对于每一次 call，都需要一个单独的 UUID 来确保回复答案的准确性
    self.corr_id = str(uuid.uuid4())
    self.channel.basic_publish(
        exchange='',
        routing_key='rpc_queue',
        properties=pika.BasicProperties(
            reply_to=self.callback_queue,
            correlation_id=self.corr_id,
        ),
        body=str(n))
    print("wait for response?")
    self.connection.process_data_events(time_limit=None)
    print("wait for response?")

    return int(self.response)

```



## 落地推荐

### DLQ 死信队列

[rabbitMQ 官方](https://www.rabbitmq.com/dlx.html)

1. 消息 TTL 过期
2. 队列达到最大长度
3. 消息被拒绝（basic.reject 或 basic.nack） 并且 requeue=false



pub-sub 模式在实际落地中，建议的操作：

1. [数据安全相关 - Publisher Confirms and Consumer Acknowledgements](https://www.rabbitmq.com/confirms.html)
2. [Production Checklist](https://www.rabbitmq.com/production-checklist.html)
3. [Monitoring](https://www.rabbitmq.com/monitoring.html).







