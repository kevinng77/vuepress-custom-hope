---
title: Celery
date: 2023-11-24
author: Kevin 吴嘉文
category:
- 知识笔记
---

# Celery

## 快速开始

### 安装

celery 官方推荐使用 RabbitMQ 作为消息中间件。因此我们需要先安装对应的 broker，比如 rabbitmq：

```
docker run -d -p 5672:5672 rabbitmq
```

对于小规模业务，可以考虑使用 backend redis + broker rabbitMQ 的组合。 可以考虑可视化界面安装（redis 后端 + rabbitmq + flower）：

::: details docker-compose 代码

```dockerfile
version: '3.8'

services:
  rabbitmq:
    image: rabbitmq:3-management
    environment:
      - RABBITMQ_DEFAULT_USER=user
      - RABBITMQ_DEFAULT_PASS=password
    ports:
      - "5672:5672"    # RabbitMQ server
      - "15672:15672"  # RabbitMQ management console
  redis:
      image: "bitnami/redis:latest"
      environment:
          - REDIS_PASSWORD=password123
      ports:
          - "6379:6379"
      # volumes:
      #     # - "redis_data:/bitnami/redis/data"
      #     - /bitnami/redis/data
  flower:
    image: mher/flower:2.0.1
    command: celery flower
    environment:
      - FLOWER_PORT=5555
      - CELERY_BROKER=amqp://user:password@rabbitmq:5672//
      - FLOWER_BROKER_API=http://user:password@rabbitmq:15672/api/
    ports:
      - "5555:5555"
    depends_on:
      - rabbitmq
```

:::

安装后，登录 `http://localhost:5555/` 查看 celery 任务管理界面。

#### 服务端

创建一个 `task.py`

```python
from celery import Celery

# 链接 rabbitMQ
app = Celery('tasks', backend='rpc://', broker='amqp://kevin:777777@localhost:5672')


@app.task
def add(x, y):
    return x + y
```

而后运行 celery worker 服务器：

```bash
celery -A tasks worker --loglevel=INFO
```

运行服务器后，开始监听我们定义的两个任务。

::: tip

查看 flower 界面中，可以看到 workers 下有我们启动的监听服务。

:::

#### 客户端

创建文件 `producer.py`：

```python
# 假设我们的 add 任务写在了 tasks.py 文件中
from tasks import add
result = add.delay(4, 4)
```

`result = add.delay(4, 4)` 返回的是一个  [`AsyncResult`](https://docs.celeryq.dev/en/stable/reference/celery.result.html#celery.result.AsyncResult) 实例，该示例可以用 

```python
from celery.result import AsyncResult
from celery_task import cel
result = AsyncResult(id="result_id", app=cel)
```

来构建。获取结果的话可以：

```python
# redis 中用该 id 取结果。
print(result.id)
print(result.ready())  # False
print(result.successful())  # False

# result.get 会阻塞，直到 task 完成，并返回结果
print(result.get(timeout=8))
print(result.successful())  # True
```

::: tip

对于所有的 CeleryResult，我们都需要执行 `get()` 或 `forget()` 来释放资源。

当我们运行了一个 `add.delay(4, 4)` 之后，可以在 flower 的 broker 下面查看到：有一条 Message 被添加到了 `celery` 队列当中，然后对应的 consumers 消费了这个消息。

:::

### 分析

在启动 celery worker 之后，我们可以看到对应的 concurrency 数量：

![image-20231126135511326](/assets/img/celery/image-20231126135511326.png)

以及 rabbitmq 的 queues 相关信息。我们尝试在客户端同时运行 21 次 `add.delay(4,4)`。在 flower 中可以看到，运行后，broker 中显示 `Message = 5`，表示有 5 条消息在等待执行。而后 worker 下显示 `Active=16`, 后变成 `Active=5`。

 **因此，celery 的 `.delay` 方法实际是向队列当中添加了对应函数的任务。我们的 celery worker 会从 queue 当中提取任务执行。** 

## 项目架构

官方示例项目架构

```python
src/
    proj/__init__.py
        /celery.py
        /tasks.py
```

其中 `proj/celery.py` 用于配置 celery app：

```python
from celery import Celery
import os


app = Celery('proj',
             broker=os.environ.get('CELERY_BROKER_URL', 'amqp://user:password@localhost:5672//'),
             backend=os.environ.get('CELERY_RESULT_BACKEND','redis://:password123@localhost:6379/1'),
             include=['proj.tasks'])

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
)

if __name__ == '__main__':
    app.start()
```

`proj/tasks.py` 用于编辑任务：

```python
from .celery import app

@app.task
def add(x, y):
    return x + y

@app.task
def mul(x, y):
    return x * y
```

启动 worker 服务：

```bash
celery -A proj worker -l INFO
```

或者在后台启动服务：

```bash
celery multi start w1 -A proj -l INFO
```

## Celery 进阶

### task 配置

task 可选参数：[文档](https://docs.celeryq.dev/en/stable/userguide/tasks.html#list-of-options)。

logging 可以直接用 print。或者参考[文档](https://docs.celeryq.dev/en/stable/userguide/tasks.html#logging)。

所有 task 都可以通过自定义 class 来批量配置参数：

```python
import celery

class MyTask(celery.Task):
	autoretry_for = (TypeError,)
    max_retries = 5
    retry_backoff = True
    retry_backoff_max = 700
    retry_jitter = False
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        print('{0!r} failed: {1!r}'.format(task_id, exc))

@app.task(base=MyTask)
def add(x, y):
    raise KeyError()
```

task 添加 bind 参数后，可以调用 `self` 中的方法，比如 `retry()`

```python
@app.task(bind=True)
def send_twitter_status(self, oauth, tweet):
    try:
        twitter = Twitter(oauth)
        twitter.update_status(tweet)
    except (Twitter.FailWhaleError, Twitter.LoginError) as exc:
        # overrides the default delay to retry after 1 minute
        raise self.retry(exc=exc, countdown=60, max_retries=5)
```

当任务中涉及到调用其他 API 任务时，可以考虑添加 `retry_backoff=True` 选项（参考[文档](https://docs.celeryq.dev/en/stable/userguide/tasks.html#automatic-retry-for-known-exceptions)）。

#### 任务状态控制

task 可以使用 broker 中的 basic_reject 方法：

```python
from celery.exceptions import Reject

@app.task(bind=True, acks_late=True)
def requeues(self):
    if not self.request.delivery_info['redelivered']:
        raise Reject('no reason', requeue=True)
    print('received two times')
```

可以用 `from celery.exceptions import Ignore` 的 `raise Ignore()` 来丢弃任务。

::: tip

每一个 task 的 `__init__` 方法只会被调用 1 次。 **因此可以用来缓存一些资源，比如数据库链接状态等** ：

```python
from celery import Task

class DatabaseTask(Task):
    _db = None

    @property
    def db(self):
        if self._db is None:
            self._db = Database.connect()
        return self._db
```

然后调用时候用：

```python
from celery.app import task

@app.task(base=DatabaseTask, bind=True)
def process_rows(self: task):
    for row in self.db.table.all():
        process_row(row)
```

[参考文档](https://docs.celeryq.dev/en/stable/userguide/tasks.html#instantiation)

:::

#### 任务钩子

任务可以添加一些钩子，参考[文档](https://docs.celeryq.dev/en/stable/userguide/tasks.html#handlers)。

#### 执行任务

[官方文档](https://docs.celeryq.dev/en/stable/userguide/calling.html)

1. 调度时建议设置好时间预期：

```python
from datetime import datetime, timedelta, timezone

tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
add.apply_async((2, 2), eta=tomorrow)
```

2. 可以使用 `on_message` 来监听对应任务的进度：

```python
# 服务器
@app.task(bind=True)
def hello(self, a, b):
    time.sleep(1)
    
 # 用 self.update_state 会触发 on_message 对应函数
self.update_state(state="PROGRESS", meta={'progress': 50})
    time.sleep(1)
    self.update_state(state="PROGRESS", meta={'progress': 90})
    time.sleep(1)
    return 'hello world: %i' % (a+b)
```

客户端调用时：

```python
def on_raw_message(body):
    print(body)

a, b = 1, 1
r = hello.apply_async(args=(a, b))
print(r.get(on_message=on_raw_message, propagate=False))
```

3. 添加 `ignore_result=True` 来节省时间：`result = add.apply_async((1, 2), ignore_result=True)`



#### 提示

1. 不要在一个 task 中调用另一个 task。如果要那么做的话，请使用 signature。
2. celery 是一个分布式系统，我们不能知道哪个进程或者哪台机器将会执行这个任务。

### Task-flows

详细的 workflow 可以在 [celery 官网](https://docs.celeryq.dev/en/stable/userguide/canvas.html#guide-canvas)查询

1.  **Groups** 

一个 group 并行调用任务列表，返回的结果是有序的。执行下列 groups 任务，broker 当中会添加 10 个 message。

```python
from celery import group
from proj.tasks import add

group(add.s(i, i) for i in range(10))().get()
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

2.  **Chains** 

接受多个 signature 的结果，而后传送给一个 message 执行任务。

```python
from celery import chain
from proj.tasks import add, mul

# (4 + 4) * 8
chain(add.s(4, 4) | mul.s(8))().get()
# 64
```

3.  **Chords** 

```python
from celery import chord
from proj.tasks import add, xsum

chord((add.s(i, i) for i in range(10)), xsum.s())().get()
```

### Task Routing

不同的任务，可以安排在不同的队列上。

```python
app.conf.update(
    task_routes = {
        'proj.tasks.objectCheck': {'queue': 'objectCheck'},
        'proj.tasks.caption': {'queue': 'caption'},
        'proj.tasks.training': {'queue': 'training'}  # 声明每个 task 对应的 queue
    },
    task_queues = (
        Queue('default'),
        Queue('objectCheck'),   # 声明有这个 queue
        Queue('caption'),
        Queue('training'),
        # rabbitmq 支持特殊 routing 设计
        Queue('tasks', Exchange('tasks', type='direct'), 	
              routing_key='tasks',
              queue_arguments={'x-max-priority': 10}),
	)
)
```

默认情况下，使用 `celery -A proj worker` 会监听所有的 queue。可以使用：

```bash
celery -A proj worker -Q objectCheck --hostname=object@%h
```

来监听不同的任务。也可以设置不同的 `exchange`，`routing_key` 等。具体查看[官方文档](https://docs.celeryq.dev/en/stable/userguide/routing.html#special-routing-options)。

关于 queue 与 routing_key 请查看 rabbitmq

### Worker 启动

对于同一份 celery 项目代码，可以通过在终端传入不同的 `-Q` 参数来指定每个 worker 需要监听的队列。

可以通过 [add_consumers](https://docs.celeryq.dev/en/stable/userguide/workers.html#queues-adding-consumers) 来添加不同消息队列的 consumer：

```
app.control.add_consumer(
    queue='baz',
    exchange='ex',
    exchange_type='topic',
    routing_key='media.*',
    options={
        'queue_durable': False,
        'exchange_durable': False,
    },
    reply=True,
    destination=['w1@example.com', 'w2@example.com'])
```

worker 启动部分参数：

+ `--logfile=%p.log`
+ `--loglevel=INFO`
+ `--concurrency=10`
+ `-n worker1@%h`
+ `--autoscale=10,3`  (always keep 3 processes, but grow to
        10 if necessary)

+ `-Q foo,bar,baz`



强制关闭所有 worker：

```bash
ps auxww | awk '/celery worker/ {print $2}' | xargs kill -9
```



### 落地代办

请查看官网的其他 [how-to-guide](https://docs.celeryq.dev/en/stable/userguide/index.html), 如 [Security](https://docs.celeryq.dev/en/stable/userguide/security.html)， [Optimizing](https://docs.celeryq.dev/en/stable/userguide/optimizing.html)， [Configuration and defaults](https://docs.celeryq.dev/en/stable/userguide/configuration.html) 等。

- [ ] flower 集成其他可视化平台
- [ ] 如何设置失败重试







