---
title: sanic |高并发 Python web 工具
date: 2023-03-19
author: Kevin 吴嘉文
category:
- 知识笔记
---

# sanic

sanic 因为 Sanic 不仅仅是一个  **框架** ，它还是一个  **Web 服务器** 。网上对他的标签有：异步，快，开发效率高等

[awesome sanic](https://github.com/mekicha/awesome-sanic) [中文用户指南](https://sanic.dev/zh/) [API doc](https://sanic.readthedocs.io/en/stable/)

 

## 快速开始

```bash
pip install sanic
```

Sanic 旨在提高性能、灵活性和易用性。 写一个简单的 api 服务（官方推荐在 server.py 中来实例化你的 app）：

```python
# server.py
from sanic import Sanic
from sanic.response import text

app = Sanic("MyHelloWorldApp")

@app.get("/")
async def hello_world(request):
    return text("Hello, world.")
```

::: warning

除非有特殊目的，请尽量使用 `async` 来声明响应函数。  **必须**  使用 `Response` 或继承自 `Response` 的类作为响应类型。不能直接 `return something`

:::

sanic 可以用于生产环境：

```bash
sanic server.app
```

## 拓展

 [Sanic Extensions](https://sanic.dev/zh/plugins/sanic-ext/getting-started.html) 包括了各种易用的功能：

-  **OpenAPI**  使用 Redoc 和/或 Swagger 的文档
-  **CORS**  保护
-  **依赖注入**  路由处理程序
- Request 参数  **检查** 
- 自动创建 `HEAD`, `OPTIONS`, 和 `TRACE` 响应函数
- 响应序列化

安装

```text
pip install sanic[ext]
```

## 入门

### 注册表

可以在其他 py 文件中，通过注册表的方式获得对应的 app 实例。

```python
# ./path/to/server.py
from sanic import Sanic

app = Sanic("my_awesome_server")

# ./path/to/somewhere_else.py
from sanic import Sanic

app = Sanic.get_app("my_awesome_server")
```

### 配置

[官方文档](https://sanic.dev/zh/guide/deployment/configuration.html)

添加 sanic 配置可以通过：

1. 直接赋值：`app.config.DB_NAME = "appdb"`

2. 通过系统环境变量赋值：

```bash
export SANIC_REQUEST_TIMEOUT=10
#  python 中 print(app.config.REQUEST_TIMEOUT)
```

3. 通过 python 字典：`app.update_config({"A": 1, "B": 2})`

4. 通过 python 类：

```python
class MyConfig:
    A = 1
    B = 2

app.update_config(MyConfig)
```

5. 通过 python 文件：

```python
app.update_config("/path/to/my_config.py")
```

其中，`my_config.py` 中定义好要配置的变量，如：

```python
# my_config.py
A = 1
B = 2
```

常见的配置有：

请求超时设置：`  app.config.RESPONSE_TIMEOUT = response_timeout`

CORS 配置：

```python
def configure_cors(
    app: Sanic, cors_origins: Union[Text, List[Text], None] = "*"
) -> None:
    """Configure CORS origins for the given app."""

    # Workaround so that socketio works with requests from other origins.
    # https://github.com/miguelgrinberg/python-socketio/issues/205#issuecomment-493769183
    app.config.CORS_AUTOMATIC_OPTIONS = True
    app.config.CORS_SUPPORTS_CREDENTIALS = True
    app.config.CORS_EXPOSE_HEADERS = "filename"

    CORS(
        app, resources={r"/*": {"origins": cors_origins or ""}}, automatic_options=True
    )
```

### 响应函数 Handlers

```python
@app.get("/")
async def hello_world(request):
    return text("Hello, world.")
```

::: warning

除非有特殊目的，请尽量使用 `async` 来声明响应函数。  **必须**  使用 `Response` 或继承自 `Response` 的类作为响应类型。不能直接 `return something`

:::

### 请求

响应函数接收 request 参数，可以通过以下方式接收不同类型的请求参数：

```python
request.json  # 接收 json 对象
request.body  # 接收 raw 对象
request.form  # 接收表单对象， 用 .get 方法获取表单内容
request.file  # 接收文件对象
```

### 上下文 Context

#### 请求上下文(Request context)

具体查看中间件(Middleware)

案例：

```python
@app.middleware("request")
async def run_before_handler(request):
    request.ctx.user = await fetch_user_by_token(request.token)

@app.route('/hi')
async def hi_my_name_is(request):
    return text("Hi, my name is {}".format(request.ctx.user.name))
```

#### 连接上下文(Connection context)

```python
@app.on_request
async def increment_foo(request):
    if not hasattr(request.conn_info.ctx, "foo"):
        request.conn_info.ctx.foo = 0
    request.conn_info.ctx.foo += 1

@app.get("/")
async def count_foo(request):
    return text(f"request.conn_info.ctx.foo={request.conn_info.ctx.foo}")
```



### 响应 Response

sanic 内置的响应方式包括返回文本， html，json，文件，streaming，raw 等方式。在 `sanic.response` 中查看

### 路由 Routing

常用的路由方式如 `@app.post()`, `@app.get()` 等。路由支持传参

```python
@app.get("/foo/<foo_id:uuid>")
async def uuid_handler(request, foo_id: UUID):
    return text("UUID - {}".format(foo_id))

```

参数类型包括 `str, int, float, path, uuid` 等。可以通过正则表达式来自定义自己想要的参数类型 [官方指南](https://sanic.dev/zh/guide/basics/routing.html#%E8%B7%AF%E7%94%B1%E5%8F%82%E6%95%B0-path-parameters)

正则表达式案例：

```python
@app.get(r"/<foo:[a-z]{3}.txt>")                # 全模式匹配
@app.get(r"/<foo:([a-z]{3}).txt>")              # 定义单个匹配组
@app.get(r"/<foo:(?P<foo>[a-z]{3}).txt>")       # 定义单个命名匹配组
@app.get(r"/<foo:(?P<foo>[a-z]{3}).(?:txt)>")   # 用一个或多个不匹配组定义单个命名匹配组
```

比如要提取 `/image/123456789.jpg` 中的数字作为参数，可以使用 

```python
app.route(r"/image/<img_id:(?P<img_id>\d+)\.jpg>")
```

### 监听器 Listeners

支持在以下 8 个 server 运行节点进行监听和操作：

- `main_process_start`
- `main_process_stop`
- `reload_process_start`
- `reload_process_stop`
- `before_server_start`
- `after_server_start`
- `before_server_stop`
- `after_server_stop`

### 中间件 Middleware

[文档](https://sanic.dev/zh/guide/basics/middleware.html#%E5%90%AF%E7%94%A8-attaching-middleware) 。中间件支持在 http 流的生命周期中挂载额外功能。以下为中间件响应的函数：

```python
@app.middleware('response')
async def prevent_xss(request, response):
    response.headers["x-xss-protection"] = "1; mode=block"

```

或者 

```python
@app.middleware("request")
async def middleware_1(request):
    print("middleware_1")


@app.middleware("request")
async def middleware_2(request):
    print("middleware_2")


@app.middleware("response")
async def middleware_3(request, response):
    print("middleware_3")


@app.middleware("response")
async def middleware_4(request, response):
    print("middleware_4")
    
@app.get("/handler")
async def handler(request):
    print("~ handler ~")
    return text("Done.")
```

sanic 中的 HTTP 流执行的顺序为：接收请求，request 中间件，response 中间件，响应函数。因此请求 `/handler` 时候，日志端会打印：

```python
middleware_1
middleware_2
~ handler ~
middleware_4
middleware_3
[INFO][127.0.0.1:44788]: GET http://localhost:8000/handler  200 5

```

### 后台任务



### channel



```python
def register(
    input_channels: List["InputChannel"], app: Sanic, route: Optional[Text]
) -> None:
    """Registers input channel blueprints with Sanic."""

    async def handler(message: UserMessage) -> None:
        await app.ctx.agent.handle_message(message)

    for channel in input_channels:
        if route:
            p = urljoin(route, channel.url_prefix())
        else:
            p = None
        app.blueprint(channel.blueprint(handler), url_prefix=p)

    app.ctx.input_channels = input_channels
```

## 高级

### 蓝图 Bluprint

蓝图是应用中可以作为子路由的对象。蓝图定义了同样的添加路由的方式，您可以将一系列路由注册到蓝图上而不是直接注册到应用上，然后再以可插拔的方式将蓝图注册到到应用程序。

蓝图对于大型应用特别有用。在大型应用中，您可以将应用代码根据不同的业务分解成多个蓝图。

```python
# ./my_blueprint.py
from sanic.response import json
from sanic import Blueprint

bp = Blueprint("my_blueprint")


@bp.route("/")
async def bp_root(request):
    return json({"my": "blueprint"})
```



```python
from sanic import Sanic
from my_blueprint import bp

app = Sanic(__name__)
app.blueprint(bp)
```

