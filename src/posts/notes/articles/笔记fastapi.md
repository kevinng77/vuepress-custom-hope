---
title: fastapi | 更方便的 Python web 工具
date: 2023-04-10
author: Kevin 吴嘉文
category:
- 知识笔记
---

# FastAPI

[官方指南 link](https://fastapi.tiangolo.com/)

> FastAPI 是一个用于构建 API 的现代、快速（高性能）的 web 框架，使用 Python 3.6+ 并基于标准的 Python 类型提示。
>
> 个人感受到的 FastAPI 使用特点：
>
> 1. 使用模型统一对接受以及输出数据进行格式管理。
> 2. 自动类型检验，以及完善的检验流程和错误信息处理方式。

## 快速开始

### 环境安装

```sh
pip install "fastapi[all]"
pip install "uvicorn[standard]"  # 安装 ASGI 服务器
```

### 快速案例

```python
# main.py
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

@app.get("/")
async def read_root():
    return {"Hello": "World"}

# 路径参数 item_id 的值将作为参数 item_id 传递给你的函数。
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    # 提供参数类型后 item_id 会被自动解析成 int 类型
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}
```

::: info

注意点：

+ 如果你的代码里会出现 `async` / `await`，请使用 `async def`
+ 提供参数类型后 `item_id` 会被自动解析成 `int` 类型，如果路径传参错误，那么会返回一个报错结果。
+ 路径参数 `item_id` 的值将作为参数 `item_id` 传递给你的函数。

:::

开始服务，以下替换 main 为你的 python 文件名称。修改 app 为 FastAPI 实例名称。

```sh
uvicorn main:app --reload
```

对于设置好的 get 服务，可以通过直接访问 url 来测试： http://127.0.0.1:8000/items/5?q=somequery

### FastAPI 特性

#### 交互式文档界面

访问  http://127.0.0.1:8000/docs 或  http://127.0.0.1:8000/redoc，能够看到 fastapi 自动整理好的 api 信息，包括结构数据结构，回复内容格式等信息。

在界面中，可以对你写好的 API 进行在线测试。

#### 编辑器支持

FastAPI 基于 Pydanic 和 Starlette，因此当你在文档中定义了结构体后，在 FastAPI 的任意地方都能支持 **数据类型校验** 以及 **自动补全** ：

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None
```

基于  **Starlette** ，可以使用一些内容：

-  **支持 WebSocket**  。
-  **支持 GraphQL**  。
- 后台任务处理。
- Startup 和 shutdown 事件。
- 测试客户端基于 HTTPX。
-  **CORS** , GZip, 静态文件, 流响应。
- 支持  **Session 和 Cookie**  。
- 100% 测试覆盖率。
- 代码库 100% 类型注释。

## 教程梳理

### 通用 API

```python
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

@app.get("/items/10086")
async def read_item():
    return {"item_id": "hello"}

# 路径参数 item_id 的值将作为参数 item_id 传递给你的函数。
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    # 提供参数类型后 item_id 会被自动解析成 int 类型
    return {"item_id": item_id, "q": q}

@app.post(
    "/items/",
    response_model=Item,
    summary="Create an item",
    description="Create an item with all the information, name, description, price, tax and a set of unique tags",
    response_description="The created item",
)
    """
    This is a doc string.
    Create an item with all the information:

    -  **name** : each item must have a name
    -  **description** : a long description
    -  **price** : required
    -  **tax** : if the item doesn't have tax, you can omit this
    -  **tags** : a set of unique tag strings for this item
    """
async def create_item(item: Item):
    return item
```

 **API 元数据：**  可以通过 `summary` 和 `description` 来添加 API 相关信息。当然也可以通过函数常用的 `docstring` 来为函数写注释信息。

 **数据类型校验：** 对于上面的例子，如果发送请求到 `http://127.0.0.1:8000/items/6.1?q=somequery`

 **自动类型转换：**  函数参数部分声明了 `item_id: int`，提供参数类型后 `item_id` 会被自动解析成 `int` 类型；如果是 `item_id: str`，那么 `item_id` 就会被解析成字符串

 **API 定义顺序很重要：**  如上例子， `/items/10086` 必须在 `items/{item_id}` 之前提供。

 **查询参数：** 声明不属于路径参数的其他函数参数时，它们将被自动解释为"查询字符串"参数，需要通过 `http://127.0.0.1:8000/items/?skip=0&limit=10` 方式来访问额外参数

```python
@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```

针对可选参数，请使用 `q: Union[str, None] = None` 来统一进行数据格式声明。

### API 额外功能

参数选项

如下，当我们希望用户请求的参数在一个可选的范围内，可以用 Enum 的方式来定义选项。当发送请求 `/models/{model_name}` 中的 `model_name` 不在枚举的 `ModelName` 中时，fastAPI 会进行参数检验，并返回 `value is not a valid enumeration member; permitted: 'alexnet', 'resnet', 'lenet'`。

```python
from enum import Enum

from fastapi import FastAPI

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

app = FastAPI()

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    # 路径参数的值将是一个枚举成员
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}
    # 用 .value 获取值
    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}
```

### 请求体 接收 JSON 数据

通过 `BaseModel` 来定义接受的请求体格式。如下定义方式，请求体必须包括 `name` 和 `price` 字段 

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None
    
    class Config:  # 可以选择添加元数据
        schema_extra = {
            "example": {
                "name": "Foo",
                "description": "A very nice Item",
                "price": 35.4,
                "tax": 3.2,
            }
        }
@app.post("/post/test")
def update_item(item: Item):
    return {"item_name": item.name}
```

#### 类型转换

如果你的数据中包含一些如 `datetime` 的与 JSON 不兼容的数据类型，那么可以使用 `jsonable_encoder` 来进行处理。

```python
from fastapi.encoders import jsonable_encoder
fake_db = {}


class Item(BaseModel):
    title: str
    timestamp: datetime
    description: str | None = None

app = FastAPI()

@app.put("/items/{id}")
def update_item(id: str, item: Item):
    json_compatible_item_data = jsonable_encoder(item)
    fake_db[id] = json_compatible_item_data
```



#### 参数校验

使用 query 对输入的字符串进行长度校验，或者要求他满足正则表达式。

```python {8-12}
from typing import Union
from fastapi import FastAPI, Query
app = FastAPI()

@app.get("/items/")
async def read_items(
    q: Union[str, None] = Query(
        default=None, min_length=3, max_length=50,
        title="Query string",
        description="Query string for the items to search in the database that have a good match",
		regex="^fixedquery$"
    )
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results
```

同样的 query 也支持对数值参数进行大小校验，可以使用 `size: float = Query(gt=0, lt=10.5),`

你可能会更喜欢统一在请求体中对参数进行管理，可以使用 `Field`：

```python {11}
from typing import Union

from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Union[str, None] = Field(
        default=None, title="The description of the item", max_length=300
    )
    price: float = Field(gt=0, description="The price must be greater than zero")
    tax: Union[float, None] = None
```

针对 JSON 嵌套情况，可以使用嵌套的 Model 来统一定义：

```python
class Image(BaseModel):
    url: str
    name: str

class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None
    tags: Set[str] = set()
    image: Union[Image, None] = None
```

除此外，`pydantic` 还提供其他参数类型来方便我们进行校验，如 `HttpUrl`。同时也可以使用 python 的 `datetime`， `UUID` 等数据类型。

#### 返回结果

同样可以对返回结果进行定义

```python
class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None
    tags: List[str] = []

@app.post("/items/", response_model=Item)
async def create_item(item: Item) -> Any:
    return item

@app.get("/items/", response_model=List[Item])
async def read_items() -> Any:
    return [
        {"name": "Portal Gun", "price": 42.0},
        {"name": "Plumbus", "price": 32.0},
    ]
```

### 接收表单数据

```python
from fastapi import FastAPI, Form

app = FastAPI()


@app.post("/login/")
async def login(username: str = Form(), password: str = Form()):
    return {"username": username}
```

### 接收请求文件

```python
from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # 异步状态下，使用以下方式获取文件内容
    contents = await file.read()
    # 同步状态下用 contents = file.file.read()
    return {"filename": file.filename}
```

多文件操作

```python
@app.post("/files/")
async def create_files(files: list[bytes] = File()):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):
    return {"filenames": [file.filename for file in files]}
```

::: info

在同一个请求中接收数据和文件时，应同时使用 `File` 和 `Form`。

:::



### 依赖项

依赖项类似于类型注释，能够在过个接口需要使用同样结构数据时使用功能。依赖项能够套娃使用。

```python
from typing import Union
from fastapi import Cookie, Depends, FastAPI

app = FastAPI()

def query_extractor(q: Union[str, None] = None):
    return q

def query_or_cookie_extractor(
    q: str = Depends(query_extractor),
    last_query: Union[str, None] = Cookie(default=None),
):
    if not q:
        return last_query
    return q


@app.get("/items/")
async def read_query(query_or_default: str = Depends(query_or_cookie_extractor)):
    return {"q_or_cookie": query_or_default}
```

当然除了函数外，也可以使用类作为依赖项：

```python
from fastapi import Depends, FastAPI

app = FastAPI()


fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit


@app.get("/items/")
async def read_items(commons: CommonQueryParams = Depends(CommonQueryParams)):
    response = {}
    if commons.q:
        response.update({"q": commons.q})
    items = fake_items_db[commons.skip : commons.skip + commons.limit]
    response.update({"items": items})
    return response
```

依赖项不一定需要进行参数传递，可以用在装饰器中，进行依赖管理。

```python
from fastapi import Depends, FastAPI, Header, HTTPException

app = FastAPI()


async def verify_token(x_token: str = Header()):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def verify_key(x_key: str = Header()):
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key


@app.get("/items/", dependencies=[Depends(verify_token), Depends(verify_key)])
async def read_items():
    return [{"item": "Foo"}, {"item": "Bar"}]
```

### 安全操作

[参考官方](https://fastapi.tiangolo.com/zh/tutorial/security/first-steps/)

#### CORS

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def main():
    return {"message": "Hello World"}
```

### StaticFiles

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
```

## 部署

可以使用 

```sh
uvicorn main:app --host 0.0.0.0 --port 80
```

来发布你的应用

## 其他更多

参考 fastapi [高级指南](https://fastapi.tiangolo.com/zh/advanced/)