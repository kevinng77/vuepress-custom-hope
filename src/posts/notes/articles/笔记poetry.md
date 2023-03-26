---
title: poetry |优雅的 Python 包管理
date: 2023-03-19
author: Kevin 吴嘉文
category:
- 知识笔记
---

# Poetry

Poetry replaces `setup.py`, `requirements.txt`, `setup.cfg`, `MANIFEST.in` and `Pipfile` with a simple `pyproject.toml` based project format.

官方链接 [github](https://github.com/python-poetry/poetry) [官网](https://python-poetry.org/)

<!--more-->

其他参考： [知乎笔记](https://zhuanlan.zhihu.com/p/448879082)

## 快速上手

### 安装

根据 [官方提示](https://python-poetry.org/docs/#installing-with-the-official-installer) 进行安装

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

清理缓存：

```bash
poetry cache clear pypi --all
```

### 基本用法

初始化项目

```sh
poetry init
```

poetry 基本项目架构如下：

```
poetry-demo
├── pyproject.toml
├── README.md
├── poetry_demo
│   └── __init__.py
└── tests
    └── __init__.py
```

其中  `pyproject.toml` 在 `poetry init` 是会被创建，用于记录项目的依赖，如。

```toml
[tool.poetry]
name = "myproject"
version = "0.1.0"
description = ""
authors = ["kevinng77 <417333277@qq.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.8"


[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[[tool.poetry.source]]
name = "tsinghua-pypi"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
```

::: tip

 **为了方便包下载，我们需要修改下 pip 源。** 

:::

可以类似 `pip install -r requirments.txt` 批量安装依赖：

```
poetry install
```

添加依赖：

```
 poetry add pendulum
```

执行后会生成 `poetry.lock` ， 他同样可以用来进行环境版本控制，但通常在开发时，我们会更倾向于使用  `pyproject.toml` 。

同步依赖（把当前的 python 包环境与 `poetry.lock`  中的同步）：

```bash
poetry install --sync
```

当你安装了新的 python 包时，可以手动更新  `poetry.lock`  文件：

```bash
poetry update
```

删除不想要的依赖：

```bash
poetry remove pendulum
```

把所有的 python 包锁定版本，By default, this will lock all dependencies to the latest available compatible versions.：

```bash
poetry lock
```

这个功能可以在  [pre-commit hooks](https://python-poetry.org/docs/pre-commit-hooks/#poetry-lock) 当中使用。

最后环境可以导出成 `requirements.txt` :

```bash
poetry export -f requirements.txt --output requirements.txt
```

### 依赖控制

可以在 `[tool.poetry.dependencies]` 中和 `requirments.txt` 一样指定包版本：`numpy = "^1.19.2"`

或针对不同的 python 版本，指定不同的包版本：

```toml
[[tool.poetry.dependencies.numpy]]
version = ">=1.19.2,<1.22.0"
python = "~=3.7.0"

[[tool.poetry.dependencies.numpy]]
version = ">=1.19.2,<1.24.0"
python = ">=3.8,<3.11"
```

### 虚拟环境

默认情况下，poetry 会在  `{cache-dir}/virtualenvs` 下面创建虚拟环境。激活虚拟环境可以直接对  `{cache-dir}/virtualenvs` 下面的 `your_env/bin/activate` 进行执行。

推荐在拥有 `pyproject.toml` 文件的目录下使用 `poetry shell` 自动激活对应 python 环境。

运行 python 文件是，官方提到了使用 `poetry run python xxx.py` 执行。不确定这样做与在虚拟环境中直接执行脚本有啥区别。

### 构建包

在构建之前需要满足以下项目架构：

```sh
.
├── README.md
├── kk                # 改文件夹名称
│   ├── __init__.py
│   └── __main__.py   # 可选
├── poetry.lock
└── pyproject.toml

```

同时你的 `toml` 文件要定义一个可运行脚本：

```toml
[tool.poetry]
name = "kk"
version = "0.1.0"
description = ""
authors = ["kevinng77 <417333277@qq.com>"]
readme = "README.md"

[tool.poetry.scripts]
mk = "kk:kevin"
```

其中 scripts 的格式为 `cli_script = "{package_name}:{function_name}"`。比如以上写的 `mk = "kk:kevin"`。在 `build` 好之后，终端运行 `mk` 相当于直接运行了 `kk`  模块下面的 `kevin()` 函数。

::: tip

此处的 package_name 最好和 `[tool.poetry]` 中的 `name` 相同。同时，在 kk 文件夹下面所有的 python import 导入都已跟目录作为路径。如你想要导入 kk 文件夹下 main.py 中的 `kevin()` 函数，就使用 `from kk.main import kevin` 

:::

准备好之后进行构建：

```sh
poetry build
```

构建好之后  install：

```
poetry install
```

之后在终端运行 `mk` 就能看到 `kevin()` 函数的运行结果了。

构建好了之后可以用 

```bash
poetry publish
```

来推送到 pypi

