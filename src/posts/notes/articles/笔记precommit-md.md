---
title: 代码自动化规范-pre-commit 
date: 2021-09-16
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- Linux
mathjax: true
toc: true
comments: 笔记
---

> 本文对代码自动化规范工具 -  pre-commit 进行了主要流程介绍，大部分内容为 [官网详细文档](https://pre-commit.com/)  的中文翻译，总结与拓展。
>
> 相关链接：[hook 指南](https://pre-commit.com/hooks.html)

<!--more-->

### 背景 - git hook

网上对于 git hook 的讲解很多，简单概括的话，git hook 就是一种可以在 git 各个操作阶段被执行的工具，这些工具可以做到如检查代码规范，阻止不合格代码提交等功能。

![相关图片](/assets/img/precommit-md/image-20211003102602000.png =x300)

（图：一些 git hook 示意图。橙色/红色/蓝色 矩形框即为 git hook）

### pre-commit 快速开始

> 一般我们手动建立的 git hook 不会被上传到 git 远程仓库中，若要共同分享 pre-commit git hook，则需要安装并配置 pre-commit。

#### 安装

```
pip install pre-commit
```

或 conda 安装

```
conda install -c conda-forge pre-commit
```

安装后在`requirements.txt`中添加 `pre-commit` 一项

#### 添加一个样本配置文件

输入 `pre-commit --version` 查看是否安装成功。

添加一个 pre-commit 配置文件，在 `git` 仓库根目录下创建并编写 `.pre-commit-config.yaml` 文件。

在 python 环境下输入 `pre-commit sample-config` ，我们能够得到一个配置文件样例：

```
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v2.3.0
    hooks:
    -   id: check-yaml
    -   id: end-of-file-fixer
    -   id: trailing-whitespace
-   repo: https://github.com/psf/black
    rev: 19.3b0
    hooks:
    -   id: black
```

将以上内容复制到刚刚创建的 `.pre-commit-config.yaml`中。更多的配置设定请参考[下文](#配置-pre-commit)。

#### 安装 pre-commit 相关 git hook 脚本

pre-commit 也是 git hook 中的一种，在对应的 git 仓库下执行：

```
pre-commit install
```

之后在`.git/hook` 文件加中会生成一个可执行脚本，此时 `pre-commit` 就会自动在 `git commit` 的时候运行了。

#### 对所有文件执行 pre-commit

通常 pre-commit 只对修改后的文件执行 pre-commit 操作。

### 配置 pre-commit

 **repos 参数配置** 

在每个 repo 下我们需要配置以下三种映射：

| [`repo`](https://pre-commit.com/#repos-repo)   | `git clone` 对象的仓库 URL（需要拷贝的 hook 的对应仓库链接）  |
| ---------------------------------------------- | ------------------------------------------------------------ |
| [`rev`](https://pre-commit.com/#repos-rev)     | 克隆对象仓库的版本或 tag: 部分之前的版本会使用 `sha`          |
| [`hooks`](https://pre-commit.com/#repos-hooks) | 包含各种 [hook mappings](https://pre-commit.com/#pre-commit-configyaml---hooks) 的列表 |

实例：

```
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v1.2.3
    hooks:
    -   ...
```

 **hooks 的重要配置** 

一下列举部分常用的配置信息，其他参数请参考  [hook mappings 官方](https://pre-commit.com/#pre-commit-configyaml---hooks) 。[这份指南](https://pre-commit.com/hooks.html) 整理了各种代码对应的 git hood id 与功能，方便查看与配置。

| [`id`](https://pre-commit.com/#config-id)                    | hook 在对应仓库中的 id                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`name`](https://pre-commit.com/#config-name)                | (optional) override the name of the hook - shown during hook execution. |
| [`files`](https://pre-commit.com/#config-files)              | (optional) 重新设定 hook 要执行的文件对象。                    |
| [`language_version`](https://pre-commit.com/#config-language_version) | (optional) 重新设定语言版本，具体参考： [Overriding Language Version](https://pre-commit.com/#overriding-language-version). |
| [`verbose`](https://pre-commit.com/#config-verbose)          | (optional) 如果设定为 true，hook 检查代码通过后也会输出执行日志。 |

一些其他可选参数：

[`default_language_version`](https://pre-commit.com/#top_level-default_language_version) 用于配置默认的语言版本，如 配置 python 的默认版本为 3.7，则添加：

```
default_language_version:
    python: python3.7
```

[`default_stages`](https://pre-commit.com/#top_level-default_stages) 默认为所有 stages。暂时没搞懂这个参数干啥的，似乎挺少人配置它。样例：

```
default_stages: [commit, push]
```

[`fail_fast`](https://pre-commit.com/#top_level-fail_fast) 默认为 false，如果设置为 true 的话， pre-commit 会在第一次 hood 失败后马上终止。

```
fail_fast: true
```

[`minimum_pre_commit_version`](https://pre-commit.com/#top_level-minimum_pre_commit_version) 默认为 `'0'` ，最小 pre-commit 版本要求。

### 参考配置：

```yaml
repos:
-   repo: https://github.com/Lucas-C/pre-commit-hooks
    rev: v1.0.1
    hooks:
    -   id: forbid-crlf
        files: \.md$
    -   id: remove-crlf
        files: \.md$
    -   id: forbid-tabs
        files: \.md$
    -   id: remove-tabs
        files: \.md$
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.0.1
    hooks:
    -   id: check-merge-conflict
    -   id: detect-private-key
    -   id: check-symlinks
    -   id: trailing-whitespace
        files: \.md$
    -   id: end-of-file-fixer
    -   id: check-docstring-first
    -   id: check-json
    -   id: check-added-large-files
    -   id: check-yaml
    -   id: debug-statements
    -   id: name-tests-test
    -   id: double-quote-string-fixer
    -   id: requirements-txt-fixer
    -   id: fix-encoding-pragma
    

```





