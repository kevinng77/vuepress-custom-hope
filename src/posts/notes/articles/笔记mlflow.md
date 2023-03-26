---
title: MLFlow
date: 2023-01-30
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- DevOps
- MLOps
mathjax: true
toc: true
---

## 概述

[MLflow 官方](https://www.mlflow.org/docs/latest/quickstart.html)

### 快速开始

```shell
pip install mlflow
```

MLflow 包含了四个应用：

+ MLflow Tracking：提供可视化模型训练记录，包括超参配置、训练指标、模型权重、模型输出文件等 artifact。数据会被记录在 MLflow 云服务器上
+ MLflow Projects：打包管理 ML 模型项目，方便复现与部署。
+ MLflow Models：主要用于 AI 模型储存、加载以及服务部署。由于目前 AI 框架较多，因此 MLflow 也提供了比较统一的保存加载和服务部署方案。
+ MLflow Registry：如果说 MLflow Model 能让我们将开发过程中的模型权重、配置等储存起来。那么 MLflow Registry 则让我们对筛选出来的、要落地的模型进行标注和管理。



## MLflow Tracking

提供可视化模型训练记录，包括超参配置、训练指标、模型权重、模型输出文件等 artifact。数据会被记录在 MLflow 云服务器上。[MLTracking Document](https://www.mlflow.org/docs/latest/tracking.html#)

### 启动 MLflow 服务器

在服务器上启动 Tracking 服务

```shell
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root <your-artifact-address> \
    --host 0.0.0.0 \
    --port 5000
```

如上，`tracking_uri=server_ip:5000`。所有文件将被储存在运行 mlflow 的服务器上。自定义存储配置查看[官网指南](https://www.mlflow.org/docs/latest/tracking.html#id13)。如配置 Azure Blob Storage 时，需要在服务端和客户端 **同时配置** 链接用的账户与密码到环境变量中。

`backend-store-uri` 用以储存结构化数据如 metrics, params, 等模型信息。`artifact-root` 用以储存非结构化信息包括：模型权重、模型配置文件、模型输出文字、图片、网页等各类数据，一般配置云储存服务。

需要注意如果 `default-artifact-root` 提供 nfs 路径，那么路径对应的文件夹需要在客户端与服务端上保持一致。

### 客户端配置 URI

参考 [MLflow 文档](https://www.mlflow.org/docs/latest/tracking.html) 不同的 URI 支持不同的功能与储存效果。`mlflow.get_tracking_uri()` 查看当前 URI。[`mlflow.set_tracking_uri()`](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri)  配置新的 URI。

### 建立 Experiment

实验类似于文件夹，能够用来快速分类与检索模型训练结果。

```python
experiment_name = "kevin_task"
mlflow.set_experiment(experiment_name)
```

默认情况下，MLFLOW 使用 `default` 实验。

### 记录训练数据

可视化 AI 模型训练指标，类似于 TensorBoard、VisualDL 等工具。可以记录图片、文字、数字、html 页面等任意格式的内容。

```python
import mlflow
import cv2

mlflow.set_tracking_uri("http://mlflow_server_ip:port")

experiment_name = "kevin_task"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)
np_img = cv2.imread("./my_img.png")

with mlflow.start_run() as run:
    mlflow.set_tag("your Keys", "your tags")
    mlflow.log_param("lambda", 0.122)
    mlflow.log_metric("f1", .6)
    mlflow.log_text("this is a output from NLP model","test.txt")
    mlflow.log_text('this is a output from <div style="font-size:40px">NLP</div> model',"custom.html")
    mlflow.log_image(np_img,"what.png")  # imput np array
    
    # 可以记录不同 step 的数值。
    for i in range(4):
        mlflow.log_metric("loss", i/30, step=i)
```

比较特别的是 MLflow 适配了自动记录训练参数，但只适配部分主流的 AI 框架，如 SKlearn，Torch 等。在训练开始前声明 `mlflow.autolog()` 即可。mlflow 会自动根据模型类型选择记录的参数和指标。

如果 MLflow 搭载在云服务器上，那么所有记录的数据会被上传到已经配置的云数据库中，因此网速慢或者 artifact 文件太大的话，都会导致代码运行很久。

## MLflow Projects

打包管理 ML 模型项目，方便复现与部署。

### 项目文件架构

一个 MLflow 项目的文件夹包含以下文件：

```shell
|- MLproject
|- python_env.yaml
|- your_al_files
```

#### MLproject 文件

官方提供的模板：

```yaml
name: My Project

python_env: python_env.yaml
# or
# conda_env: my_env.yaml
# or
# docker_env:
#    image:  mlflow-docker-example

entry_points:
  main:
    parameters:
      data_file: path
      regularization: {type: float, default: 0.1}
    command: "python train.py -r {regularization} {data_file}"
  validate:
    parameters:
      data_file: path
    command: "python validate.py {data_file}"
```

其中我们需要定义：

+ `name`: 项目名字
+ `xxx_env`：项目运行的环境
+ `entry_points`：如何运行这个项目，一般是一些命令。

#### 环境项目配置

普通情况下我们只需要使用 Virtualenv 虚拟环境运行项目，因此可以编写以下 `python_env.yaml` 文件：

```yaml
# Python version required to run the project.
python: "3.8.15"
# Dependencies required to build packages. This field is optional.
build_dependencies:
  - pip
  - setuptools
  - wheel==0.37.1
# Dependencies required to run the project.
dependencies:
  - -r requirments.txt
```

然后在 MLproject 中配置改文件的 **相对路径** 

```yaml
python_env: python_env.yaml
```

注意：使用 `python_env` 时，需要安装`pyenv` 相关命令。

当然也可以使用 docker（默认从 DockerHub 拉取

）：

```yaml
docker_env:
  image: mlflow-docker-example-environment
  volumes: ["/local/path:/container/mount/path"]
  environment: [["NEW_ENV_VAR", "new_var_value"], "VAR_TO_COPY_FROM_HOST_ENVIRONMENT"]
```

#### 配置项目运行方式

配置 Entry Point

```yaml
entry_points:
  main:
    parameters:
      data_file: path
      regularization: {type: float, default: 0.1}
    command: "python train.py -r {regularization} {data_file}"
  validate:
    parameters:
      data_file: path
    command: "python validate.py {data_file}"
```

参数格式：

```yaml
parameter_name: {type: data_type, default: value}  # Short syntax

parameter_name:     # Long syntax
  type: data_type
  default: value
```

其中 `type` 支持 `string`, `float`, `path`, `uri`。

### 运行 MLProject 项目

MLproject 能够让你知道这个项目的环境依赖还有运行方式，他仅会在 MLtracking 上多记录一行 `Run Command=mlflow run ...`。

官方推荐将所有项目的配置都记录在 MLproject 文件下，然后直接用一行简单的命令就能够顺利启动项目。

#### 使用 python 文件运行

 `mlflow.projects.run()`

```python
import mlflow
import os

# 要在运行前配置好 tracking uri，当然也可以在环境变量中定义好
mlflow.set_tracking_uri("http://40.76.242.139:8005")
os.environ["ddd"

os.environ["AZURE_STORAGE_ACCESS_KEY"] = r"xxx"

project_uri = "./"
params = {"alpha": 0.9, "l1_ratio": 0.0666}

# Run MLflow project and create a reproducible conda environment
# on a local host
mlflow.run(project_uri, experiment_name="kevin from python",parameters=params)
```

注意：如果使用 MLProject 启动训练或者推理，那么在你的训练文件中就不应该出现 `with mlflow.start_run()` 等新建 `run`的 代码。

 **比较特别的是** ， [`mlflow.projects.run()`](https://www.mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run) 可以和  [`mlflow.client`](https://www.mlflow.org/docs/latest/python_api/mlflow.client.html#module-mlflow.client) 结合，来实现 pipeline。 [`mlflow.projects.run()`](https://www.mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run) 每次运行后，使用 `mlflow.client` 接受本次 run 的结果。而后根据结果来判断下一个 run 什么。

#### 使用终端运行

使用 `mlflow run` 时，需要提前在环境变量里定义好 tracking_uri 等其他信息，比如你的 artifact 网盘账号和密码。

```python
export MLFLOW_TRACKING_URI="http://40.76.242.139:8005"
mlflow run . --experiment-name="new kevin 2" -P alpha=5.0
```

## MLflow Models

主要用于 AI 模型储存、加载以及服务部署。由于目前 AI 框架较多，因此 MLflow 也提供了比较统一的保存加载和服务部署方案。

### 记录模型

使用 `mlflow.xxx.log_model()` 可以将模型记录在 mlflow 服务器中，如对于 sklearn 的模型，可以使用 [mlflow.sklearn.log_model](https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model) 

```python
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

iris = datasets.load_iris()
iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(iris_train, iris.target)
signature = infer_signature(iris_train, clf.predict(iris_train))
mlflow.sklearn.log_model(clf, "iris_rf", signature=signature)
```

运行以上代码，在 mlflow 会储存相对应的模型文件：

```python
# Directory written by mlflow.sklearn.save_model(model, "my_model")
my_model/
├── MLmodel
├── model.pkl
├── conda.yaml
├── input_example.json
├── python_env.yaml
└── requirements.txt
```

其中 MLmodel 文件长这样

```yaml
artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.8.16
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 0.23.2
mlflow_version: 2.1.1
model_uuid: 3a3c5f7183734848a7d80ad08baaab0c
run_id: 47b2603aa0444991939200ac846d87cf
utc_time_created: '2023-02-02 14:32:54.030002'
```

### 记录模型输入输出

等级的 MLmodel 中可以同时记录模型的输入、输出等信息。

可以自动根据数据推导输入输出格式：

```python
from mlflow.models.signature import infer_signature

signature = infer_signature(testX, model.predict(testX))
```

可以手动记录表格输入：

```python
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_schema = Schema([
  ColSpec("double", "sepal length (cm)"),
  ColSpec("double", "sepal width (cm)"),
  ColSpec("double", "petal length (cm)"),
  ColSpec("double", "petal width (cm)"),
])
output_schema = Schema([ColSpec("long")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
```

或者 tensor 输入：

```python
from mlflow.types.schema import Schema, TensorSpec

input_schema = Schema([
  TensorSpec(np.dtype(np.uint8), (-1, 28, 28, 1)),
])
output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
```

同时可以直接记录 input_example:

```python
# each input has shape (4, 4)
input_example = np.array([
   [[  0,   0,   0,   0],
    [  0, 134,  25,  56],
    [253, 242, 195,   6],
    [  0,  93,  82,  82]],
   [[  0,  23,  46,   0],
    [ 33,  13,  36, 166],
    [ 76,  75,   0, 255],
    [ 33,  44,  11,  82]]
], dtype=np.uint8)
mlflow.tensorflow.log_model(..., input_example=input_example)
```

### 模型保存与加载

很多主流的 AI 框架都有自己的保存、加载模型方法。mlflow 仅是在这些主流 AI 的 API 上套一层，已实现用统一 MLflow API 保存和加载 AI 模型。

#### 上传模型

官方为主流框架提供了 `mlflow.sklearn`， `mlflow.torch` 等类，可以使用 `save_model` 保存到本地, `log_model` 保存到云端 artifact 服务器等 API，比如对于 onnx 模型：

```python
# convert model to ONNX and load it
torch.onnx.export(net, X, "model.onnx")
onnx_model = onnx.load_model("model.onnx")

# log the model into a mlflow run
with mlflow.start_run():
    model_info = mlflow.onnx.log_model(onnx_model, "model")

# load the logged model and make a prediction
onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = onnx_pyfunc.predict(X.numpy())
print(predictions)
```

或者对于 paddle 模型：

```python
import mlflow.paddle
import paddle

class Regressor(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

        self.fc = Linear(in_features=10, out_features=1)

    @paddle.jit.to_static
    def forward(self, inputs):
        x = self.fc(inputs)
        return x
    
# ... your training code here

mlflow.paddle.log_model(model, "model")

```

如以上 `mlflow.paddle.log_model` 中，包含了 `paddle.save()` 等方法。

此外可以使用  [`mlflow.models.Model`](https://www.mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.Model) 来自定义 MLflow 模型的创建和储存。或者参考 [custom python models](https://www.mlflow.org/docs/latest/models.html#custom-python-models)

#### 模型推理

在 mlflow server 对应的 artifact model 中，可以查看到模型的 uri。

```python
model_uri = 'runs:/a60a1cdd9c644e6bb6594e7ff911e0ea/model'

# 在已知 run id 情况下，可以通过 model_uri = mlflow.get_artifact_uri("model")


pd_model = mlflow.paddle.load_model(model_uri)
training_data, test_data = load_data()
np_test_data = np.array(test_data).astype("float32")
print(pd_model(np_test_data[:, :-1]))
```

`mlflow.paddle.load_model` 中包含了 飞桨的官方加载模型方法：`paddle.load()` 

### 模型部署

[官方文档链接](https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools)

 **本地部署** 

可以直接通过

````shell
model models serve -m model_uri -p 8000
````

在本地部署 rest_api，需要使用 [`pyenv`](https://github.com/pyenv/pyenv)，笔者部署过程中还是遇到很多环境依赖问题，如本地设备缺少 bz2 依赖等问题。

 **创建 docker 镜像并部署** 

该方案部署相对顺利，但是需要科学上网环境才能成功。

```shell
mlflow models build-docker -m "modeluri" -n "image-name"

docker run -p 5001:8080 "my-image-name"
```

 **部署到云服务** 

MLflow 可以部署到 Azure ML Studio，Apache Spark UDF 等。具体参考 [官方文档链接](https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools)

 **模型调用** 

使用 `curl -X POST -H "Content-Type:application/json" --data your_input`，其中，输入 `your_input` 的格式可以在上文提到的 [记录模型输入输出](#记录模型输入输出) 查看。 

```shell
curl -X POST -H "Content-Type:application/json" --data '{
  "inputs": [
    [
      1.7080316543579102,
      2.5316741466522217,
      1.6952152252197266,
      1.9768019914627075,
      1.7444103956222534,
      1.7886496782302856,
      1.963786244392395,
      1.849048137664795,
      1.9302328824996948,
      1.7990881204605103
    ]
  ]
}' http://127.0.0.1:8000/invocations
```

返回结果

```shell
{"predictions": [{"0": 566.980224609375}]}%
```

## MLflow Registry

如果说 MLflow Model 能让我们将开发过程中的模型权重、配置等储存起来。那么 MLflow Registry 则让我们对筛选出来的、要落地的模型进行标注和管理。[官网文档](https://www.mlflow.org/docs/latest/model-registry.html)

### 注册模型

可以在 MLflow 的 web UI 上直接注册：

![相关图片](https://www.mlflow.org/docs/latest/_images/oss_registry_1_register.png )

也可以在代码中注册：

```python
# log 模型时提供 registerd_model_name 即可
mlflow.sklearn.log_model(
    sk_model=sk_learn_rfr,
    artifact_path="sklearn-model",
    registered_model_name="sk-learn-random-forest-reg-model"
) 

# 或者基于现有的模型 MODEL URI 注册
result = mlflow.register_model(
    "runs:/d16076a3ec534311817565e6527539c0/sklearn-model",
    "sk-learn-random-forest-reg"
)
```

对于注册的模型，可以设置 `Description`， `Tags` ， `Stage` 等信息。

### 使用注册的模型

在上一节中，我们使用 MODEL_URI (Run_id) 来索引模型，对于已经注册了的模型，可以使用以下字符代替：

```python
model_uri=f"models:/{model_name}/{model_version}"

model_uri = f"models:/{model_name}/{stage}"
```

## 其他功能

有一部分 Data Bricks MLflow 的功能在开源版中没有.

