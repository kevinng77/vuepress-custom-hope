---
title: Airflow Advanced
date: 2023-12-08
author: Kevin 吴嘉文
category:
- 知识笔记
---

# Airflow Advanced

## Airflow 参数

### Variable 

通常用来配置一些全局参数，如服务器地址等（[参考链接](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/variables.html)）。

在  UI 中设置 `Admin -> Variable`，而后在 dag 中可以调用 variable

```python
from airflow.models import Variable
Variable.get("you_var_name", deserialize_json=True, default_var=None)
```

也可以在 jinja template 中使用：

```bash
# Raw value
echo {{ var.value.<variable_name> }}

# Auto-deserialize JSON value
echo {{ var.json.<variable_name> }}
```

### XComs

XComs (“cross-communications”) 用于 Task 和 Operator 之间的数据传递。

1. PythonOperator 中使用

```python
def send_http_request(params, ti, **kwargs):
    xcom_value = ti.xcom_pull(task_ids='extract_file_locations')  # 默认的 key='return_value'
    
    # 可以自定义 push 值
	ti.xcom_push(key="custom_key", value=any_serializable_value)
    
    # return 的值会被自动传到 XComs 中
    return 123
```

2. 在 jinja Template 当中使用：

```python
SELECT * FROM {{ task_instance.xcom_pull(task_ids='foo', key='table_name') }}
```

此处 `task_instance` 为具体 operator 实例对象





### 坑

::: tip

 bash operator 中，尽量使用 `env` 来传参和调用。

:::

```python
# check_train_sample_number 为 python @task
train_folder = check_train_sample_number(user_id=user_id, 
                                         username=username,
                                         data_processing_result=data_processing_result)

train_task = BashOperator(
    task_id="run_train",
    bash_command="echo {{params.train_data_folder}}",
    params={"user_id": user_id,
            "train_data_folder": "{{ti.xcom_pull('train.check_train_sample_number')}}",  # 这边传递后，是不会解析的
            "train_steps": 800},
    queue="train",
    max_active_tis_per_dag=1,
    env={"test":"{{ti.xcom_pull('train.check_train_sample_number')}}"}
)  # env 中传递没问题，这边也可以直接用 train_folder 参数传递。
```





### Params

#### DAG 级别参数 

用于提供每次运行 DAG 时候的临时参数

 **定义和传参：** 可以在 DAG 中设置：也可以通过 REST API 传入参数

```python
dag = DAG(
    'my_concurrent_dag',
    default_args=default_args,
    params={
         "useruuid": Param(
            default="666",
            type="string",
            minLength=5,
            maxLength=20,
        ),
         "username": Param(
            default="kevin",
            type="string",
            minLength=1,
            maxLength=30,
        ),
     }，
    		render_template_as_native_obj=True
)
```

 **使用参数：** 

```python
# python 中使用参数：
def processing(params, *args, **kwargs):
    # Your task logic goes here
    print(f">>> kwargs: {kwargs['dag_run'].conf}")  # 这个是 dag 中传入的参数，在 REST API 那边也可以设置
    print(f">>> args: {params.get('useruuid')}")  # params 当中会含有 dag 级别的参数，以及 task 级别的参数
    return 0
```

jinja template 使用参数：

```python
caption = SimpleHttpOperator(
    task_id="Caption_Images",
    http_conn_id="test_server_4090",
    method="POST",
    endpoint="{{var.value.caption_endpoint}}",
    data=json.dumps({
        "username": "{{var.value.caption_endpoint}}/caption",
        "uuid": "{{params.useruuid}}",
    }),
    headers={"Content-Type": "application/json"},
    response_check=lambda response: response.json()["uuid"] == "{{params.useruuid}}",
    dag=dag
)
```

#### Task 级别 Params

Task 级别 Params 调用方式与 dags 大致相同。定义 task params 在 operator 相对应地方添加：

```python
PythonOperator(
    task_id="print_my_int_param",
    params={"my_int_param": 10},
    python_callable=print_my_int_param,
)
```

对应的 Operator 需要定义 `provide_context=True` 。

Python 外的 Operator 可以通过 [Jinja Templating](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/operators.html#concepts-jinja-templating) 来调用 dag 当中定义的 params

## Operator

[如何提高 DAG 运行效率](https://airflow.apache.org/docs/apache-airflow/stable/faq.html#how-to-improve-dag-performance)

对于 Operator 可以配置以下参数：

- `max_active_tis_per_dag`：the number of concurrent running task instances across dag_runs per task.

- `pool`: See [Pools](https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/pools.html#concepts-pool).
- `priority_weight`: See [Priority Weights](https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/priority-weight.html#concepts-priority-weight).
- `queue`: 见下文 queue

1. HTTP Operator: 配置 HTTP 链接 [参考 API - SimpleHttpOperator](https://airflow.apache.org/docs/apache-airflow-providers-http/stable/_api/airflow/providers/http/operators/http/index.html#airflow.providers.http.operators.http.SimpleHttpOperator)

::: tip

Airflow HTTP Operator 自带的解码器可能存在无法处理中文的情况，可以考虑用 python Operator 替代：

```python
def send_http_request(params, ti, **kwargs):
    xcom_value = ti.xcom_pull(task_ids='extract_file_locations', key='return_value')
    url = "http://192.168.136.245:8000/caption"
    headers = {"Content-Type": "application/json; charset=utf-8"}
    data = {
        "username": params.get('username'),
        "uuid": params.get('user_id'),
        "image_folder_path": xcom_value,
    }
    
    print(f"Sending Body: {data}")
    response = requests.post(url, data=json.dumps(data, ensure_ascii=False).encode('utf-8'), headers=headers)
        # Check if response is valid and decode it
    if response.status_code == 200:
        response_data = response.json()
        if response_data.get('uuid') != params.get("user_id"):
            raise ValueError(f"Task failed, UUID is not 111. Received UUID: {response_data.get('uuid')}")
        
        # Constructing the desired response format
        output_response = {
            **response_data,
            # add other information here
        }
        return output_response
    else:
        raise Exception(f"HTTP Request failed with status code {response.status_code}")
```

:::



## Airflow 2 新特性

使用 decorator 是的 airflow 更加的简洁明了：  

### dag decorator

```python
from airflow.decorators import dag, task
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2023, 1, 1),
}

@dag(default_args=default_args, 
     description='An example DAG with concurrency settings',
    schedule=None,
    catchup=False,
    max_active_runs=10, # Maximum number of active DAG runs
    max_active_tasks=10,  #  the number of task instances allowed to run concurrently,
    )
def main_flow(user_id: str = "fe03bda5-d2ed-49c7-848f-eb21f84d15dc",   # 定义 dag 参数
              username: str = "kk"):  
    # your dag code here
    
main_flow()  # 加载 dag
```

在这种 dag 情况下，所有 operator 和 flow 都需要写在 `main_flow` 下。

### 参数

jinja template 中使用参数方法不变：

```python
sql_query = """
                SELECT upload_file FROM images
                WHERE user_id = '{{ params.user_id }}';
                """
```

task 可以直接调用 dag 的参数，

```python
@task()
def caption(filename, user_id: str,  username: str):
        print(filename)
        print(username, user_id)
```

::: tip

这边设置的默认值 `user_id: str = user_id` 是不会生效的，必须再调用 task 时候传入参数。再 operator 中，dag 的参数可以直接通过变量调用:

```python
result = caption(extract_file_locations_task.output,
                 user_id = user_id,
                 username = username)
```

:::

### 动态 flow

动态 flow 可以用来实现类似 map reduce 的功能：

比如我们需要对 20 个不同的 `filename` 经过 `caption` 处理后，才会进入下一阶段，那么我们可以使用 mapping:

```python
@task()
def caption(filename, user_id: str,  username: str):
        print(filename)
        print(username, user_id)
        
        
result = caption.partial(user_id=user_id, username=username).expand(filename=extract_file_locations_task.output) 
```

可以使用 task_group:

```python

```







## 环境配置

[airflow.cfg 官方配置](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html)

### Docker 镜像配置

参考文档 https://airflow.apache.org/docs/docker-stack/index.html

[Running Airflow in Docker - 官方提供的 Celery Executor compose file](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) 

### 数据库配置

1. 用户信息数据库
2. 用户图片数据库

### Celery Executor

Airflow  worker 容器中启动 CeleryExecutor 时，需要在系统环境变量中设置好 celery worker 对应的 concurrency 数量，如`AIRFLOW__CELERY__WORKER_CONCURRENCY: "2"`。

::: tip

airflow operator 中设置的 concurrency 和 celery 中的 concurrency 不同。airflow 的 conccurency 似乎是通过 airflow 的 schedular 来实现的。参考 airflow 的 dag 执行流程，schedular 会分发任务到 queue 上。因此如果我们设置了：

`AIRFLOW__CELERY__WORKER_CONCURRENCY: "2"`, 但是 airflow 中 `task_concurrency=10` 时。可能会出现有 8 个任务在 queue 中等待的情况。

:::

![](https://airflow.apache.org/docs/apache-airflow/stable/_images/task_lifecycle_diagram.png)

#### 关于 Queue

 **不同任务（Operator）可以使用不同的 queue：** 当使用 CeleryExecutor 时，可以指定任务发送到的 Celery 队列。

 **Worker 可以监听一个或多个任务队列。**  监听方法与 Celery 中的一样：

```bash
airflow celery worker -q spark,quark
```

而后在 operator 中，设置 `queue=spark` 参数后，可以把任务单独交给指定的这个 worker 去完成。

::: tip

这在需要特殊工作器时非常有用，无论是从资源角度（例如，对于非常轻量级的任务，一个工作器可以处理成千上万个任务而不成问题），还是从环境角度（您希望工作器在 Spark 集群内部运行，因为它需要非常特定的环境和安全权限）。

:::

#### 关于 Celery Worker 对应的 app

参考 [Running Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)，celery worker 启动的时候，添加了这样的参数进行活动监听：

```dockerfile
healthcheck:
      # yamllint disable rule:line-length
      test:
        - "CMD-SHELL"
        - 'celery --app airflow.providers.celery.executors.celery_executor.app inspect ping -d "celery@$${HOSTNAME}" || celery --app airflow.executors.celery_executor.app inspect ping -d "celery@$${HOSTNAME}"'
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s
```







## 问题

 UserWarning: Using the in-memory storage for tracking rate limits as no storage was explicitly specified. This is not recommended for production use. See: https://flask-limiter.readthedocs.io#configuring-a-storage-backend for documentation about configuring the storage backend.

### DAG python API 参考

[DAG Python API 参考](https://airflow.apache.org/docs/apache-airflow/2.3.4/python-api-ref.html)