---
title: ElasticSearch 使用笔记
date: 2022-10-25
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- 数据库
mathjax: true
toc: true
comments: 笔记
---

用于快速了解 ES 的资料： [狂神说 ElasticSearch 笔记](https://www.kuangstudy.com/bbs/1354069127022583809) 或 [【尚硅谷】ElasticSearch 教程入门到精通](https://www.bilibili.com/video/BV1hh411D7sb?p=5&spm_id_from=pageDriver&vd_source=4418d5cd5be787be7e3ff4138eeb9b0a)

上手与入门推荐： [推荐 中文 Elastic 文档](https://www.elastic.co/guide/cn/elasticsearch/guide/current/changing-similarities.html) 熟悉了 ES 的相关操作和环境。查看 [python elastic search 文档](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/overview.html)，[推荐 ES 语法文档](https://www.elastic.co/guide/cn/elasticsearch/guide/current/query-dsl-intro.html)

## 概述

Elasticsearch 是一个 **实时分布式搜索和分析引擎** 。比如博客全站搜索等功能可使用 ES 实现。

Elasticsearch 是一个基于 Apache Lucene(TM)的开源搜索引擎。无论在开源还是专有领域, Lucene 可被认为是迄今为止最先进、性能最好的、功能最全的搜索引擎库。

但是,  **Lucene 只是一个库** 。 想要使用它,你必须使用 Java 来作为开发语言并将其直接集成到你的应用中,更糟糕的是, Lucene 非常复杂,你需要深入了解检索的相关知识来理解它是如何工作的。

### ELK

ELK 是 **Elasticsearch、Logstash、 Kibana 三大开源框架首字母大写简称** 。市面上也被成为 Elastic Stack。

1. 收集清洗数据(Logstash) ==> 搜索、存储(ElasticSearch) ==> 展示(Kibana)

ES 大致数据结构：

| Relational DB      | ElasticSearch          |
| ------------------ | ---------------------- |
| 数据库（database） | 索引（indices）        |
| 表（tables）       | types \<慢慢会被弃用!> |
| 行（rows）         | documents              |
| 字段（columns）    | fields                 |

 **elasticsearch（集群）** 中可以包含多个 **索引（数据库）**  ,每个索引 `index` 中可以包含多个 **类型（表）**  ,每个类型下又包含多个 **文档（行）**  ,每个文档中又包含多个 **字段（列）** 。

## 安装

### docker 示例

该方法较少用，但是能够快速，安全地在本地上创建一个可以用于学习地 ES 环境。参考官方指南 [link](https://www.elastic.co/guide/en/elasticsearch/reference/7.5/docker.html)。以单节点为例，根据 dockerhub ElasticSearch 的指南拉取对应镜像并开启。

```bash
https://www.elastic.co/guide/en/elasticsearch/reference/7.5/docker.html
```

不同版本的 ES 操作差别较大，以下以 7.8.0 版本进行实例，安装时需要 ES 和 kibana 版本相同。

```bash
docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:7.8.0 
docker run -d --name kibana --link elasticsearch:elasticsearch -p 5601:5601 kibana:7.8.0
```

启动之后可以通过 `curl http://localhost:9200/` 检查 ES 是否安装成功，通过浏览器  `http://localhost:5601/`检查 kibana 安装情况。若失败，则可以 `docker logs elasticsearch` 查看运行日志，而后结合官网文档查询解决方案。

## ES 操作 

ES 主要通过 Java API 或者使用 RESTful API 通过 9200 端口进行交互。官方也提供了其他语言的客户端，如 [python API](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/overview.html)。

ES 中使用 JSON 作为文档序列化格式。数据库整体的结构为：`索引 -> 类型 -> 文档 -> 属性`。如，对于地址 `localhost:9200/megacorp/employee/1?pretty` 其中 `megacorp` 为索引， `employee` 为类型。在高版本中，类型已经被弃用。

### 检索理论

#### ES 的搜索功能

ES 可以做到在某些字段上使用结构化查询，某些字段使用排序（与 SQL 结构化查询一样）。同时，应用全文检索，找出与关键字匹配的文档。

ES 的搜索存在三个关键概念：

+ 映射（Mapping）- 描述数据在每个字段内如何存储

+ 分析（Analysis）- 全文是如何处理使之可以被搜索的
+ 领域特定查询语言（Query DSL）- Elasticsearch 中强大灵活的查询语言

##### ES 中的倒排索引

[倒排索引文档](https://www.elastic.co/guide/cn/elasticsearch/guide/current/inverted-index.html) 简单的说就是给每个单词，简历一个单词到文档的映射。这样一来，搜索关键词时候就不需要遍历所有文档了。

##### 分析器

倒排索引的一个关键点在于句词标准化，及分词和标准化。如中文分词，英文的词根提取等等。这个标准化也就是我们提到的 ES 关键概念之一：分析 `Analysis`。如对于英文的分析器，其中包含了：

- 字符过滤器（& 转 and，It's 转 it is）
- 分词器（根据空格和标点进行分词）
- Token 过滤器。（过滤停用词、词干提取等）

分析器与常规的 NLP 预处理方法大同小异。ES 中提供能标准分析器`standard`，简单分析器、空格分析器等等。

当然只有在进行全文检索的时候，才用到分析器。更多关于分析器的介绍，参考 [官方指南](https://www.elastic.co/guide/cn/elasticsearch/guide/current/analysis-intro.html)。

当然，ES 中还支持各种语言处理方式，包括拼写错误、同义词转换等 [官方链接](https://www.elastic.co/guide/cn/elasticsearch/guide/current/using-language-analyzers.html)。

自定以分析器大致模板：

```json
{
    "settings": {
        "analysis": {
            "char_filter": {
                "&_to_and": {
                    "type":       "mapping",
                    "mappings": [ "&=> and "]
            }},  // 把 & 替换成 and
            "filter": {
                "my_stopwords": {
                    "type":       "stop",
                    "stopwords": [ "the", "a" ]  // 过滤停用词
            }},
            "analyzer": {
                "my_analyzer": {
                    "type":         "custom",
                    "char_filter":  [ "html_strip", "&_to_and" ],
                    "tokenizer":    "standard",
                    "filter":       [ "lowercase", "my_stopwords" ]
            }} // 自定义分析器
}}}
```



##### 映射

映射 `mapping` 中储存了每个文档中对应字段的类型。如：

```json
"mappings": {
    "properties": {
        "ziduan": {
            "type": "text",
            "analyzer": "standard",
            "similarity": "BM25"
        },
    }
}
```

其中，数据的类型有 `string`, `long`, `float`, `double`, `boolean`, `date`, `object`。`object` 的存在让 ES 支持内部对象的映射（多层结构的 JSON）。

除了 `type`， `mapping` 中还包含了其他元数据，如：

- `index`：指示该字段的分析方式。（全文检索或精确检索?）
- `analyzer`：进行全文检索时候的分析器，如 `english` 等。 当然可以 [自定义分析器](https://www.elastic.co/guide/cn/elasticsearch/guide/current/custom-analyzers.html)
- 等

### 操作概览

ES 采用 RESTful 进行请求，使用 get/post  等操作来创建、查询等操作。   

| 功能     | 请求方式 | body                    | 地址(localhost:9200+) |
| -------- | -------- | ----------------------- | --------------------- |
| 创建索引 | PUT      |                         | /yourdata             |
| 创建数据 | POST     | `{"doc":{"title":xxx}}` | /shopping/_doc        |
| 修改数据 | POST     | {}                      | /index/id/_update/    |
| 删除     | DELETE   |                         | /shopping/_doc/1001   |

此外也可以通过 `PUT /megacorp/employee/2` 来同时创建索引，并保存数据，如：

```bash
curl -X PUT "localhost:9200/megacorp/employee/1?pretty" -H 'Content-Type: application/json' -d'
{
    "first_name" : "John",
    "last_name" :  "Smith",
    "age" :        25,
    "about" :      "I love to go rock climbing",
    "interests": [ "sports", "music" ]
}
'
```

创建文档时，可以采用 `POST /website/blog/` 让 ES 自动分配唯一 `_id`。同时可以使用 `PUT /website/blog/123/_create` 来创建指定 `_id` 的文档，当 `_id` 存在时创建失败，返回 409，创建成功返回 201。

##### 更新文档

更新文档示例：

```bash
curl -X POST "localhost:9200/website/pageviews/1/_update?pretty" -H 'Content-Type: application/json' -d'
{
   "script" : "ctx._source.views+=1",
   "upsert": {
       "views": 1
   }
}
'
```

##### 分布式文件控制

ES `index` ， `GET` 和 `delete` 请求时，我们指出每个文档都有一个 `_version`，可以通过 `_version` 确保应用中文档不会冲突。如：

```bash
curl -X PUT "localhost:9200/website/blog/1?version=1&pretty" -H 'Content-Type: application/json' -d'
{
  "title": "My first blog entry",
  "text":  "Starting to get the hang of this..."
}
'
```

以上指定了修改文档对应的 `version`。

### 检索数据

更多查询语句方式，查看 [Python Elastic search 官网文档](https://elasticsearch-py.readthedocs.io/en/v8.4.2/api.html?highlight=search), [推荐 ES 语法文档](https://www.elastic.co/guide/cn/elasticsearch/guide/current/query-dsl-intro.html)

数据检索可以分为查询与过滤，通常查询（Query）语句用来进行任何需要影响相关性得分的搜索。而过滤则（Filtering）用来进行文档匹配。查询与过滤的性能差异较大。

查询主要是发送请求到`http://127.0.0.1:9200/shopping/_search`。以 `body` 的不同来实现不同检索方式：

#### 搜索结果

对于批量检索，返回的结果中有以下字段：

- `timed_out` 值告诉我们查询是否超时，可以指定。
- `took` 搜索请求耗费了多少毫秒。
- `max_score` 值是与查询所匹配文档的 `_score` 的最大值。
- `hits` 包含 `total` 字段来表示匹配到的文档总数，并且一个 `hits` 数组包含所查询结果的前十个文档。在 `hits` 数组中每个结果包含文档的 `_index` 、 `_type` 、 `_id` ，加上 `_source` 字段。

示例搜索结果：

```bash
{
   "hits" : {
      "total" :       14,
      "hits" : [
        {
          "_index":   "us",
          "_type":    "tweet",
          "_id":      "7",
          "_score":   1,
          "_source": {
             "date":    "2014-09-17",
             "name":    "John Smith",
             "tweet":   "The Query DSL is really powerful and flexible",
             "user_id": 2
          }
       },
        ... 9 RESULTS REMOVED ...
      ],
      "max_score" :   1
   },
   "took" :           4,
   "_shards" : {
      "failed" :      0,
      "successful" :  10,
      "total" :       10
   },
   "timed_out" :      false
}
```

 **结果分页** 

类似 SQL 使用 `LIMIT` 限制输出数量，ES 使用 `size` 和 `from` 来限制。

检索时，ES 会见不同字段的内容进行组合，拼接成一个 `_all` 字段，而后对 `_all`字段进行检索。

#### 基础检索操作

通过文档的 ID 直接进行检索：

```bash
curl -X GET "localhost:9200/megacorp/employee/1?pretty"
```

将 `GET` 改为 `DELETE`，则变为删除指定文档。

#### 模糊匹配操作

主要使用 `match` 语法，`match` 中的文字会首先经过分词，而后通过倒排索引进行检索，以下示例仅展示请求的 `body` 内容。

```json
{
    "query":{
        "match":{
            "last_name":"smith"
        }
    },
    "from" : 0, // 其实位置
    "size": 2, 
    "_source":["age"],  // 想要得到的数据。
    "sort": {
        "age": "decs"  // 降序排序
    }
}
```

模糊查询的结果中，有 `_score` 的属性。其为相关性评分，在早期 ES 中，使用  TF-IDF 评分，后期采用 BM25。

##### 其他常用查询方式

其他常见的查询方式有：

```json
// 在多个字段上查询
{
    "multi_match":{
        "query": "text searched",
        "fields": ["title", "body"]
    }
}

// 对于数字或者时间查询
{
    "range": {
        "age": {
            "gte":  20, // gt, lt, lte
            "lt":   30
        }
    }
}

// 精确匹配
{ "term": { "tag":    "full_text"  }}

// 多字段匹配，只要包含其中一个，就算满足条件
{ "terms": { "tag": [ "search", "full_text", "nosql" ] }}
```

#### 复合查询操作

比如我们希望查询语句中必须包含某些字符，必须不出现某些字符，则可以使用复合查询。

```json
// must 相当于 and
// should 相当于 or
// must_not 相当于 not (... and ...)
// filter 过滤

{
    "bool": {
        "must": { "match":   { "email": "business opportunity" }},
        "should": [
            { "match":       { "starred": true }},
            { "bool": {
                "must":      { "match": { "folder": "inbox" }},
                "must_not":  { "match": { "spam": true }}
            }}
        ],
        "minimum_should_match": 1，
        "filter": {
          "range": { "date": { "gte": "2014-01-01" }} 
        }
    }
}
```

#### 其他搜索

正则表达式匹配 [link](https://www.elastic.co/guide/cn/elasticsearch/guide/current/_wildcard_and_regexp_queries.html) 等

#### 相关性

这里有一份 [ES2.x TF-IDF 相关性计算方法](https://www.elastic.co/guide/cn/elasticsearch/guide/current/practical-scoring-function.html) 指南。可以看到官方的 TF-IDF 与传统计算方式有些不同。

当然，这里的相关性是支持自定义的，比如根据某个字段的值加上权重 [参考](https://www.elastic.co/guide/cn/elasticsearch/guide/current/boosting-by-popularity.html) ：

```bash
GET /blogposts/post/_search
{
  "query": {
    "function_score": { 
      "query": { 
        "multi_match": {
          "query":    "popularity",
          "fields": [ "title", "content" ]
        }
      },
      "field_value_factor": { 
        "field": "votes" ,
        "modifier": "log1p",
        "factor":   2
      }
    }
  }
}
```

或者对 TF-IDF 进行改造 [链接](https://www.elastic.co/guide/cn/elasticsearch/guide/current/function-score-filters.html)。

TF-IDF 还支持了部分其他流行的相似度，如 BM25，词频饱和度等。可以通过映射来配置 BM25 的超参：

```bash
{
  "settings": {
    "similarity": {
      "my_bm25": { 
        "type": "BM25",
        "b":    0  # 设置 BM25 的超参 b 或 k
      }
    }
  },
  "mappings": {
    "doc": {
      "properties": {
        "title": {
          "type":       "string",
          "similarity": "my_bm25" 
        },  # 在 mapping 中配置即可生效
        "body": {
          "type":       "string",
          "similarity": "BM25" 
        }
      }
    }
  }
}
```

此外我们可以提取搜索结果中的 TOP k 个相关语句，并返回，这需要用到 `minimum_should_match` 和 `rescore`，如：

```bash
curl -X GET "localhost:9200/my_index/my_type/_search?pretty" -H 'Content-Type: application/json' -d'
{
    "query": {
        "match": {  
            "title": {
                "query":                "quick brown fox",
                "minimum_should_match": "30%"
            }
        }
    },
    "rescore": {
        "window_size": 50, 
        "query": {         
            "rescore_query": {
                "match_phrase": {
                    "title": {
                        "query": "quick brown fox",
                        "slop":  50
                    }
                }
            }
        }
    }
}
'
```

具体查看 [官方文档](https://www.elastic.co/guide/cn/elasticsearch/guide/current/_Improving_Performance.html)

#### 书本金句

本章介绍了 Lucene 是如何基于 TF/IDF 生成评分的。理解评分过程是非常重要的，这样就可以根据具体的业务对评分结果进行调试、调节、减弱和定制。（所以书中大篇幅的介绍了如何去修改 TF-IDF 权重，或者自定义自己的相关性权重）

通常，经过对策略字段应用权重提升，或通过对查询语句结构的调整来强调某个句子的重要性这些方法，就足以获得良好的结果。

相关度是一个十分主观的概念，很容易进入反复调整相关度，而没有明显进展的怪圈。一定要在实践中，监控用户的行为，而后通过大众的反应来进行对应的调整。

### 书中其他内容

等用到的时候在学吧，不然肯定忘记了。。。

+  **集群相关操作：** 集群维护、水平扩容、分布式文档储存
+  **其他匹配方式：** 正则，Ngram 等
+  **管理监控和部署：** 集群管理，索引统计等

## Python Elasticsearch 笔记

### 安装

ES 版本 7.8.0，对应安装 `pip install -U 'elasticsearch<7'`。版本不对应时，会出现各种链接问题。以下代码测试版本： `elasticsearch-6.8.2`

### 操作

实例化

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("https://localhost:9200", 
                   api_key=("id", "api_key"))
```

创建索引

```python
body = {
    "settings": {
        "number_of_shards": 1,
        "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
    },
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "analyzer": "standard",
                "similarity": "BM25"
            },
        }
    },
}

result = es.indices.create(index = "test", body = body)

# 删除：
es.indices.delete("test")
```

插入删除数据

```python
# 插入数据
es.index(index = "test", doc_type = "_doc", id = 1, body = {"id":1, "name":"小明"})

# 删除指定数据
es.delete(index='test', doc_type='_doc', id=1)

#修改字段
es.update(index = "test", doc_type = "_doc", id = 1, body = {"doc":{"name":"张三"}})
```

### 查询

[Python Elastic search 官网文档](https://elasticsearch-py.readthedocs.io/en/v8.4.2/api.html?highlight=search), [推荐 ES 语法文档](https://www.elastic.co/guide/cn/elasticsearch/guide/current/query-dsl-intro.html)

最基础的查询语法：

```python
# 定义过滤字段，最终只显示此此段信息
filter_path=['hits.hits._source.ziduan1',  # 字段 1         'hits.hits._source.ziduan2']  # 字段 2

body = {
    'query': {
        'match': {  
            'type': '我爱你中国'
        }
    }
}

a = es.search(index='yourindex', filter_path=filter_path, body=body, size=20)
for hit in a['hits']['hits']:
    print(hit)
```

注意，以上方法添加 `filter_path` 后，结果中将只有 `_source` 内容，不会显示 `_score` 等元数据。`size` 为返回最相关数据的数量。相似文档查询 [官方链接](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html)

## 其他笔记

### 如何批量导入？



### 与 huggingface datasets 如何配合？

[TLM 参考](https://github.com/yaoxingcheng/TLM)datasets 与 ES 有配合 [huggingface datasets 官网](https://huggingface.co/docs/datasets/installation),：

- [add_elasticsearch_index](https://huggingface.co/docs/datasets/v2.6.1/en/package_reference/main_classes#datasets.Dataset.add_elasticsearch_index)：能够

```python
data_files = "./TLM/example_data/source.csv"
ds = load_dataset("text", data_files=data_files)["train"]
es = Elasticsearch("http://localhost:9200")

if not es.indices.exists(index="tt"):
    ds.add_elasticsearch_index(
        "text",
        es_client=es,
        es_index_name="tt",
        es_index_config=es_config,
    )
else:
    ds.load_elasticsearch_index(
                "text",
                es_client=es,
                es_index_name='tt'
            )
body = {
    'query': {
        'match': {  
            'text': 'hello world'
        }
    }
}

print(ds.get_nearest_examples("text","hello world",k=1))
```

通过上述方案，能够将 huggingface 的 datasets 导入到 es 数据库中，而后通过 es 数据库的方式进行检索。

### 如何备份与转移数据?

[网络资源很多](https://cloud.tencent.com/developer/article/1839060)

### ES 与 Solr

1、 **es** 基本是 **开箱即用** (解压就可以用!) ,非常简单。Solr 安装略微复杂一丢丢!
2、 **Solr 利用 Zookeeper 进行分布式管理** ,而 **Elasticsearch 自身带有分布式协调管理功能。** 
3、Solr 支持更多格式的数据,比如 JSON、XML、 CSV ,而 **Elasticsearch 仅支持 json 文件格式** 。
4、Solr 官方提供的功能更多,而 Elasticsearch 本身更注重于核心功能，高级功能多有第三方插件提供，例如图形化界面需要 kibana 友好支撑
5、 **Solr 查询快,但更新索引时慢(即插入删除慢)**  ，用于电商等查询多的应用;

-  **ES 建立索引快(即查询慢)**  ，即 **实时性查询快** ，用于 facebook 新浪等搜索。
- Solr 是传统搜索应用的有力解决方案，但 Elasticsearch 更适用于新兴的实时搜索应用。

6、Solr 比较成熟，有一个更大，更成熟的用户、开发和贡献者社区，而 Elasticsearch 相对开发维护者较少,更新太快,学习使用成本较高。
