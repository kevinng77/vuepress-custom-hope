---
title: Azure AI Search 基础
date: 2024-03-15
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
- AIGC
---



# Azure AI Search

Azure AI Search formerly known as "Azure Cognitive Search"

[产品文档参考](https://learn.microsoft.com/en-us/python/api/overview/azure/search-documents-readme?view=azure-python)

[API 参考文档](https://azuresdkdocs.blob.core.windows.net/$web/python/azure-search-documents/latest/index.html)

[示例](https://github.com/Azure/azure-sdk-for-python/tree/azure-search-documents_11.4.0/sdk/search/azure-search-documents/samples)

## 0. 概念概述

### 0.1 关键概念

-  **Data Source** : 持久化源数据连接信息，包括凭证。数据源对象专门用于索引器。.
-  **Index** : 用于全文搜索和其他查询的物理数据结构。
-  **Indexer** : 配置对象，指定数据源、目标索引、可选的 AI 技能集、可选的调度，以及错误处理和 base-64 编码的可选配置设置。
-  **Skillset** : 操作、转换和整理内容的完整指令集，包括从图像文件中分析和提取信息。除了非常简单和有限的结构外，它包括对提供增强的 Azure AI 服务资源的引用。
-  **Knowledge store** : 在 Azure 存储的表和 blob 中存储来自 AI 增强管道的输出，以便独立分析或下游处理。

### 0.2 关键类

- `SearchClient`:
  - [搜索](https://docs.microsoft.com/azure/search/search-lucene-query-architecture) 索引文档，使用 [丰富的 queries](https://docs.microsoft.com/azure/search/search-query-overview) 和 [强大的数据整形](https://docs.microsoft.com/azure/search/search-filters)
  - [自动完成](https://docs.microsoft.com/rest/api/searchservice/autocomplete) 基于索引中文档的部分输入的搜索词
  - [建议](https://docs.microsoft.com/rest/api/searchservice/suggestions) 文档中最可能匹配的文本作为用户输入
  - 从索引中[添加、更新或删除文档](https://docs.microsoft.com/rest/api/searchservice/addupdate-or-delete-documents) 
- `SearchIndexClient` ([source code](https://github.com/Azure/azure-sdk-for-python/blob/azure-search-documents_11.4.0/sdk/search/azure-search-documents/azure/search/documents/indexes/_search_index_client.py#L31)):
  - [创建、删除、更新或配置搜索索引 index](https://docs.microsoft.com/rest/api/searchservice/index-operations)
  - [声明自定义同义词映射以扩展或重写查询 queries](https://docs.microsoft.com/rest/api/searchservice/synonym-map-operations)
  - Analyze text 分析文本
- `SearchIndexerClient`:
  - [启动索引器 indexer 自动爬取数据源](https://docs.microsoft.com/rest/api/searchservice/indexer-operations)
  - [定义 AI 驱动的技能集以转换和丰富你的数据](https://docs.microsoft.com/rest/api/searchservice/skillset-operations)


```bash
pip install azure-search-documents
```

# 快速概述

### 1. 在 Azure 门户上创建 AI 搜索服务


要创建新的搜索服务，你可以使用 [Azure portal](https://learn.microsoft.com/en-us/azure/search/search-create-service-portal), [Azure PowerShell](https://learn.microsoft.com/en-us/azure/search/search-manage-powershell#create-or-delete-a-service), 或 [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/search/service?view=azure-cli-latest#az-search-service-create)..




```python
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    ComplexField,
    CorsOptions,
    ScoringProfile,
    SearchIndex,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    HnswParameters,
    VectorSearchProfile,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SemanticConfiguration,
    SemanticField,
    SemanticSearch,
    SemanticPrioritizedFields,
)

service_endpoint = ""  # Your Azure AI Search Endpoint (The Azure AI Search URL)
key = ""  # Your Azure AI Search Key

index_name = "kevin_tests"
```

### 2. 创建索引

在 Azure AI 搜索中，索引是用于启用搜索功能的 JSON 文档和其他内容的持久集合。

Azure AI 搜索需要知道你希望如何搜索和显示文档中的字段。你可以通过为这些字段分配属性或行为来指定这一点。对于文档中的每个字段，索引存储其名称、数据类型和支持字段的行为，例如字段是否可搜索、字段是否可以排序？

最高效的索引仅使用所需的行为。如果在设计时忘记设置字段上的必要行为，获取该功能的唯一方法是重建索引。

[参考 index CRUD 代码](https://github.com/Azure/azure-sdk-for-python/blob/azure-search-documents_11.4.0/sdk/search/azure-search-documents/samples/sample_index_crud_operations.py).

```python
## create index
from azure.search.documents.indexes import SearchIndexClient
from typing import List


client = SearchIndexClient(service_endpoint, AzureKeyCredential(key))
fields = [
    SimpleField(name="hotelId", type=SearchFieldDataType.String, key=True),
    SimpleField(name="baseRate", type=SearchFieldDataType.Double),
    SearchableField(
        name="description", type=SearchFieldDataType.String, collection=True
    ),
    SearchableField(name="title", type=SearchFieldDataType.String, collection=True),
    SearchField(
        name="vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=2,
        vector_search_profile_name="myHnswProfile",
    ),
    ComplexField(
        name="address",
        fields=[
            SimpleField(name="streetAddress", type=SearchFieldDataType.String),
            SimpleField(name="city", type=SearchFieldDataType.String),
        ],
        collection=True,
    ),
]

# Vector Search Reference:
# 1. https://learn.microsoft.com/en-us/azure/search/vector-search-overview
# 2. https://learn.microsoft.com/en-us/azure/search/vector-search-ranking
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(
                m=4,
                ef_construction=400,
                ef_search=500,
                metric=VectorSearchAlgorithmMetric.COSINE,
            ),
        ),
        ExhaustiveKnnAlgorithmConfiguration(
            name="myExhaustiveKnn",
            kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
            parameters=ExhaustiveKnnParameters(
                metric=VectorSearchAlgorithmMetric.COSINE
            ),
        ),
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw",
        ),
        VectorSearchProfile(
            name="myExhaustiveKnnProfile",
            algorithm_configuration_name="myExhaustiveKnn",
        ),
    ],
)

# Semantic Search Reference:
# 1. https://learn.microsoft.com/en-us/azure/search/semantic-search-overview
semantic_config = SemanticConfiguration(
    name="adp-test-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="title"),
        keywords_fields=[
            SemanticField(field_name="filename"),
            SemanticField(field_name="title"),
            SemanticField(field_name="headings"),
        ],
        content_fields=[SemanticField(field_name="content")],
    ),
)

semantic_search = SemanticSearch(configurations=[semantic_config])


cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
scoring_profiles: List[ScoringProfile] = []  # Config Profile Score, Default tf-idf

index = SearchIndex(
    name=index_name,
    fields=fields,
    scoring_profiles=scoring_profiles,
    cors_options=cors_options,
    vector_search=vector_search,
    # semantic_search=semantic_search,
)

exist_index_name = [x.name for x in client.list_indexes()]
if index_name not in exist_index_name:
    result = client.create_or_update_index(index)
else:
    print(f"Index Name {index_name} Already Exist")
# result
```

### 3. 上传文档

[Azure sample](https://github.com/Azure/azure-sdk-for-python/blob/azure-search-documents_11.4.0/sdk/search/azure-search-documents/samples/sample_crud_operations.py).

```python
search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
DOCUMENT = [
    {
        "hotelId": "1000",
        "baseRate": 4.0,
        "description": ["A sample Document"],
        "title": ["A sample Document"],
        "address": ["Chaim Time", "Shanghai"],
        "vector": [1.0, 2.0],  # Dummy vector - replace with your embedding
    },
    {
        "hotelId": "1001",
        "baseRate": 13.0,
        "description": ["hello world"],
        "title": ["hotel2"],
        "address": ["Chaim Time", "Shanghai"],
        "vector": [10.0, 20.0],
    },
    {
        "hotelId": "1002",
        "baseRate": 13.0,
        "description": ["hello world2"],
        "title": ["hotel2"],
        "address": ["Chaim Time", "Shanghai"],
        "vector": [1.0, 2.0],
    },
]

result = search_client.upload_documents(documents=DOCUMENT)

print("Upload of new document succeeded: {}".format(result[0].succeeded))
```


### 4. 搜索

参考:

- [hybird search](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)
- [Full Text Search](https://learn.microsoft.com/en-us/azure/search/search-lucene-query-architecture)
- [Vector Search](https://learn.microsoft.com/en-us/azure/search/vector-search-overview)

```python
### text search
from azure.search.documents.models import VectorizedQuery

search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
vector_query = VectorizedQuery(vector=[0.5, 1], k_nearest_neighbors=1, fields="vector")

results = search_client.search(
    search_text="hotel2", search_fields=["description"], vector_queries=[vector_query]
)
for result in results:
    print(result)
```

#### 4.1 Full Text Search

1. 用 Analyser 分析器对文档进行分词。
2. 使用倒排索引进行[文档检索](https://learn.microsoft.com/en-us/azure/search/search-lucene-query-architecture#stage-3-document-retrieval)。
3. 使用句子 embedding 进行 [Scoring](https://learn.microsoft.com/en-us/azure/search/search-lucene-query-architecture#stage-4-scoring). 可以通过 `scoring_profiles` 调整。

#### 4.2  向量搜索

当前支持 HNSW 和 KNN。

#### 4.3 Hybird Search

1. 单个查询请求包含搜索和向量查询参数
2. 并行执行
3. 在查询响应中合并结果，使用 [Reciprocal Rank Fusion (RRF)](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking) 进行评分。（[RRF 算法详细示例](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html)）

RRF 计算方法：
```python
score = 0.0
for q in queries:
    score += 1.0 / ( k + rank(q,d) )
return score
# 其中 d 为 文档，rank(q,d) 为不同 query 算法的排名
```

Hybird Search 评分计算流程图：

![Hybird Search 评分计算流程图](https://learn.microsoft.com/en-us/azure/search/media/hybrid-search/search-scoring-flow.png)



### 5. Indexer 索引器

你可以使用 [indexers](https://learn.microsoft.com/en-us/azure/search/search-indexer-overview) 自动化数据摄取，并使用 [integrated vectorization](https://learn.microsoft.com/en-us/azure/search/vector-search-integrated-vectorization) 生成嵌入。Azure AI 搜索可以从两个数据源自动索引向量数据： [Azure blob indexers](https://learn.microsoft.com/en-us/azure/search/search-howto-indexing-azure-blob-storage) 和 [Azure Cosmos DB for NoSQL indexers](https://learn.microsoft.com/en-us/azure/search/search-howto-index-cosmosdb). 

示例代码： [indexer crud operations example](https://github.com/Azure/azure-sdk-for-python/blob/azure-search-documents_11.4.0/sdk/search/azure-search-documents/samples/sample_indexers_operations.py) 和 [indexer datasource skillset example](https://github.com/Azure/azure-sdk-for-python/blob/azure-search-documents_11.4.0/sdk/search/azure-search-documents/samples/sample_indexer_datasource_skillset.py)。


以下显示了如何使用索引器连接 Azure BLOB 存储的快速开始：

```python
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexer,
)

connection_string_key = "your blob storage account key"
connection_account_name = "your blob storage account name"
connection_string = f"DefaultEndpointsProtocol=https;AccountName={connection_account_name};AccountKey={connection_string_key}"
```

```python
indexers_client = SearchIndexerClient(service_endpoint, AzureKeyCredential(key))

# create a datasource
# IMPORTANT: Please make sure that you have create a container named `container_name`
container_name = "searchcontainer"
container = SearchIndexerDataContainer(name=container_name)
data_source_connection = SearchIndexerDataSourceConnection(
    name="indexer-datasource-sample",
    type="azureblob",
    connection_string=connection_string,
    container=container,
)
data_source = indexers_client.create_data_source_connection(data_source_connection)

# create an indexer
indexer = SearchIndexer(
    name="sample-indexer",
    data_source_name="indexer-datasource-sample",
    target_index_name=index_name,
)
result = indexers_client.create_indexer(indexer)
print("Create new Indexer - sample-indexer")
```

Azure AI 搜索中的索引器是一个爬虫，它从云数据源提取文本数据并使用源数据与搜索索引之间的字段到字段映射填充搜索索引。索引器还驱动[技能集执行和 AI 增强](https://learn.microsoft.com/zh-cn/azure/search/cognitive-search-concept-intro)，你可以配置技能以在内容进入索引的途中集成额外的处理。


AI 搜索内置了技能，并且我们也可以自定义技能。

### 6. Analyzer 分析器

[分析文本示例](https://github.com/Azure/azure-sdk-for-python/blob/azure-search-documents_11.4.0/sdk/search/azure-search-documents/samples/sample_analyze_text.py)，更多详情，请参考`AnalyzeText`源码。


在 Azure AI 搜索中 `Analyzer = Tokenizer + TokenFilter + CharFilter`. AI 搜索的 analyzeText 具有内置的分析器。也可以自定义这些分析器。

```python
# Analyzer Can use for Debug for performance analysis
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import AnalyzeTextOptions

client = SearchIndexClient(service_endpoint, AzureKeyCredential(key))

analyze_request = AnalyzeTextOptions(
    text="One's <two/>", analyzer_name="standard.lucene"
)

result = client.analyze_text(index_name, analyze_request)
print(result.as_dict())
""" result.as_dict()
{'tokens': [{'token': "one's",
'start_offset': 0,
'end_offset': 5,
'position': 0},
{'token': 'two', 'start_offset': 7, 'end_offset': 10, 'position': 1}]}
"""
```