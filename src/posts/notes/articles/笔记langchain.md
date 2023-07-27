---
title: LangChain 源码理解
date: 2023-05-15
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
- AIGC
---

# LangChain

link: https://github.com/hwchase17/langchain

langchain 算是工程类的仓库，大模型应用的框架，提供了大模型和个人数据交互的框架，目前很多应用都有着和 langchain 非常相似的架构和目的，如 LLAMA INDEX，haystack， semantic-kernel 等。

主要的应用场景有：[私人助理](https://python.langchain.com/en/latest/use_cases/personal_assistants.html) ，[问答系统](https://python.langchain.com/en/latest/use_cases/question_answering.html) 等。

主要特点在于：

+ 支持异构数据输入
+ 提供了各种 Plugin 的实现方案
+ 支持除文本回复之外的反馈操作，如执行代码等。

::: warning

由于 LangChain 在快速更新截断，因此文中的示例代码，可能与源码存在差异。但大致上，框架的逻辑思路是相同的。

:::

## 源码框架梳理

### Models

LangChain 中的模型相关代码较少，大部分主要是调用外部资源，如 OPENAI 或者 Huggingface 等模型/API。Model 模块下提供了 `llms`, `cache`, `embeddings` 等子模块。

#### llms

llms 用于输入 input 文本，输出文本回复。该模块保存了各种 llm 接口。

```python
from langchain import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(model_id="bigscience/bloom-1b7", task="text-generation", model_kwargs={"temperature":0, "max_length":64})

print(llm("Hello, what is apple?"))
```

::: info

在 langchain 中自定义 LLM 的话，只需要重写 `_call()` 方法即可。使其接受一个字符串，输出一个字符串

:::

#### cache

当用户对相同的问题进行提问时，如果配置了 cache 的话，回复会更快。

```python
from redis import Redis
from langchain.cache import RedisCache

langchain.llm_cache = RedisCache(redis_=Redis())
```

#### embeddings

`embeddings` 输入文本，输出对应的 embedding。主要用于编码本地数据以及用户 query，以方便检索。langchain 整合了部分开箱即用的 embedding 服务，包括比较流行的 sentance Transformer 库：

```python
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
query_result = embeddings.embed_query("your text input")
doc_result = embeddings.embed_documents(["text input1", "text input2."])
```

##### 自定义 Embedding

如果需要使用自定义的模型进行编码的话，仅需重写 `embed_documents`， `embed_query` 两个方法即可

如 `embeddings.huggingface` 下的 `HuggingFaceEmbeddings`，使用 `sentence_transformers` 对文本进行编码，那么  `embed_documents`， `embed_query` 的实现方式为：

```python
#   self.client = sentence_transformers.SentenceTransformer(self.model_name)

def embed_documents(self, texts: List[str]) -> List[List[float]]:
    texts = list(map(lambda x: x.replace("\n", " "), texts))
    embeddings = self.client.encode(texts)
    return embeddings.tolist()

def embed_query(self, text: str) -> List[float]:
    text = text.replace("\n", " ")
    embedding = self.client.encode(text)
    return embedding.tolist()
```

### Indexes 和 数据库

Langchain 提供好了 `index`, `document_store` 等模块，能够方便地进行数据库管理、异构数据处理、数据读写等操作。Indexes 主要用于数据检索。

#### document_loaders

`document_loaders` 主要用于管理数据库元数据。提供了读写文件时候必要的方法，包括文件地址，文件读取方式，文件分段方式等。

```python
from langchain.document_loaders import TextLoader
loader = TextLoader('./mydoc.txt', encoding='utf8')
```

如上创建 loader 之后，loader 中储存了 `loader.file_path =  "./mydoc.txt"` 等元数据。

#### text splitter

通常，我们需要对一整篇文章进行分段，才能方便我们对段落进行检索。langchain 提供了部分 splitter，可以针对 markdown，pdf 等进行分段。分段后的文章，会被储存成 `Document` 数据格式。

```python
markdown_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)

mymdstr = """### this is md content"""
## 可以使用 create_documents 从字符串列表直接获得 List[Documents]
docs = markdown_splitter.create_documents([mymdstr])

## 也可以使用 split_documents() 为现有的 Documents 进行二次细分
docs = markdown_splitter.split_documents(documents)
```

如上代码， `docs` 为 `List[Document]`

#### vectorstores

`vectorstores` 用于储存文档、数据及其对应 embedding 等信息的引擎。

比如，要将文档与对应的 embedding 储存与 ElasticSearch 当中，可以参考 `elastic_vector_search.py` 中的内容。几个比较常见的数据库以及 NLP 检索框架在 langchain 中都写好了对应的 vectorstores， 如 `redis`, `faiss`, `elasticsearch` 等。以下为使用 `FAISS` 进行相似文档检索的操作：

```python
from langchain.vectorstores import FAISS

# 可以更换自定义 Embedding
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(docs, embeddings)
query = "What did the president say about Ketanji Brown Jackson"
similar_docs = db.similarity_search(query)
```

要支持其他数据库的话，可以自定义 vectorstore 类。自定义的话需要继承 `base.VectorStore`，而后重写下面几个主要的功能函数即可：

1. `add_texts`, `from_texts`：往数据库中添加数据的一些操作
2. `similarity_search`：从数据库中检索本文的一些操作。该操作自定义程度搞，如对于  `elastic search`， 可以将 `retrievers` 下的 `elastic_search_bm25` 召回，以及 `elastic_vector_search` 的精排结合使用，以进行性能调优。

自定义 vectorstores 可以考虑参考 `vectorstores.redis` ，使用另个一类 `RedisVectorStoreRetriever`来负责数据的检索。

### chains

Chain 的作用相当于 pipeline，主要将多个 LLM 模块和程序处理环节进行拼接，将复杂任务成应用。有点像 transformers 中的 pipeline。以下是基础的 LLMChain 示例：

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["thing"],
    template="Hello, what is {thing}?",
)
mychain = LLMChain(llm=llm, prompt=prompt, verbose=True)
print(mychain.run("banana"))
```

以上示例对输入进行 **模板填充** ，而后进行  **LLM 推理** ，将答案整理并返回给用户。Chain 之间能够相互组合：

```python
from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

# Run the chain specifying only the input variable for the first chain.
catchphrase = overall_chain.run("colorful socks")
```

[官方文档](https://python.langchain.com/en/latest/modules/chains/how_to_guides.html) 中有部分 chain 的示例代码，在源码 `chains.__init__` 中可以查看到所有默认的 chain。一些比较有意思的 chain：

1. `ChatVectorDBChain`：从 vectorstore 检索和 query 最相近的 k 条文档；而后根据这些文档生成回复。
2. `LLMBashChain`：先通过  `LLMChain` 对 query 生成回复，如果回复中包含了 ````bash` 字样，则执行对应位置的代码。
3. `LLMMathChain`：首先让 LLM 将需要数学计算的部分用 ````python` 进行标注，而后通过 python 进行对应的数学计算返回结果。
4. `SQLDatabaseChain`：先用 LLM 解析 query，输出 sql；用 sql database 执行 sql 语句，抽取回复 query 时需要用到的数据；根据检索的数据，将结果返回给用户。对于这种 chain，以下是一个可以参考的 prompt：

```python
"""Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the tables listed below.

{table_info}

Question: {input}"""
```

自定义 chain 需要重写下面三种方法：

```python
from langchain.chains import LLMChain
from langchain.chains.base import Chain

from typing import Dict, List

class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains.
        all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['concat_output']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + output_2}
```

### Agent

Agent 是 AI 的一个关键概念。Agent 能够根据不同的问题，决定要执行哪一个 chain。以下是一个基础 agent 的调用示例。

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
```

以下是上面这串代码执行后的结果：

```python
> Entering new AgentExecutor chain...
 I need to find out who Leo DiCaprio's girlfriend is and then calculate her age raised to the 0.43 power.
Action: Search
Action Input: "Leo DiCaprio girlfriend"
Observation: Camila Morrone
Thought: I need to find out Camila Morrone's age
Action: Search
Action Input: "Camila Morrone age"
Observation: 25 years
Thought: I need to calculate 25 raised to the 0.43 power
Action: Calculator
Action Input: 25^0.43
Observation: Answer: 3.991298452658078

Thought: I now know the final answer
Final Answer: Camila Morrone is Leo DiCaprio's girlfriend and her current age raised to the 0.43 power is 3.991298452658078.

> Finished chain.
```

#### Agent 单元

一个 Agent 单元负责执行一次任务。如 langchain 中的 `langchain.agents.agent.Agent` 

```python
class Agent(BaseSingleActionAgent):
    llm_chain: LLMChain
    allowed_tools: Optional[List[str]] = None
```

agent 的执行功能在于 `Agent.plan()`：

```python
def plan(intermediate_steps: List[Tuple[AgentAction, str]], **kwargs) :   
    # 将各种异构的历史信息转换成 inputs，传入到 LLM 当中
    thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
            
    full_input = {"agent_scratchpad": thoughts, "stop": self._stop, **kwargs}  
    
    # 根据 LLM 生成的反馈，采取决策
    ## prompt template: "your input xx {other prompt slot} xx, {agent_scratchpad}"
    ## full_output: LLM 输出的回复，字符串形式
    full_output = self.llm_chain.predict(**full_inputs)
    
    parsed_output = self._extract_tool_and_input(full_output)
    while parsed_output is None:
        full_output = self._fix_text(full_output)
        full_inputs["agent_scratchpad"] += full_output
        output = self.llm_chain.predict(**full_inputs)
        full_output += output
        parsed_output = self._extract_tool_and_input(full_output)
        
    # 或者返回 AgentFinish({"output": action.tool_input}, action.log)
    return AgentAction(
            tool=parsed_output[0], tool_input=parsed_output[1], log=full_output
        )
```

对于 Agent，能够从历史操作中获取信息，并根据目的做出决策是很重要的。对于 Langchain 的 Agent，我们解析他的 `.plan()` 方法：

`.plan()` 的参数有 `intermediate_steps` 和 `**kwargs`，前者保存了 agent 之前做过的一些行为。后者通常会储存 agent 的目标，如用户提出的 query 等。

`.plan()` 可以看做两步：

1. 将各种异构的历史信息转换成 inputs，传入到 LLM 当中；对于 Langchain，其做法就是简单的将 `intermediate_steps` 中的 action 日志和 `observation` 进行拼接，而后统一放在 `inputs` 后面。在 langchain 的 prompt template 当中，经常会看到 `agent_scratchpad` 的填充字段，它就是用来放历史信息的。
2. 根据 LLM 生成的反馈，采取决策。LLM 生成的回复是 string 格式，我们会将回复出来的 `full_output` 在 `_extract_tool_and_input` 中进行处理。一个例子就是 `agents.mrkl.base.get_action_and_tool()`。

```python
def get_action_and_input(llm_output: str) -> Tuple[str, str]:
    """Parse out the action and input from the LLM output.

    Note: if you're specifying a custom prompt for the ZeroShotAgent,
    you will need to ensure that it meets the following Regex requirements.
    The string starting with "Action:" and the following string starting
    with "Action Input:" should be separated by a newline.
    """
    if FINAL_ANSWER_ACTION in llm_output:
        return "Final Answer", llm_output.split(FINAL_ANSWER_ACTION)[-1].strip()
    # \s matches against tab/newline/whitespace
    regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
    match = re.search(regex, llm_output, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse LLM output: `{llm_output}`")
    action = match.group(1).strip()
    action_input = match.group(2)
    return action, action_input.strip(" ").strip('"')
```

可以看到，langchain 中`ZeroShotAgent` 通过字符串匹配的方式来识别 action。因此，agent 能否正常运行，与 prompt 格式，以及 LLM 的 ICL 以及 alignment 能力有着很大的关系。

::: info

最后的输出 `AgentAction` 中会包括：需要使用的 tool，使用该 tool 时候，对应的执行命令。可以使用的 tool 会在 prompt template 中提示出来，比如如果你采用 `create_pandas_dataframe_agent` 构造了一个 agent，那么它的 prompt template 就会编程：

```python
"""
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the question posed of you:
python_repl_ast
A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "When using this tool, sometimes output is abbreviated - "
        "make sure it does not look abbreviated before using it in your answer."
This is the result of `print(df.head())`:
Begin!
Question: {input}
{agent_scratchpad}
"""
```

:::

#### AgentExecutor

不论是通过 `initialize_agent` 或者是 `create_pandas_dataframe_agent` 等方式，我们都可以得到 `AgentExecutor` 来执行 agent 相关的任务。

`AgentExecutor` 实际上是一个 `Chain`，可以通过 `.run()` 或者 `_call()` 来调用。如：

```python
agent_executor = create_pandas_dataframe_agent(llm=llm, df=df)
agent_executor.run("how many rows are there?")
```

`agent_executor.run()` 通过迭代地执行 agent 

```python
intermediate_steps = [] # List[(AgentAction, str)]
# 其中 str 为 oservation，是每次迭代后，我们调用 tool 得到的结果。
agent_action = self.agent.plan(
                intermediate_steps,
                **inputs)
# inputs 是一个用于填充 prompt.format 的字典，参考 prompts 部分。

for agent_action in actions:
	tools2run = name_to_tool_map[agent_action.tool]
    observation = tool.run(agent_action.tool_input, **tool_run_kwargs,)

intermediate_steps.append([agent_action, observation])
```

因此，结合上面 Agent 部分的 `plan`，我们就可以理解 langchain 中 Agent 执行的大致逻辑。

### Prompts

Langchain 当中，定义好了不同的 Prompt template，以面对不同的用户提问，prompt template 应该与模型绑定。Prompts 模块当中也提供了 example selector，方便用户进行 few shot 选择。

### Callback

可以在各个环节对你的 LLM 应用进行监控， callback 可以用于各个环节，如 Chain 开始，Chain 结束，LLM 开始， LLM 出错，接受到新的 Token 等等。

```python
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

handler = StdOutCallbackHandler()
llm = OpenAI()
prompt = PromptTemplate.from_template("1 + {number} = ")

# First, let's explicitly set the StdOutCallbackHandler in `callbacks`
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])
chain.run(number=2)
```

自定义 callback 需要实现 `langchain.callbacks.base.BaseCallBackHandler` 中的接口。

### Other Utils

#### Output Parser

#### API 工具

在 `tools` 目录下，可以看到各种适配 Plugin 等插件。支持通过 python 运行各种脚本、调用各类 API 。

