---
title: RASA 回忆录（一） - 自定义 RASA NLU 模块
date: 2022-09-25
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
---

RASA 感觉像是时代的眼泪了，自动 LLM 火了之后，类似 RASA 一类的传统 NLP 对话架构变得冷门，不少技术似乎也被摒弃。但如今不少厂家在实现对话机器人、智能服务等 AIGC 服务，却也会用到这类型的框架。市面上提供类似 RASA 服务的 low code 平台也不少，如 Nuance, Omilia, Kore.ai 等。

该系列文章主要围绕 RASA 源码，对落地部署传统 NLP 对话框架的相关话题展开。本文针对 RASA 自定义 NLU 功能展开。

RASA 默认的 NLU 策略不支持比如细腻度情感分析、指代消解等方案。通过自定义 NLU 模块，我们可以任意 NLU 方案，比如添加额外的实体链接、实体纠错、信息抽取模块。

## 传统 NLP 对话框架

以下是一种传统的 NLP 对话框架，用户输入语句后，会经过 NLU 和 Dialog 两个模块，最后得到回复。

NLU  模块：

2. 使用各种算法对语句进行 Embedding，然后将各种 Embedding 拼接。（比如用 Conv 1D 抽取上下文意思，对语句中单词进行分词后进行额外 Embedding 等）
3. 将拼接后的 Embedding 传入模型当中，如 LSTM，Transformer 等，得到 hidden state
4. 将 hidden state 链接不同的 output layer 进行不同 NLU 任务的解析。比如链接 linear layer 做多分类或实体识别，链接 pointer Network 做关系抽取等等。
5. 将所有意图整理在对话状态中，传入给 dialog 模块。

对话模块（dialog 模块）

1. 采用 rule base：提前设计好用户故事线，以买咖啡为例；那么我们需要用条件语句，分别从用户那边问道咖啡的大小、甜度、加冰与否、打包还是堂食等信息（类似槽位填充）。在获得所有信息之后，对用户的咖啡进行下单。
2. 采用其他方式：使用 rule base 的问题在于，我们需要多所有的场景进行规则构建，当用户问一点点规则以外的内容时，这个对话 session 可能会被影响，甚至打乱。因此有不少针对多轮对话的传统策略提出，比如将对话状态编码，与用户输入的 hidden state 一起生成回复等；或将用户输入与机器人的一些候选决策进行相关性匹配（RASA 的 Action 策略）。

## RASA 架构

RASA 的架构分成 nlu 模块和 core 模块。

Core 模块（即上文中的 dialog 模块）需要的操作有：

1. 写 Rule：定义如上文中提到的基于规则的对话逻辑
2. 写故事：
   - 定义机器人的 Action，比如机器人可能会发送邮件，或者为客户下单一个咖啡，或者让客户付款。
   - 准备一些多轮对话数据集作为训练集，用于训练模型的对话能力。比如训练集中会包含有用户提问以及机器人采取的对应 Action 信息。模型就会根据这部分信息，训练一个 Action 和 Query 匹配器。

NLU 需要的操作有：

1. 准备训练集：填写你要训练的 intent，entity 等信息。人工准备一些语料并针对不同任务进行打标签。
2. 配置 NLU 流程：选择你要进行的 NLU 流程，比如你希望用哪些 Embedding 进行拼接。而后采用那个模型架构对拼接后的 Embedding 进行解析，得到 hidden state。最后用哪些解析器来讲 hidden state 转换为输出结果。
3. 训练模型：RASA 提供了一键训练，训练速度和配置取决于你选择的 NLU 配置。

当然如何配置 RASA NLU，可参考官方给出的详细文档。以下针对自定义 RASA NLU 模块做介绍。

## RASA 自定义 NLU

### 目录

该节展示了

1. 如何使用 Huggingface 上的模型，作为 RASA 的 embedding 工具。（默认 RASA 采用 TF 模型）
2. 如何添加一个情感分析引擎（这在 RASA 中是没有的）。

通过这个展示，我们能够延申到以下功能：

1. 参考这个代码，我们能够实现使用任何形式的模型作为我们的 Feature/Embedding 工具，比如 paddle, onnx 等。
2. 当 RASA 默认的训练效果不好时，你应该意识到 RASA 的自带 Featurizer 存在很大问题：他不能够被训练。
3. RASA 默认使用 CLS 位置的 feature 作为 sentence embedding 进行分类，而其他位置的 feature 作为 sequence embedding 进行 NER 等任务。 **大部分的 Transformer 使用了 SUB-TOKEN 的分词方式，而 RASA 默认的 NER 方式时将 subtoken 对应的 embedding 结合，作为完成 token 的 embedding。** 
4. 猜测：在 featurizer 中，没有输出任何完整 token 信息，估计在 NER 环节使用的 token 信息，是从 Tokenizer 传过去的，因此如果 Featurizer 输出 feature 维度和 Tokenizer 对应不上，就会报错。
5. RASA 训练策略比较单调，如果你享受调参等自定义模型训练方法。可以在自己机器上训练好权重，然后把他放在 RASA 中直接用，这将大大减少模型训练时间。最后通过自定义 Component 集成到 RASA 中。
6. 我们能够在 RASA NLU 的基本功能（Intent，NER）上，添加上任意的 NLU 处理结果，比如添加额外的细腻度情感分析结果、添加额外的实体链接、实体纠错、信息抽取结果。

### 代码与实践

https://github.com/kevinng77/rasa_example/tree/master/examples/2_custom_clu

#### RASA 配置文件 Config.py 中的 Pipeline 

Pipeline 由多个 RASA GraphComponent 组成，当用户发出消息后，消息会  **依次**  经过 Pipeline 的每一个 GraphComponent 处理，以完成 NLU。比如一下是一个经典的 Pipeline 写法：

```
pipeline:
  - name: "WhitespaceTokenizer"  
  - name: "CountVectorsFeaturizer"
  - name: "DIETClassifier"
    epochs: 100
```

以上面的 Pipeline 为例，rasa 进行 nlu 的时候，会从上到下进行每一个 GraphComponent。

如果用伪代码来表示这个流程，就是：

```python
# 毕竟是伪代码，因此逻辑不会很严谨
message = parse_user_input
# message 为数据结构，在 `rasa.shared.nlu.training_data.message` 可查看。
for GraphComponent in pipeline.name:
    message = GraphComponent.process(message)
```

每个 Component 可以在 `rasa.rasa.nlu` 文件夹下面找到，如 `WhitespaceTokenizer` 对应 `rasa.rasa.nlu.tokenizer.whitespace_tokenizer.WhitespaceTokenizer`。

### 自定义  GraphComponent (for NLU)

使用自定义 NLU GraphComponent 需要以下几个步骤：

1. 写一个 `.py` 文件，里面定义好你要的 `GraphComponent`。本案例中的自定义 GraphComponent 都写在 Component 文件夹下面了。

2. 在 Pipeline 中引用对应的  `GraphComponent`

#### 1. 定义 Custom GraphComponent

在 `rasa.data.test_classes` 中，我们能够看到一些官方提供的 `GraphComponent` 自定义方法和模板。如 `nlu_component_skeleton.py`。从下面的代码中可以看出，GraphComponent 主要的入口就是 `create`, `train`, `process`, `process_training_data` 四个方法。

```python
@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER], is_trainable=True
)
class CustomNLUComponent(GraphComponent):
    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        # TODO: Implement this
        ...

    def train(self, training_data: TrainingData) -> Resource:
        # TODO: Implement this if your component requires training
        ...

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        # TODO: Implement this if your component augments the training data with
        #       tokens or message features which are used by other components
        #       during training.
        ...

        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        # TODO: This is the method which Rasa Open Source will call during inference.
        ...
        return messages
```

因此自定义的 `GraphComponent` 中，必须覆盖重写以上四个方法。我们根据  `rasa.rasa.nlu`  中的文件进行修改，尝试使用 pytorch 的模型来计算模型的 embedding。大致方法是继承 `rasa.nlu.featurizers.dense_featurizer.dense_featurizer` 中的 `DenseFeaturizer` 抽象类，

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class LanguageModelFeaturizer(DenseFeaturizer, GraphComponent):
	#...
```

具体查看该文件夹下的 `components/myintent.py`。比较值得注意的点是： **默认情况下，所有的 Featurizer 都是不能被 train 的** ，包括 RASA 自带的 TF transformers。而我们知道，Transformer 系列小模型在不微调的情况下，效果是不怎么好的。因此我们可以考虑，

1. 将 Featurizer 和 Intent Classifier 合并成一个 Compoent，而后统一在 RASA 中训练。
2. 或者选个领域预训练+微调好的 embedding transformer
3. 其他骚操作

#### 2. Pipeline 中引用自定义模块

我们将原先的词袋模型替换为我们自定义的模型 `components.myintent.LanguageModelFeaturizer`， 并使用 huggingface 上的权重`kevinng77/TinyBERT_4L_312D_SIMCSE_finetune`，这便是自定义模型的好处之一：这是一个在个人 GPU 上蒸馏好的模型，速度比标准 bert 快 20+倍，且在 NLI 数据集上的精度（较 SIMCSE）保留了 98%。我们能够进一步对他进行量化、检索、推理部署等，以进一步提高预测速度。

```python
pipeline:
  - name: "WhitespaceTokenizer"
  - name: components.myintent.LanguageModelFeaturizer
    model_name: kevinng77/TinyBERT_4L_312D_SIMCSE_finetune 
  - name: "DIETClassifier"
    epochs: 100
```

#### 3. 我们添加额外的情感分析模块，从用户回答中提取更多信息

根据 `rasa.nlu` 下的代码，大致可以猜测到，整个 Pipeline 的 NLU 过程中，所有的结果都会被记录在 `Message` 上。

因此如果我们想要添加额外的 nlu 信息，如实体间关系、细腻度情感分析。那么，我们就可以自定义 `Component` 模块，而后通过 `process()` 函数，将 NLU 处理的结果添加到 `Message` 中就行。`Message` 中有 `data` 字典，可以用来储存其他特征信息。

比如说，我想要在 intent 分析和 NER 分析的基础上，加上一层情感分析：

```yml
pipeline:
  - name: "WhitespaceTokenizer"
  - name: components.myintent.LanguageModelFeaturizer
    model_name: distilbert
  - name: components.sentiment_classifier.SentimentClassifier
  - name: "DIETClassifier"
    epochs: 50
```

在 `components.sentiment_classifier.SentimentClassifier` 中，我们提供以下方法（具体可查看文件夹中代码）：

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=False
)
class SentimentClassifier(GraphComponent):
    """Intent classifier using the Logistic Regression."""
    
    # ...
    
	def process(self, messages: List[Message]) -> List[Message]:
        """Return the most likely intent and its probability for a message."""

        for idx, message in enumerate(messages):
            sentiment, score = "Positive", 0.666  # this should come from your model
            message.set("sentiment", sentiment, add_to_output=True)
            message.set("sentiment_confidence", score, add_to_output=True)
        return messages
```

那么在输出结果中，我们就能看到 `Message.data` 中，多了情感分析的结果，执行以下语句进行测试 ：

```
rasa train nlu
rasa shell
```

输入 `hello world`，系统回复内容中，就会多出来 `sentiment` 和 `sentiment_confidence` 两个字段了：

```sh
NLU model loaded. Type a message and press enter to parse it.
Next message:
hello world
{
  "text": "hello world",
  "intent": {
    "name": "greet",
    "confidence": 0.4089217782020569
  },
  "entities": [],
  "text_tokens": [
    [
      0,
      5
    ],
   # ...
  ],
  "sentiment": "Positive",
  "sentiment_confidence": 0.666,
  "intent_ranking": [
    {
      "name": "greet",
      "confidence": 0.4089217782020569
    },
	#...
  ]
}
```

### RASA NLU 部署

默认 RASA NLU 是单独部署成一个 API 的。因此我们可以讲实现好的自定义 NLU 模块部署成 API，而后进行测试。首先运行服务：

```sh
rasa run --enable-api
```

发送 POST 请求到 `http://localhost:5005/model/parse`， body 中为：

```json
{
    "text":"what restaurants do you recomment?",
    "sender": "test_user"
}
```

RASA 返回 Message 结果。而 Message 架构如下：

```json
// request 返回的结果
{
  "text": "what restaurants do you recomment?",  // 用户 query
  "intent": {
    "name": "query_knowledge_base",
    "confidence": 1.0
  },  // 最终判断意图
  "entities": [ // 所有实体抽取的结果
    {
      "entity": "object_type",
      "start": 5,
      "end": 16,
      "value": "restaurants",
      "extractor": "RegexEntityExtractor"
    },
    {
      "entity": "object_type",
      "start": 5,
      "end": 16,
      "confidence_entity": 0.9987316727638245,
      "value": "restaurants",
      "extractor": "DIETClassifier"
    }
  ],
  "text_tokens": [
    [0, 4],
    [5, 16],
    // ... 所有 token 对应的 idx
  ],
  "intent_ranking": [
    {
      "name": "query_knowledge_base",
      "confidence": 1.0
    },
    {
      "name": "greet",
      "confidence": 4.942866560497805e-9
    }
  ]
}
```

其他的 API 服务接口可以在 [RASA API](https://rasa.com/docs/rasa/pages/http-api) 查看。

### 进一步解析结果来源

在 `rasa.server.create_app` 中，我们可以找到 NLU api 的入口：

```python
@app.post("/model/parse")
@requires_auth(app, auth_token)
@ensure_loaded_agent(app)
async def parse(request: Request) -> HTTPResponse:
    # NLU 处理
    return response.json(response_data)
```

其中的 NLU 处理流程，我们可以在 `rasa.rasa.core.agent.Agent.parse_message` 查看到。

在上一个仓库`2-custom_nlu`  中，我们提到了 RASA nlu 的执行单元 `GraphComponent` 可以在 `rasa/rasa/nlu` ，如果你定义了这样一个 Pipeline：

```yml
pipeline:
  - name: "WhitespaceTokenizer"
  - name: "CountVectorsFeaturizer"
  - name: "CountVectorsFeaturizer"
    analyzer: "char_wb"
    min_ngram: 1
    max_ngram: 4
  - name: RegexEntityExtractor
    use_regexes: True
  - name: components.diet_cls.DIETClassifier
    epochs: 100
```

那么，NLU 处理的过程大概就是：

1. 通过`rasa/nlu/emulators` 预处理请求（可查看 `emulators.normalise_request_json`方法）。

2. 在 `rasa/core/processor` 中将文本信息包装到 `Message` 中。此时的 `Message` 仅包括 `text` 等基础字段

3. 通过 `rasa/nlu/tokneizers/whitespace_tokenizer` 中 `WhitespaceTokenizer.tokenize()` 往  `Message` 中添加 `text_tokens` 结果和字段。

4. 通过 `rasa/nlu/featurizers/sparse_featurizer/count_vectors_featurizer.py` 中 `CountVectorsFeaturizer.process()` ，往 `Message.features` 中添加 `features` 内容。（`Message.features` 中的所有内容仅被用于辅助 其他 NLU 环节处理，不会被当成最终结果返回）

   > 如果有多个 `featurizer`，那么他们的输出将会被统一储存在 `Message.features` 列表中。

5. 通过 `rasa/nlu/extractors/regex_entity_extractor.py` 中的 `RegexEntityExtractor.process()`，往 `Message` 中添加 `entities` 结果和字段。

6. 通过 `rasa/nlu/classifier/diet_classifier.py` 中的 `DIETClassifier.process()`，往 `Message` 中添加 `intent`, `intent_ranking`, `entities` 结果和字段。

7. 最终结果经过 `emulator.normalise_response_json` 后处理，被包装成 json 返回。

### NLU 模块思考

1. 整个 NLU 过程采用了几年前 NLU 领域特征工程大杂烩 + 基础模型训练的操作。你可以任意的添加 Features，但是 features 在最后进行意图识别，或者实体识别时候，将会以拼接的方式结合（如 `rasa.utils.tensorfloe.ConcatenateSparseDenseFeatures`），而后加上下游模型进行训练和预测。

2. 如果要自定义 Intent 和 NER 模块，只需要重新包装好 `GraphComponent`，确保在 `process()` 方法中，将 `entities`， `intent`，`intent_ranking` 添加到 Message 中即可。

3. 对于 RASA 中的 Transformer，意图识别默认使用 CLS 位置的 `hidden_state` 进行分类；实体抽取任务默认使用其他位置的 `hidden_state` 进行预测。预测的基础单位取决于 Pipeline 中的 `Tokenizer`。比如你使用了 `WhitespaceTokenizer`。假设用户输入 `say HelloWorld` ，那么大致的实体抽取流程会是：

   ```python
   tokens = "say HelloWorld".split()
   token_features = []
   for token in tokens:
       sub_token = MybertTokenzier(token)
       sub_token_feature = MyBertModel(sub_token)
       token_feature = combine(sub_token_feature) 
       token_features.append(token_feature)
   NER_result = my_ner_model(token_features)
   ```

   部分 Transformer 模型在 tokenize 之前都会进行基础 tokenize（如  `.split()`）。但对那些不进行基础 tokenize 的 Transformer 模型，则会使 X 分布偏移，导致效果受影响。

 参考代码：

[https://github.com/kevinng77/rasa_example/tree/master/examples/2_custom_clugithub.com/kevinng77/rasa_example/tree/master/examples/2_custom_clu](https://link.zhihu.com/?target=https%3A//github.com/kevinng77/rasa_example/tree/master/examples/2_custom_clu)

## 小结

不少 low code 对话系统开发平台的 NLU 模块与 RASA 几乎一致，后续的系列文章也会分析落地部署类似的 NLU 引擎，以及部署 huggingface NLU 模型实现高吞吐量方案。

印象中研究 RASA 也就一年前左右，而今 LLM 和 Agent 等 AIGC 话题的火热，使得 RASA 这样的对话框架受到的关注减少。不知多久后，RASA 会变成 NLP 的历史产物。