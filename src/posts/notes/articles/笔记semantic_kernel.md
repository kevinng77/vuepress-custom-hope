---
title: Semantic Kernel | 上手与分析
date: 2024-05-01
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- NLP
- AIGC
- LLM
- Agent
---

# Semantic Kernel

本文对 Semantic Kernel 中的 Kernel，Plugin，KernelFunction，Semantic Memory，Planner，Services，reliability 等进行概念介绍。 

## 1. Kernel

![image-20240411094303977](/assets/img/semantic_kernel/image-20240411094303977.png)

Kernel 是一个执行单元，用来执行以下内容

- 选择最佳的人工智能服务来运行提示。 
- 使用提供的 prompt 模板构建提示。（必要时候调用额外的工具和函数） 
- 将 prompt 发送给人工智能服务。 
- 接收并解析响应。
- 最后将响应返回到您的应用程序之前。

具体发送什么 prompt，如何调用外部工具，都需要再 `KernelFunction` 中实现。

kernel 的核心函数是`invoke`，用的话就是用 `kernel.invoke(functions, args)`，大致逻辑如下：

```python
results = []

for func in functions:
    while True:
        # 首先获得一些参数
		function_invoking_args = self.on_function_invoking(func.metadata, arguments) 
        
        # 查看 args 中 skip, cancle, update args 等参数是否为 True
        # 是的话采取对应措施
        
        # 调用 kernel function	
        function_result = await func.invoke(self, arguments)
        
        # 参数后处理
        function_invoked_args = self.on_function_invoked(func.metadata, arguments, function_result, exception)
        results.append(function_invoked_args.function_result)
        
        # 查看 args 中 skip, cancle, update args 等参数是否为 True
		# 是的话采取对应措施
        # 查看 is_repeat_requested 参数是否为 True
        if function_invoked_args.is_repeat_requested:
            continue
        break
return result
```

## 2. Plugin

> Plugin 由一个或者多个 KernelFunction 组。Semantic Kernel 的 Plugin 设计是参考 [OpenAi Plugin （Action）](https://platform.openai.com/docs/actions/introduction)实现的。因此 Semantic Kernel 的 Plugin 也是可以在如 ChatGPT，M365 Copilot 等平台上使用的，[参考指南](https://learn.microsoft.com/zh-cn/semantic-kernel/agents/plugins/openai-plugins)。

上文提到，Kernel 具体发送什么 prompt，如何调用外部工具，都需要再 `KernelFunction` 中实现。在 Semantic Kernel 中， `Plugin` 这个类中可以搭载多个 `KernelFunction`。

比如我们希望 GPT 能够支持数学服务，那么我们可以建立一个 `MathPlugin`，里面装载 `线性代数 Function`, `微积分 MathFunction`, `建模工具 Function` 等等。

### 代码解析

KernelPlugin 类中储存有如下信息：

```python
class KernelPlugin(KernelBaseModel):
    name: Annotated[str, StringConstraints(pattern=PLUGIN_NAME_REGEX, min_length=1)]
    description: Optional[str] = Field(default=None)
    functions: Optional[Dict[str, "KernelFunction"]] = Field(default_factory=dict)

```

初始化后，可以通过字典的方法获取 Plugin 下面储存的 function （`KernelFunction`）

即： `func = KernelPlugin['function_name']`。

## 3. KernelFunction

::: info

KernelFunction 包括 2 大类型，`KernelFunctionFromPrompt` 和 `KernelFunctionFromMethod` 。

- `KernelFunctionFromPrompt` 中会整理目前任务的参数，生成 prompt，调用 `Service` 来向大模型发送 `Completion` 请求。
- `KernelFunctionFromMethod` 中会整理 method 运行需要的参数，而后调用该 method 来获取结果。

以上两种 Function 都可以通过 `kernel.invoke()` 来调用，具体参考下文。

:::

#### 3.1 Kernel Function from Prompt

#####  **基础使用** 

```python
# https://learn.microsoft.com/zh-cn/semantic-kernel/prompts/your-first-prompt?tabs=python
import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig

kernel = sk.Kernel()
# 指明需要用的 AI 服务
kernel.add_service(
  AzureChatCompletion(
      service_id="default",
      deployment_name=deployment,
      endpoint=endpoint,
      api_key=api_key,
    api_version="2024-02-15-preview"
  )
)

prompt = "What is the intent of this request? {{$request}}"
req_settings = kernel.get_service("default").get_prompt_execution_settings_class()(service_id=service_id)
chat_prompt_template_config = PromptTemplateConfig(
    template=prompt,
    description="Chat with the assistant",
    execution_settings={service_id: req_settings},
    input_variables=[
        InputVariable(name="request", description="The user input", is_required=True),
    ],
)
# 通过 prompt template，构造 KernelFunctionFromPrompt
prompt_function = kernel.create_function_from_prompt(
    function_name="sample_zero", plugin_name="sample_plugin", prompt=prompt
)
async def main():
    request = "Hello"
    result = await kernel.invoke(prompt_function, request=request)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

##### invoke 源码逻辑分析

 `KernelFunctionFromPrompt` 中的  `_invoke_internal` 为主要入口函数。当使用  `kernel.invoke( KernelFunctionFromPrompt)` 时，主要会调用  `_invoke_internal` 。该 function 调用大致的逻辑为：

 **Step 1：首先整理 arguments，对 prompt template 填充默认值** 

提前在 function 对应的配置文件 `config.json` （或 `prompt_template_config`）中定义 prompt template，如：

```python
"What is the intent of this request? {{$request}}"
```

以及对应填充参数的内容：

```python
  "input_variables": [
    {
      "name": "request",
      "description": "Your description here",
      "default": "Hello"
    }
  ]
```

`_invoke_internal`  会对配置信息进行处理，如果 `arguments` 中缺少了 prompt template 需要的填充参数时，会使用 `default` 值进行填充。



 **Step 2：选择需要执行的 GPT Service** 

每一个 KernelFunction 都可以指定用什么 AI 模型来进行推理服务，比如我们可以定义 OpenAI GPT-4 为 `default` 模型，那么当希望某个 KernelFunction 使用 OpenAI GPT-4 进行推理式，只要设置对应的 `service_id="default"`  即可。

在对应的 `config.json` 中需要定义好 `"execution_settings"`，包括调用 GPT-4 服务时候使用的参数（如 `max_tokens` 等)，如下:

```python
  "execution_settings": {
    "default": { //此处的 default 为 service ID
      "max_tokens": 1000,
      "temperature": 0.9,
      "top_p": 0.0,
      "presence_penalty": 0.0,
      "frequency_penalty": 0.0
    }
  },
```

如果 function 对应的 service_id 没找到的话，那么 kernel 会自动选用一个 Service 来进行大模型推理。

 **Step 3：生成 prompt** 

prompt 支持 jinja template，实际的填充 prompt template，而后调用 chat 服务的位置在 `functions.kernel_function_from_prompt.KernelFunctionFromPrompt` 中

 **Step 4：大模型推理** 

将历史聊天内容以及当前 prompt 整合后，进行大模型推理，得到模型回复。Semantic kernel python 版本用的也是 openai 的 python 包来调用 Openai 的服务，因此理论上支持所有的 openai compatible api。

 **Step 5：生成 Result 并返回** 

返回的结果格式是 `FunctionResult`

```python
{
	"function": KernelFunctionMetadata,
    "value": [
        {
            "role": "",
            "content": "",
            "encoding": ...
        }
    ],
    "metadata": {
        "arguments": arguments,
        "metadata": [],
    }
}
```

::: info

但由于  `FunctionResult` 及其中的 attr 大部分被重写了 `__str__` 方法，因此在后续进行字符串转变操作过程中，只会输出 `content` 的内容，及调用 GPT 的文字回复内容。

:::

#### 3.2 KernelFunctionFromMethod

#####  **基础使用** 

```python
import semantic_kernel as sk
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel

class MyPlugin(KernelBaseModel):
    @kernel_function(description="say hello")
    def hello(self) -> str:
        return "Hello"
    
kernel = sk.Kernel()
test = kernel.import_plugin_from_object(MyPlugin(), "MyPlugin")

async def main():
    result = await kernel.invoke(test['hello'])
    print(result)
```

##### invoke 源码逻辑分析

 **Step 1：首先整理 arguments，提取 method 需要的参数** 

 **Step 2：调用 method，整理返回结果** 

因为函数调用可能返回 generator，或 Awaitable，或 asyncgen 对象，因此需要对返回对象进行处理，确保返回的是最终结果。

 **Step 3：生成 Result** 

#### 3.3 Semantic Kernel Prompt Template 的特点

1. [prompt 中可以自动调用其他的 plugin](https://learn.microsoft.com/zh-cn/semantic-kernel/prompts/calling-nested-functions)

2. [prmopt 可以通过 file 的方式来导入](https://learn.microsoft.com/zh-cn/semantic-kernel/prompts/saving-prompts-as-files?tabs=python)

示例：在 prmopt template 中，使用 KernelFunctionFromMethod：

```python

class MyPlugin(KernelBaseModel):
    @kernel_function(description="say hello")
    def get(self) -> str:
        result = "numpy"
        if random.random() > 0.5:
            result = "pandas"
        print("result is ", result)
        return result
    
kernel.import_plugin_from_object(MyPlugin(), plugin_name="GetLanguage")

# 转换这个 template 时，会自动调用 GetLanguage.get 函数
prompt = "Please generate {{$number}} sample {{GetLanguage.get}} code"
req_settings = kernel.get_service("default").get_prompt_execution_settings_class()(service_id=service_id)

chat_prompt_template_config = PromptTemplateConfig(
    template=prompt,
    description="Chat with the assistant",
    execution_settings={service_id: req_settings},
    input_variables=[
        InputVariable(name="number", description="The user input", is_required=True),
    ],
)

prompt_function = kernel.create_function_from_prompt(
    function_name="sample_zero", plugin_name="sample_plugin", prompt=prompt
)


async def main():
    result = await kernel.invoke(prompt_function, number=2)
    print(result) 
```

转换以上 template 时，会自动调用 GetLanguage.get 函数。同时支持对 function 的输入和输出结果进行数据格式验证，比如：

```python
@kernel_function(name="Add")
def add(
   self,
   number1: Annotated[float, "the first number to add"],
   number2: Annotated[float, "the second number to add"],
) -> Annotated[float, "the output is a float"]:
   return float(number1) + float(number2)
```

在使用 Annotated 之后，函数结果在填充到 prompt 中时，会自动进行类型转换。

如果 kernel function 设置了`name` ，在 prompt 中也可以直接使用 `prompt =  "{{Add '1' '2'}}"` 来调用函数名进行传参。 

## 4. Semantic Memory

如果用 SK 做一个传统的 chat 服务的话，可以用以下简单的方式实现：

```python
async def chat(input_text: str) -> None:
    print(f"User: {input_text}")
    chat_history.add_user_message(input_text)
	
    answer = await kernel.invoke(chat_function, KernelArguments(user_input=input_text, history=chat_history))

    print(f"ChatBot: {answer}")

    chat_history.add_assistant_message(str(answer))
```

其中 `chat function ` 中写好 system prompt 和其他 prompt 参数。

但当历史对话内容非常多的时候，往往会出现长度超出上下文的情况。Semantic Kernel 中有 semantic Memory 一类，可以通过 embedding 相似度来讲相关的历史对话内容，加入到最终的 prompt 当中。

具体实现方法可以参考 `core_plugins/test_memory_plugin.py`。总结来说可以分为以下几点：

1. 构造一个 plugin ，里面包含了各种 method function 用于检索 vector store 的相关接口。
2. 构造另一个 prompt function，将 method function 检索的相关信息整理拼接到最终的 prompt 里。
3. 调用  prompt function，即可实现 Semantci Memory。

## 5. Services

> 用于定义使用的 AI 模型，可以从 huggingface，GPT 等地方选取。


Services 的源代码集中在 SK 仓库的 `services` 中。

SK 支持我们在同一个 kernel 中调用不同的模型，比如 `gpt-35-turbo`，`Llama`，`GPT-4-TURBO` 等。需要用 `kernel.add_service` 添加对应的类。

```python
kernel.add_service(
  AzureChatCompletion(
      service_id=service_id,
      deployment_name=deployment,
      endpoint=endpoint,
      api_key=api_key,
    api_version="2024-02-15-preview"
  )
)
```

service 用于定义大模型服务，如 huggingface， openai 等。Function 会调用 service 中的 `complete_chat` 函数进行推理，获得回复。具体位置在 `connectors.ai.openai.services.open_ai_chat_completion_base.OpenAIChatCompletionBase` 中。 

## 6.  Planner

### 6.1 Basic Planner

basic planner 可以看做一个 `KernelFunctionFromPrompt`，我们需要：

1. 一段你的任务描述，比如“帮我写一份行业报告”。
2. 做以上任务需要用到的 function。比如你可能需要定义一些，可以从网上搜索到行业报告的函数。
3. 运行 planner，planner 会将 function 信息整合后，放到以下 prompt template 当中，生成 plan。 

```python
""" 
You are a planner for the Semantic Kernel.
Your job is to create a properly formatted JSON plan step by step, to satisfy the goal given.
Create a list of subtasks based off the [GOAL] provided.
Each subtask must be from within the [AVAILABLE FUNCTIONS] list. Do not use any functions that are not in the list.
Base your decisions on which functions to use from the description and the name of the function.
Sometimes, a function may take arguments. Provide them if necessary.
The plan should be as short as possible.
For example:

[AVAILABLE FUNCTIONS]
EmailConnector.LookupContactEmail
description: looks up the a contact and retrieves their email address
args:
- name: the name to look up

WriterPlugin.EmailTo
description: email the input text to a recipient
args:
- input: the text to email
- recipient: the recipient's email address. Multiple addresses may be included if separated by ';'.

WriterPlugin.Translate
description: translate the input to another language
args:
- input: the text to translate
- language: the language to translate to

WriterPlugin.Summarize
description: summarize input text
args:
- input: the text to summarize

FunPlugin.Joke
description: Generate a funny joke
args:
- input: the input to generate a joke about

[GOAL]
"Tell a joke about cars. Translate it to Spanish"

[OUTPUT]
    {
        "input": "cars",
        "subtasks": [
            {"function": "FunPlugin.Joke"},
            {"function": "WriterPlugin.Translate", "args": {"language": "Spanish"}}
        ]
    }

[AVAILABLE FUNCTIONS]
{{$available_functions}}

[GOAL]
{{$goal}}

[OUTPUT]
"""
```

几个细节：

- 被整合到 prompt 中的 funciton 信息有：function 名称；functino 所属 plugin 名称；function description；function 参数。（可以看做是 把 function 的 docstring 加入到了 prompt 中） 
- plan 解析：尽管在 prompt 中告诉 GPT 只返回一个 json，但是在解析过程中，semantic kernel 还是先用正则表达式来抽取出关键信息，而后在使用 json.load 加载。
- 该 planner 还是很考验 GPT 性能的，如果 function 名称匹配不上，那么 basic planner 就会报错。

### 6.2 SequentialPlanner

SequentialPlanner 与 basicPlanner 有点类似，以下是他的 prompt template  

```txt
Create an XML plan step by step, to satisfy the goal given, with the available functions.

[AVAILABLE FUNCTIONS]

{{$available_functions}}

[END AVAILABLE FUNCTIONS]

To create a plan, follow these steps:
0. The plan should be as short as possible.
1. From a <goal> create a <plan> as a series of <functions>.
2. A plan has 'INPUT' available in context variables by default.
3. Before using any function in a plan, check that it is present in the [AVAILABLE FUNCTIONS] list. If it is not, do not use it.
4. Only use functions that are required for the given goal.
5. Append an "END" XML comment at the end of the plan after the final closing </plan> tag.
6. Always output valid XML that can be parsed by an XML parser.
7. If a plan cannot be created with the [AVAILABLE FUNCTIONS], return <plan />.

All plans take the form of:
<plan>
    <!-- ... reason for taking step ... -->
    <function.{FullyQualifiedFunctionName} ... />
    <!-- ... reason for taking step ... -->
    <function.{FullyQualifiedFunctionName} ... />
    <!-- ... reason for taking step ... -->
    <function.{FullyQualifiedFunctionName} ... />
    (... etc ...)
</plan>
<!-- END -->

To call a function, follow these steps:
1. A function has one or more named parameters and a single 'output' which are all strings. Parameter values should be xml escaped.
2. To save an 'output' from a <function>, to pass into a future <function>, use <function.{FullyQualifiedFunctionName} ... setContextVariable="<UNIQUE_VARIABLE_KEY>"/>
3. To save an 'output' from a <function>, to return as part of a plan result, use <function.{FullyQualifiedFunctionName} ... appendToResult="RESULT__<UNIQUE_RESULT_KEY>"/>
4. Use a '$' to reference a context variable in a parameter, e.g. when `INPUT='world'` the parameter 'Hello $INPUT' will evaluate to `Hello world`.
5. Functions do not have access to the context variables of other functions. Do not attempt to use context variables as arrays or objects. Instead, use available functions to extract specific elements or properties from context variables.

DO NOT DO THIS, THE PARAMETER VALUE IS NOT XML ESCAPED:
<function.Plugin-Name4 input="$SOME_PREVIOUS_OUTPUT" parameter_name="some value with a <!-- 'comment' in it-->"/>

DO NOT DO THIS, THE PARAMETER VALUE IS ATTEMPTING TO USE A CONTEXT VARIABLE AS AN ARRAY/OBJECT:
<function.Plugin-CallFunction input="$OTHER_OUTPUT[1]"/>

DO NOT DO THIS, THE FUNCTION NAME IS NOT A FULLY QUALIFIED FUNCTION NAME:
<function.CallFunction input="world"/>

Here is a valid example of how to call a function "_plugin-Function_Name" with a single input and save its output:
<function._plugin-Function_Name input="this is my input" setContextVariable="SOME_KEY"/>

Here is a valid example of how to call a function "Plugin-FunctionName2" with a single input and return its output as part of the plan result:
<function.Plugin-FunctionName2 input="Hello $INPUT" appendToResult="RESULT__FINAL_ANSWER"/>

Here is a valid example of how to call a function "Name3" with multiple inputs:
<function.Plugin-Name3 input="$SOME_PREVIOUS_OUTPUT" parameter_name="some value with a &lt;!-- &apos;comment&apos; in it--&gt;"/>

Begin!

<goal>{{$input}}</goal>
```

一些细节：

- Function 部分 plan 抽取：先用 `defusedxml` 进行 XML 抽取，失败的话，用 re 先进行预处理，而后再次用 `defusedxml` 进行 XML 抽取
- 相比于 basic planner，sequence planner 提供了 `allow_missing_functions` 参数。当 function 名称解析错误时，依然能够返回 plan。但能不能执行就不一定了

### 6.3 Action Planner

大致流程为：

1. 使用以下 prompt template 生成 plan。

```python
A planner takes a list of functions, a goal, and chooses which function to use.
For each function the list includes details about the input parameters.
[START OF EXAMPLES]
{{ActionPlanner_Excluded.GoodExamples}}
{{ActionPlanner_Excluded.EdgeCaseExamples}}
[END OF EXAMPLES]
[REAL SCENARIO STARTS HERE]
- List of functions:
{{ActionPlanner_Excluded.ListOfFunctions}}
- End list of functions.
Goal: {{ $goal }}

```

其中需要填充的有几个 semantic function：

1. `List of Functions` : 将所有函数的 description，名字，输入，输出等信息拼接。示例如下：

```
// Returns the Addition result of the values provided.
math.Add
Parameter ""input"": the first number to add.
Parameter ""amount"": the second number to add.
// Subtracts value to a value.
math.Subtract
Parameter ""input"": the first number.
Parameter ""amount"": the number to subtract.
```

2. `GoodExamples`：固定的 prompt example，应该是用来控制输出样式

```
"""
[EXAMPLE]
- List of functions:
// Get the current time.
TimePlugin.Time
No parameters.
// Makes a POST request to a uri.
HttpPlugin.PostAsync
Parameter ""body"": The body of the request.
- End list of functions.
Goal: get the current time.
{""plan"":{
""rationale"": ""the list contains a function that gets the current time (now)"",
""function"": ""TimePlugin.Time""
}}
#END-OF-PLAN
"""
```

3. `EdgeCaseExamples`：固定的 prompt example，应该是用来控制输出样式

```python
'''
[EXAMPLE]
- List of functions:
// Get the current time.
TimePlugin.Time
No parameters.
// Write a file.
FileIOPlugin.WriteAsync
Parameter ""path"": Destination file. (default value: sample.txt)
Parameter ""content"": File content.
// Makes a POST request to a uri.
HttpPlugin.PostAsync
Parameter ""body"": The body of the request.
- End list of functions.
Goal: tell me a joke.
{""plan"":{
""rationale"": ""the list does not contain functions to tell jokes or something funny"",
""function"": """",
""parameters"": {
}}}
#END-OF-PLAN
'''

```

一些细节：

- 与其他 planner 相似的，虽然要求 GPT 输出 json 格式，但是在解析 json 前都会对字符串进行预处理。

### 6.4 Stepwise Planner

根据官方描述，该 planner 参考了 MRKL (Modular Reasoning, Knowledge and Language) ，该 planner 有点类似 ReACT。

使用示例：

```python

class MathPlugin:
    """
    Description: MathPlugin provides a set of functions to make Math calculations.
    """
    @kernel_function(
        description="Subtracts value to a value",
        name="Subtract",
    )
    def subtract(
        self,
        input: Annotated[float, "the first number"],
        amount: Annotated[float, "the number to subtract"],
    ) -> float:
        """
        Returns the difference of numbers provided.
        """

        return input - amount
    
    @kernel_function(
        description="Add value to a value",
        name="Add",
    )
    def add(
        self,
        input: Annotated[float, "the first number"],
        amount: Annotated[float, "the number to add"],
    ) -> float:
        """
        Returns the sum of numbers provided.
        """

        return input + amount 

    @kernel_function(
        description="Multiply value to a value",
        name="Multiply",
    )
    def multiply(
        self,
        input: Annotated[float, "the first number"],
        amount: Annotated[float, "the number to multiply"],
    ) -> float:
        """
        Returns the difference of numbers provided.

        :param initial_value_text: Initial value as string to subtract the specified amount
        :param context: Contains the context to get the numbers from
        :return: The resulting subtraction as a string
        """

        return input * amount
        
# from semantic_kernel.core_plugins import MathPlugin, TextPlugin, TimePlugin


text_plugin = kernel.import_plugin_from_object(MathPlugin(), "MathPlugin")

from semantic_kernel.planners.stepwise_planner import StepwisePlanner, StepwisePlannerConfig

planner = StepwisePlanner(kernel, StepwisePlannerConfig(max_iterations=5, min_iteration_time_ms=1000))
 
async def main(): 
    ask = "Figure out how much I have if first, my investment of 2130.23 dollars increased by 23%, and then I spend $5 on a coffee"  # noqa: E501

    plan = planner.create_plan(goal=ask)
    result = await plan.invoke(kernel)
    for index, step in enumerate(plan._steps):
        print("Step:", index)
        print("Description:", step.description)
        print("Function:", step.plugin_name + "." + step._function.name)
        print(f"  Output: {','.join(str(res) for res in result.metadata['results'])}")

```

分析：

1. 潜在 bug：

由于采用了类似 ReACT 的思路，因此在调用 openai api 时，需要设置 `stop` 参数为 `["[OBSERVATION]", "\n[THOUGHT]"]`。部分官方的源码中，`stop` 参数为 None，却设置了 `"stop_sequences": ["[OBSERVATION]", "\n[THOUGHT]"]` 导致 planner 运行不成功。

1. 初始 prompt 如下：

```python
[INSTRUCTION]
Answer the following questions as accurately as possible using the provided functions.

[AVAILABLE FUNCTIONS]
The function definitions below are in the following format:
<functionName>: <description>
  inputs:
    - <parameterName>: <parameterDescription>
    - ...

{{$function_descriptions}}
[END AVAILABLE FUNCTIONS]

[USAGE INSTRUCTIONS]
To use the functions, specify a JSON blob representing an action. The JSON blob should contain an "action" key with the name of the function to use, and an "action_variables" key with a JSON object of string values to use when calling the function.
Do not call functions directly; they must be invoked through an action.
The "action_variables" value should always include an "input" key, even if the input value is empty. Additional keys in the "action_variables" value should match the defined [PARAMETERS] of the named "action" in [AVAILABLE FUNCTIONS].
Dictionary values in "action_variables" must be strings and represent the actual values to be passed to the function.
Ensure that the $JSON_BLOB contains only a SINGLE action; do NOT return multiple actions.
IMPORTANT: Use only the available functions listed in the [AVAILABLE FUNCTIONS] section. Do not attempt to use any other functions that are not specified.

Here is an example of a valid $JSON_BLOB:
{
  "action": "pluginName-functionName",
  "action_variables": {"parameterName": "some value", ...}
}

Here is another example of a valid $JSON_BLOB:
{
  "action": "Plugin-Function",
  "action_variables": {"parameterName": "some value", ...}
}

Here is another example of a valid $JSON_BLOB:
{
  "action": "Plugin-FunctionName2",
  "action_variables": {"parameterName": "some value", ...}
}

The $JSON_BLOB must contain an "action_variables" key, with the {"parameterName": "some value", ...} value in the response.
[END USAGE INSTRUCTIONS]
[END INSTRUCTION]

[THOUGHT PROCESS]
[QUESTION]
the input question I must answer
[THOUGHT]
To solve this problem, I should carefully analyze the given question and identify the necessary steps. Any facts I discover earlier in my thought process should be repeated here to keep them readily available.
[ACTION]
{
  "action": "plugin-functionName",
  "action_variables": {"parameterName": "some value", ...}
}
[OBSERVATION]
The result of the action will be provided here.
... (These Thought/Action/Observation can repeat until the final answer is reached.)
[FINAL ANSWER]
Once I have gathered all the necessary observations and performed any required actions, I can provide the final answer in a clear and human-readable format.
[END THOUGHT PROCESS]

Let's break down the problem step by step and think about the best approach. Questions and observations should be followed by a single thought and an optional single action to take.

Begin!

[QUESTION]
{{$question}}
{{$agent_scratch_pad}}

```

后续的会根据 GPT 的回复，进行 Aciton 提取，Observation 计算以及 Thought 生成，逻辑与 ReACT 一致。参考上文中的实例代码，我们将对应的 GPT 回复都打印出来如下：

```python
>>>>>>>>>>>>>
[THOUGHT]
To solve this problem, we need to perform two actions: first, we need to calculate the increase in the investment by 23%, and then we need to subtract $5 from the result. We can use the MathPlugin-Add and MathPlugin-Multiply functions to calculate the increase, and the MathPlugin-Subtract function to subtract $5 from the result.

[ACTION]
{
  "action": "MathPlugin-Multiply",
  "action_variables": {
    "input": "2130.23",
    "amount": "0.23"
  }
}


>>>>>>>>>>>>>
Error parsing XML of prompt: mismatched tag: line 87, column 10
>>>>>>>>>>>>>
[THOUGHT]
Now that we have the increase in the investment, we can use the MathPlugin-Add function to add it to the original investment.
[ACTION]
{"action": "MathPlugin-Add", "action_variables": {"input": "2130.23", "amount": "489.9529"}}

>>>>>>>>>>>>>
Error parsing XML of prompt: mismatched tag: line 93, column 11
>>>>>>>>>>>>>
[THOUGHT]
Finally, we need to subtract $5 from the result to get the final amount.
[ACTION]
{"action": "MathPlugin-Subtract", "action_variables": {"input": "2620.1829", "amount": "5"}}

>>>>>>>>>>>>>
Error parsing XML of prompt: mismatched tag: line 99, column 11
>>>>>>>>>>>>>
[FINAL ANSWER]
After investing $2130.23 and increasing it by 23%, then spending $5 on a coffee, the final amount is $2615.1829.
>>>>>>>>>>>>>
Step: 0
Description: Execute a plan
Function: StepwisePlanner.ExecutePlan
  Output: After investing $2130.23 and increasing it by 23%, then spending $5 on a coffee, the final amount is $2615.1829.

```

关于 GPT 回复的处理，与其他 planner 类似的，当尝试使用 json 或者 xml 等工具无法正确提取时，都会使用 re 进行预处理。

## 7. 其他

[promptflowopen in new window](https://github.com/microsoft/promptflow/tree/main) 中有对 Planner 进行评测的示例





