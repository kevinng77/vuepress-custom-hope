---
title: Vue 3 基础|更新 
date: 2023-07-14
author: Kevin 吴嘉文
tag:
- webfront
category:
- 知识笔记
---

# VUE 3

VUE 3 官网：https://cn.vuejs.org/guide/quick-start.html

## 快速回顾

安装相关依赖

```
npm install -g @vue/cli vue-router
```

## vue 架构回顾

创建一个项目，执行`vue create your_project` 后生成一系列文件夹。主要修改 `src` 文件夹就行。

```python
src
├── App.vue                   
├── assets             
│   └── logo.png
├── components
│   └── HelloWorld.vue
└── main.js
```

文件回顾：

1. `main.js `用于注册 `router`, `store`,  全局`css`, 插件等，如：

```js
import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'

const app = createApp(App)
app.use(router)
   .use(Element, {
  				size: Cookies.get('size') || 'medium',
				})

app.mount('#app')
```

2. `App.vue`

该文件中主要用于配置大致架构，比如简单的使用 `<router-view />`：

::: code-tabs#shell

@tab 组合式

```vue
<template>
  <div id="app">
    <router-view />
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>
```

@tab 使用 setup

```ts
// main.js
import { createSSRApp } from "vue";
import App from "./App.vue";
export function createApp() {
  const app = createSSRApp(App);
  return {
    app,
  };
}
```

:::

3. `router.js` 中配置 router 信息，具体可查看: [vue-router](https://router.vuejs.org/zh/introduction.html)。简单的 router 模板示例：

```js
import xxx from ''
import Component2 from ''
import {createRouter}
```

## 常见问题

### API 风格

vue 分为响应式和选项式两种 API 风好，两种风格的差异主要在 `script` 部分。

::: info

当你在模板中使用了一个 ref，然后改变了这个 ref 的值时，Vue 会自动检测到这个变化，并且相应地更新 DOM。这是通过一个基于依赖追踪的响应式系统实现的。当一个组件首次渲染时，Vue 会 **追踪** 在渲染过程中使用的每一个 ref。然后，当一个 ref 被修改时，它会 **触发** 追踪它的组件的一次重新渲染。

:::

通常，响应式会结合 `<script setup>` 使用，以此来暴露状态和方法：

```vue
<script setup>
import { ref } from 'vue'

const count = ref(0)

function increment() {
  count.value++
}
</script>

<template>
  <button @click="increment">
    {{ count }}
  </button>
</template>
```

变量或对象声明：

```ts
import { ref, reactive  } from 'vue'

const count = ref(0)
// const isActive = ref(true)
count.value = 1
const state = reactive({ count ,other: 1}) // 用 reactive 追踪嵌套的变量
console.log(state.count) // 0
```

计算属性 `computed`

```ts
import { computed } from 'vue'
// 一个计算属性 ref
const publishedBooksMessage = computed(() => {
  return xxx
})
```

定义传参 `props`

::: code-tabs#shell

@tag 使用 setup

```ts
const props = defineProps<{
  title?: string
  likes?: number
}>()
console.log(props.title)
```

@tag 不用 setup

```ts
export default {
  props: {
    title: String,
    likes: Number
  }
}
```

:::

监听事件 `emit`，加入要触发 `'enlarge-text'` 事件。

```vue
<!-- BlogPost.vue -->
<script setup>
defineProps(['title'])
defineEmits(['enlarge-text'])
</script>
<template>
  <div class="blog-post">
    <h4>{{ title }}</h4>
    <button @click="$emit('enlarge-text')">Enlarge text</button>
  </div>
</template>
```

调用父组件中调用 `BlogPost` 时，可以定义 `enlarge-text` 触发时，采取的行动：

```vue
<BlogPost
  ...
  @enlarge-text="postFontSize += 0.1"
 />
```

### 条件渲染

通过 v-bind 可以动态渲染 class，来控制 styles 等。

```vue
<!-- class 绑定 -->
<div :class="{ red: isRed }"></div>
<div :class="[classA, classB]"></div>
<div :class="[classA, { classB: isB, classC: isC }]"></div>
```

`v-bind` 可以涵盖 class 部分的修改，可以使用 js 语法来实现添加或者删除 `attr`。 

```vue
<!-- 动态 attribute 名，比如复制 id="123" -->
<button :[key]="value"></button>
<!-- 在 script 部分定义 key 和 value -->

<!-- style 绑定 -->
<div :style="{ fontSize: size + 'px' }"></div>

<!-- 传递子父组件共有的 prop -->
<MyComponent v-bind="$props" />
```

### 列表渲染

```vue
<script setup>
const items = ref([{ message: 'Foo' }, { message: 'Bar' }])
</script>

<li v-for="item in items">
  {{ item.message }}
</li>
```

或渲染对象：

```vue
<script setup>
const myObject = reactive({
  title: 'How to do lists in Vue',
  author: 'Jane Doe',
  publishedAt: '2016-04-10'
})
</script>
<li v-for="(value, key) in myObject">
  {{ key }}: {{ value }}
</li>
```

### 内置功能  Teleport

`Transition` 和 `Teleport` 可以用于实现一些全屏窗口切换的功能，比如搜索栏，状态提示栏，广告等。[官方指南](https://cn.vuejs.org/guide/built-ins/teleport.html#basic-usage)

::: details

```vue
<button @click="open = true">Open Modal</button>

<Teleport to="body">
    <transition>
      <div v-if="open" class="modal">
        <p>Hello from the modal!</p>
        <button @click="open = false">Close</button>
      </div>
    </transition>
</Teleport>
<style>
</style>
```

:::

### 动态渲染 component 

如果某个组件的渲染时间非常长，那么可以使用动态 [component](https://cn.vuejs.org/api/built-in-special-elements.html#component) 和 [keepAlive](https://cn.vuejs.org/guide/built-ins/keep-alive.html#include-exclude) 优化

::: details

```vue
<script>
import Foo from './Foo.vue'
import Bar from './Bar.vue'

export default {
  components: { Foo, Bar },
  data() {
    return {
      view: 'Foo'
    }
  }
}
</script>

<template>
    <KeepAlive include="a,b">
      <component :is="view" />
    </KeepAlive>
</template>
```

:::

### 页面资源和 HOOK 

整个 DOM 渲染的周期（渲染前，后；更新前，后）都可以添加钩子。[官方链接](https://cn.vuejs.org/api/options-lifecycle.html)。常用的有 `mounted`, `updated` 等。

::: code-tabs#shell

@tab 组合式

```js
export default {
  mounted() {
    console.log(`the component is now mounted.`)
    // this.some_methods()
  }
}
```

@tab 响应式

```ts
import { onMounted, ref } from 'vue'

mounted(()=>{
     console.log(`the component is now mounted.`)
    // some_methods()
 })
```

:::





### `axios` 请求发送

可以将 axios 包装成一个方法，而后通过 `http` 统一发送各种请求。

::: details

`setup_axios.js`

```js
import axios from 'axios'

const instance = axios.create({
	baseURL: 'xxx',
    timeout:1000,
    headers: {'X-ucsom-Header': 'foobar'}
})
class RequestHandler {
    getAll(input) {
        return instance.get(`/sub_path/${input}`)
    }
    
    postMethod() {
        return instance.post(`/sub_path`.{your_json:your_json})
	}
}

export default new RequestHandler();
```

在 vue 组件中调用：

```js
import RequestHandler from 'setup_axios.js'

some_method (){
    RequestHandler.getAll(this.input).then(res=>{this.data = res.data}).catch()
}
```

:::

### 元素监听

可以通过 watch 来监听某个定义好的参数. [官方指南](https://cn.vuejs.org/guide/essentials/watchers.html#basic-example)。但监听器默认监听的是指针变动情况，如果需要监听嵌套的内容，需要加上 `deep:true`

::: code-tabs#shell

@tab 组合式

```js
export default {
  data() {
    return {
      question: '',
      answer: 'Questions usually contain a question mark. ;-)'
    }
  },
  watch: {
    // 每当 question 改变时，这个函数就会执行
    question {
      handler(newQuestion, oldQuestion){
      if (newQuestion.includes('?')) {
        this.getAnswer()
      }
},
    deep: true
    }
  }
}
```

@tab 响应式

```vue
<script setup>
import { ref, watch } from 'vue'

const question = ref('')
const answer = ref('Questions usually contain a question mark. ;-)')

// 可以直接侦听一个 ref
watch(question, async (newQuestion, oldQuestion) => {
  if (newQuestion.indexOf('?') > -1) {
    answer.value = 'Thinking...'
    try {
      const res = await fetch('https://yesno.wtf/api')
      answer.value = (await res.json()).answer
    } catch (error) {
      answer.value = 'Error! Could not reach the API. ' + error
    }
  }
})
</script>
<template>
  <p>
    Ask a yes/no question:
    <input v-model="question" />
  </p>
  <p>{{ answer }}</p>
</template>
```

可以通过 js 来控制监听的开始和结束

```js
export default {
  created() {
    this.$watch('question', (newQuestion) => {
      // ...
    })
  }
}

const unwatch = this.$watch('foo', callback)

// ...当该侦听器不再需要时
unwatch()
```

### 模板引用

`this.$refs.ref_name` 能够调用 DOM 中的模板、元素。（在 `mounted` 之后 ref 才会生效）。`this.$ref` 可以获得一个对应模板的实例。可以使用对应的如 `.focus()` 等功能。

```vue
<script setup>
import { ref, onMounted } from 'vue'

const list = ref([
  /* ... */
])
// 声明一个 ref 来存放该元素的引用
// 必须和模板里的 ref 同名
const itemRefs = ref([])

onMounted(() => console.log(itemRefs.value))
</script>

<template>
  <ul>
    <li v-for="item in list" ref="itemRefs">
      {{ item }}
    </li>
  </ul>
</template>
```











