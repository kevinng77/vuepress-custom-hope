---
title: Vue 及 Vuepress 基础
date: 2023-01-20
author: Kevin 吴嘉文
tag:
- webfront
category:
- 知识笔记
---

## Vue 基础

VUE [官网 ](https://v2.cn.vuejs.org/index.html) 

Vue.js 的核心是一个允许采用简洁的模板语法来声明式地将数据渲染进 DOM 的系统

<!--more-->

### 快速上手

完整示例如下：

```html
<html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/vue@2/dist/vue.js"></script>

    </head>
    <body>
        <div id="app">
            <!-- 需要在下面中声明 new Vue, 对应的数据需要在
            对应的 Vue 示例下进行储存 -->
            在 id="app" 层下可以直接使用 message 变量：
            <br>
            {{message}}<br>
            <span title="this is a title" >
                鼠标悬停查看<br>
            </span>
            <span v-bind:title="message" >
                鼠标悬停查看<br>
            </span>
            <a v-bind:href="message" >
                自定义链接<br>
            </a>
            <!-- 使用 v-model 支持改变页 vue 中的 data，
                他的数据来源是该标签下的 value。-->
            <textarea v-model="message"> v-model 数据来源 </textarea>
            <br>
            <input type="radio" name="sex" value=true v-model="view">true
            <input type="radio" name="sex" value="" v-model="view">false
            <ol>
                <!-- 创建一个 todo-item 组件的实例 -->
                <div v-if="view">
                <todo-item 
                    v-for="item in mylist"
                    v-bind:todo="item"
                    v-bind:key="item.id"></todo-item>
                </div>
            </ol>
            {{message}}
        </div>


        <script>
            Vue.component('todo-item', {
                props: ['todo'],
                template: '<li>{{todo}}</li>'
                });

            var app = new Vue({
            el: '#app',
            data: {
                message: 'vue 中的 data',
                mylist:["123","234",2345446,],
                view: false
            }
            })
        </script>

    </body>
</html>
```

其中我们可以通过  axios 来加载外部的数据

```html
<html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/vue@2/dist/vue.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>

    </head>
    <body>
        <!-- 钩子函数 https://v2.cn.vuejs.org/v2/guide/instance.html -->
        <div id="app">
            {{info.message}}
        </div>
        <script>
            var app = new Vue({
            el: '#app',
            data(){
                return {
                    info:{
                        "message":null
                    }
                }
            },
            mounted(){
                axios.get('data.json').then(response=>(
                    this.info=response.data
                ));
            }
            })
        </script>

    </body>
</html>
```

Vue 中的变量引用比较灵活，支持 `{{}}` 符号内使用表达式。 对于标签中的值，可以直接使用 `"'var' + 1"` 等进行操作

```html
{{ number + 1 }}

{{ ok ? 'YES' : 'NO' }}

{{ message.split('').reverse().join('') }}

<div v-bind:id="'list-' + id"></div>
```

#### v-model

[常与选择框一起用](https://v2.cn.vuejs.org/v2/guide/forms.html) 如

```html
<input type="checkbox" id="jack" value="Jack" v-model="checkedNames">
<label for="jack">Jack</label>
<input type="checkbox" id="john" value="John" v-model="checkedNames">
<label for="john">John</label>
<input type="checkbox" id="mike" value="Mike" v-model="checkedNames">
<label for="mike">Mike</label>
<br>
<span>Checked names: {{ checkedNames }}</span>
```

```js
new Vue({
  el: '...',
  data: {
    checkedNames: []
  }
})
```

在默认情况下，`v-model` 在每次 `input` 事件触发后将输入框的值与数据进行同步 (除了[上述](https://v2.cn.vuejs.org/v2/guide/forms.html#vmodel-ime-tip)输入法组合文字时)。你可以添加 `lazy` 修饰符，从而转为在 `change` 事件_之后_进行同步：

```html
<!-- 在“change”时而非“input”时更新 -->
<input v-model.lazy="msg">

<!-- 输入转数字 -->
<input v-model.number="age" type="number">
```



 #### v-bind

```html
<!-- 完整语法 -->
<a v-bind:href="url">...</a>

<!-- 缩写 -->
<a :href="url">...</a>

<!-- 动态参数的缩写 (2.6.0+) -->
<a :[key]="url"> ... </a>
```

除此外，我们还可以绑定 class 或者 style，如 [link](https://v2.cn.vuejs.org/v2/guide/class-and-style.html)：

```html
<div v-bind:style="styleObject"></div>
```

```js
data: {
  styleObject: {
    color: 'red',
    fontSize: '13px'
  }
}
```

对于覆盖型的 style，可以:

```html
<div v-bind:style="[baseStyles, overridingStyles]"></div>
```

同样的，这种才做对于 class 也成立：

```html
<div v-bind:class="[isActive ? activeClass : '', errorClass]"></div>
```

#### V-ON

可以用 `v-on` 指令监听 DOM 事件，并在触发时运行一些 JavaScript 代码。

```html
<!-- 完整语法 -->
<a v-on:click="doSomething">...</a>

<!-- 缩写 -->
<a @click="doSomething">...</a>

<!-- 动态参数的缩写 (2.6.0+) -->
<a @[event]="doSomething"> ... </a>
```

使用的方法与 on-click 类似，能够对 vue 示例中的值进行更改等

```html
<div id="example-3">
  <button v-on:click="say('hi')">Say hi</button>
  <button v-on:click="say('what')">Say what</button>
</div>
```

```js
new Vue({
  el: '#example-3',
  methods: {
    say: function (message) {
      alert(message)
    }
  }
})
```

Vue.js 为 `v-on` 提供了 **事件修饰符** 。之前提过，修饰符是由点开头的指令后缀来表示的。

```html
<!-- 阻止单击事件继续传播 -->
<a v-on:click.stop="doThis"></a>

<!-- 提交事件不再重载页面 -->
<form v-on:submit.prevent="onSubmit"></form>

<!-- 修饰符可以串联 -->
<a v-on:click.stop.prevent="doThat"></a>

<!-- 只有修饰符 -->
<form v-on:submit.prevent></form>

<!-- 添加事件监听器时使用事件捕获模式 -->
<!-- 即内部元素触发的事件先在此处理，然后才交由内部元素进行处理 -->
<div v-on:click.capture="doThis">...</div>

<!-- 只当在 event.target 是当前元素自身时触发处理函数 -->
<!-- 即事件不是从内部元素触发的 -->
<div v-on:click.self="doThat">...</div>

<!-- 点击事件将只会触发一次 -->
<a v-on:click.once="doThis"></a>
```

同样 vue 有对键盘事件监听的特殊写法：

```html
<!-- 只有在 `key` 是 `Enter` 时调用 `vm.submit()` -->
<input v-on:keyup.enter="submit">
```



#### v-if

```html
<div v-if="type === 'A'">
  A
</div>
<div v-else-if="type === 'B'">
  B
</div>
<div v-else-if="type === 'C'">
  C
</div>
<div v-else>
  Not A/B/C
</div>
```

相比之下，`v-show` 就简单得多——不管初始条件是什么，元素总是会被渲染，并且只是简单地基于 CSS 进行切换。

一般来说，`v-if` 有更高的切换开销，而 `v-show` 有更高的初始渲染开销。因此，如果需要非常频繁地切换，则使用 `v-show` 较好；如果在运行时条件很少改变，则使用 `v-if` 较好。

```html
<h1 v-show="ok">Hello!</h1>
```

 **不推荐** 同时使用 `v-if` 和 `v-for`。

#### v-for

[link](https://v2.cn.vuejs.org/v2/guide/list.html)

数组内容更改时，如被 `push()`，`pop()`等。

可以用以下方式进行筛选显示：

```html
<li v-for="n in evenNumbers">{{ n }}</li>
```

```js
data: {
  numbers: [ 1, 2, 3, 4, 5 ]
},
computed: {
  evenNumbers: function () {
    return this.numbers.filter(function (number) {
      return number % 2 === 0
    })
  }
}
```



#### computed 属性

[link](https://v2.cn.vuejs.org/v2/guide/computed.html)

我们可以将同一函数定义为一个方法而不是一个计算属性。两种方式的最终结果确实是完全相同的。然而，不同的是 **计算属性是基于它们的响应式依赖进行缓存的** 。只在相关响应式依赖发生改变时它们才会重新求值。这就意味着只要 `message` 还没有发生改变，多次访问 `reversedMessage` 计算属性会立即返回之前的计算结果，而不必再次执行函数。

```html
<div id="app">
    <p>{{'current time ' + gettime()}}</p>
    <p>{{'current time ' + computed_gettime}}</p>

</div>
<script>
    var app = new Vue({
        el: '#app',
        methods: {
            gettime: function (){
                return Date.now();
            }
        },
        computed: {
            // 不能与 methods 重名, 将计算结果进行缓存
            // this 指向 vm 示例(此处为 app)
            computed_gettime: function (){
                return Date.now();
            }
        }
    })
</script>
```

可以设置 getter 和 setter 来对属性进行更新：

```js
computed: {
  fullName: {
    // getter
    get: function () {
      return this.firstName + ' ' + this.lastName
    },
    // setter
    set: function (newValue) {
      var names = newValue.split(' ')
      this.firstName = names[0]
      this.lastName = names[names.length - 1]
    }
  }
}
```

再运行 `vm.fullName = 'John Doe'` 时，setter 会被调用，`vm.firstName` 和 `vm.lastName` 也会相应地被更新。

在这个示例中，使用 `watch` 选项允许我们执行异步操作 (访问一个 API)，限制我们执行该操作的频率，并在我们得到最终结果前，设置中间状态。这些都是计算属性无法做到的。

### 组件

```html
<div id="components-demo">
  <button-counter></button-counter>
</div>
```

```js
// 定义一个名为 button-counter 的新组件
Vue.component('button-counter', {
  data: function () {
    return {
      count: 0
    }
  },
  template: '<button v-on:click="count++">You clicked me {{ count }} times.</button>'
})
```

组件中 data 必须给函数，否则所有组件将共享数据。

#### prop

组件通过 props 定义接受  `title` 参数：

```js
Vue.component('blog-post', {
  props: ['title'],
  template: '<h3>{{ title }}</h3>'
})
```

html 中传递参数：

```html
<blog-post title="My journey with Vue"></blog-post>
<blog-post v-bind:title="post.title"></blog-post>
<blog-post title="Why Vue is so fun"></blog-post>
```

当然也可以传入其他类型的参数，如对象，数组等。[参考官方文档](https://v2.cn.vuejs.org/v2/guide/components-props.html)

```html
<!-- 即便对象是静态的，我们仍然需要 `v-bind` 来告诉 Vue -->
<!-- 这是一个 JavaScript 表达式而不是一个字符串。-->
<blog-post
  v-bind:author="{
    name: 'Veronica',
    company: 'Veridian Dynamics'
  }"
></blog-post>

<!-- 用一个变量进行动态赋值。-->
<blog-post v-bind:author="post.author"></blog-post>
```



#### 自定义事件

再模板中，模板可以自定义事件，更改模板内的局部变量。并且 `v-on` 事件监听器在 DOM 模板中会被自动转换为全小写 

```html
<div id="app">

    <div v-bind:style="{fontSize: myfontsize + 'px'}" > 
        <!-- 通过 event 传递 enlarge-text 事件参数 -->
        <blog-post v-bind:post="post" 
                   v-on:enlarge-text="myfontsize += $event">
        </blog-post>
    </div>
</div>

<script>
    Vue.component('blog-post', {
        props: ['post'],
        template: `
                        <div class="blog-post">
                        <h3>{{ post.title }}</h3>
                        <button v-on:click="$emit('enlarge-text', 4)">
                            Enlarge text
    </button>
                        <div v-html="post.content"></div>
    </div>
                    `
    })

    var app = new Vue({
        el: '#app',
        data:{
            post: {title:"this is title",
                   content:"<h2>this is content</h2>"
                  },
            myfontsize: 20
        },
        methods: {
        },

    })
</script>
```

其实 v-model 事件与自定义事件的 `$event` 类似：

```html
{{searchText}}
<custom-input
  v-bind:value="searchText"
  v-on:input="searchText = $event"
></custom-input>
```

```js
Vue.component('custom-input', {
  props: ['value'],
  template: `
    <input
      v-bind:value="value"
      v-on:input="$emit('input', $event.target.value)"
    >
  `
})
```

#### slot

组件中可以使用插槽 [link](https://v2.cn.vuejs.org/v2/guide/components-slots.html)

```js
Vue.component('alert-box', {
  template: `
    <div class="demo-alert-box">
      <strong>Error!</strong>
      <slot></slot>
    </div>
  `
})
```

slot 一般用于传递文本内容。并将其放在 `<slot></slot>` 标签处。

```html
<alert-box>
  Something bad happened.
</alert-box>
```

## VUE CLI

需要 nodejs, git

[官网 ](https://cli.vuejs.org/zh/#%E8%B5%B7%E6%AD%A5) [指南](https://cli.vuejs.org/zh/guide/creating-a-project.html#%E6%8B%89%E5%8F%96-2-x-%E6%A8%A1%E6%9D%BF-%E6%97%A7%E7%89%88%E6%9C%AC)

```shell
npm install -g @vue/cli

# 创建项目
vue create my-project
```





```sh
npm install -g @vue/cli-init
# `vue init` 的运行效果将会跟 `vue-cli@2.x` 相同
vue init webpack my-project
```

再 my-project 下运行，验证环境安装成功。

```sh
npm run dev
```

观察 my-project 项目下的文件架构布局，了解 vue 项目的设计方式。

![img](/assets/img/vue/image-20221230145444706.png "title" =200x)

#### export vue 模块

在 `helloword.vue` 中， export `HelloWorld` 模块

```js
export default {
  name: 'HelloWorld',
  data () {
    return {
      msg: 'Welcome to Your Vue.js App'
    }
  }
}
```

在其他文件加中导入并使用：

```js
import HelloWorld from '@/components/HelloWorld'

```

## Vue router

[指南](https://router.vuejs.org/zh/guide/) [官网](https://router.vuejs.org/)

```sh
npm install vue-router@4
```

router 的作用相当于在点击对应的 `router-link` 后，`<router-view>` 会显示对应的模块内容。

```html
<div id="app">
  <h1>Hello App!</h1>
  <p>
    <!--使用 router-link 组件进行导航 -->
    <!--通过传递 `to` 来指定链接 -->
    <!--`<router-link>` 将呈现一个带有正确 `href` 属性的 `<a>` 标签-->
    <router-link to="/">Go to Home</router-link>
    <router-link to="/about">Go to About</router-link>
  </p>
  <!-- 路由出口 -->
  <!-- 路由匹配到的组件将渲染在这里 -->
  <router-view></router-view>
</div>
```

在 `router` 文件夹中建立 `index.js` ，而后在其中配置你期望的 `router-link` 目的地与对应的组件 

```js
import Vue from 'vue'
import Router from 'vue-router'
import HelloWorld from '@/components/HelloWorld'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'HelloWorld',
      component: HelloWorld
    }
  ]
})
```

在 `main.js` 中配置好 Vue 下对应的 `router`

```js
import Vue from 'vue'
import App from './App'
import router from './router'

Vue.config.productionTip = false

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})
```

### 嵌套路由

routes 中定义：

```js
    {
      path: '/about',
      name: 'about',
      component: about,
      children: [
        {path: '/github',component: github},
        {path: '/blog',component: blog},
      ]
    }
```

 **注意，以 `/` 开头的嵌套路径将被视为根路径。这允许你利用组件嵌套，而不必使用嵌套的 URL。**  如果没有定义 `/`，则不会显示。

在模块 `about.vue` 中使用：

```js
<router-link to="/github">github</router-link>
<router-link to="/blog">blog</router-link>
<router-view />
```

### 正则匹配

在 router 下

```js
    {
      path: '*',
      component: yournotfoundcomponent,

    }
```

## Axios

Axios [官网](https://axios-http.com/docs/intro)

```shell
npm install axios --save
```

## UI 仓库

### element plus

[官网](https://element-plus.gitee.io/zh-CN/guide/quickstart.html#%E5%AE%8C%E6%95%B4%E5%BC%95%E5%85%A5)

```shell
npm i element-plus -S
```

### element ui

[element ui 官网](https://element.eleme.cn/#/zh-CN), [vue cli](https://github.com/ElementUI/vue-cli-plugin-element)，[github](https://github.com/ElemeFE/element)

在创建的 cli 项目模板下执行

```sh
npm install element-ui -S
npm install -D unplugin-vue-components unplugin-auto-import

```

按需自动导入 element 插件

```
// vite.config.ts
import { defineConfig } from 'vite'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

export default defineConfig({
  // ...
  plugins: [
    // ...
    AutoImport({
      resolvers: [ElementPlusResolver()],
    }),
    Components({
      resolvers: [ElementPlusResolver()],
    }),
  ],
})
```

默认情况下 elementui 会选择全部导入插件里面的元素，可以选择仅仅导入你需要的部件，来减少无用的 js 加载。这种方式也相当于 **局部组件注册** 。

```js
import Vue from 'vue';
import { Button, Select } from 'element-ui';
import App from './App.vue';

Vue.component(Button.name, Button);
Vue.component(Select.name, Select);
/* 或写为
 * Vue.use(Button)
 * Vue.use(Select)
 */

new Vue({
  el: '#app',
  render: h => h(App)
});
```

安装 element-ui 后遇到 `Cannot find module ‘core-js/library/fn/object/assign`：需要

```
npm install async-validator@1.11.5
```

#### 通过 flask 保存 upload 图片

flask 中填写 api：

```python
@app.route('/uploadimg',methods=['POST'])
def editorData():
    img = request.files.get('file')
    img.save(img.filename)
```

而后 element ui 的 upload 插件中 action 换为该 api 地址，如 `localhost:5000/uploadimg`

```html
<el-upload
           action="http://localhost:5000/"
           ref="upload"
           list-type="picture"
           :data="{data:'your json data'}"
           :limit="1"
           :on-exceed="handleExceed"
           :auto-upload="false"
           :file-list="fileList"
           :on-change="handleChange"
           :on-preview="handlePreview">
    <el-button slot="trigger" size="small" type="primary">选取文件</el-button>
    <el-button style="margin-left: 10px;" size="small" type="success" @click="submitUpload">上传并预测</el-button>
</el-upload>
```

注意如果 `:auto-upload="false"` 需要设置 `list-type="picture"`，否则上传图片前是无法从 `file.url` 获取到图片的 blob 链接的。

定义 `:data` 后，上传是会一同通过 form 的方式发送。

#### flask 返回图片到 element ui

```python
with open(img, 'rb') as img_f:
    img_stream = img_f.read()
    img_stream = str("data:;base64," + str(base64.b64encode(img_stream).decode('utf-8')))
    return img_stream
```

而后在前端中，接收 base64 并作为图片 url 即可。

```js
handleSuccess(response){
    this.loading = false;
    this.dialogImageUrl = response;
 },
```

### vuepress

通过以下方式开始 vuepress，下载官方提供的通用模板。

```shell
npx create-vuepress-site [optionalDirectoryName]
cd docs
npm install
npm run dev
```

[推荐主题](https://theme-hope.vuejs.press/zh/)

#### vuepress 中的特殊 markdown

### 文章永久链接

```js
// .vuepress/config.js
module.exports = {
  permalink: "/:year/:month/:day/:slug"
};
```

能够让文章生成一个永久的 URL，防止出现分享链接时候页面丢失情况。

### 插件

## vuepress hope

[vuepress hope](https://theme-hope.vuejs.press/zh/)

缺少对应包时，自己安装即可。 **遇到问题一定要多看官方文档！！！！**  多查看 [常见文档错误](https://vuepress-theme-hope.github.io/v2/zh/faq/common-error.html#vue-warn-failed-to-resolve-component-xxx)

```sh
pnpm add -D element-plus
pnpm add -D axios
pnpm add -D vuepress-plugin-search-pro
pnpm add -D @vuepress/utils@2.0.0-beta.61
pnpm add -D vuepress-shared
```

所有 `@vuepress` 包的版本必须与 vuepress 版本一直

### 配置技巧

#### 搜索功能

```shell
npm i -D vuepress-plugin-search-pro
```

#### 自定义页面

在 yaml front master 中定义好你想引用的自定义 layout，同时在 `.vuepress/components/SpecialLayout.vue` 中写好你的自定义布局。

```md
---
layout: SpecialLayout
pageClass: custom-page-class
---
```

`.vuepress/styles/index.styl` 中别写对应页面的 CSS

#### 自定义主页

所有的组件都可以被替换，比如要用自己写的 `HomePage` 替换主页的组件，那么则：

```js
// .vuepress/config.js
import { getDirname, path } from "@vuepress/utils";
import { hopeTheme } from "vuepress-theme-hope";

const __dirname = getDirname(import.meta.url);

export default {
  theme: hopeTheme({
    // 主题选项
    // ...
  }),

  alias: {
    // 你可以在这里将别名定向到自己的组件
    // 比如这里我们将主题的主页组件改为用户 .vuepress/components 下的 HomePage.vue
    "@theme-hope/components/HomePage": path.resolve(
      __dirname,
      "./components/HomePage.vue"
    ),
  },
};
```

对于 `vuepress-theme-hope` 下面的 layout，大部分通过 `.js` 定义。如 `BlogHome` 该 layout 则定义在了 `node_module/vuepress-theme-hope/lib/client/components` 下面。如要根据该主题修改的话，可以在 `.vuepress/components` 下写好对应 layout 的 js 文件如:

```js
// .vuepress/components/myBlogHome.js 中自定义你的 layout
import { defineComponent, h } from "vue";
import MarkdownContent from "@theme-hope/components/MarkdownContent";
import DropTransition from "@theme-hope/components/transitions/DropTransition";
import ArticleList from "@theme-hope/modules/blog/components/ArticleList";
import BlogHero from "@theme-hope/modules/blog/components/BlogHero";
import InfoPanel from "@theme-hope/modules/blog/components/InfoPanel";
import ProjectPanel from "@theme-hope/modules/blog/components/ProjectPanel";
import { useArticles } from "@theme-hope/modules/blog/composables/index";
// import "../styles/home.scss";
export default defineComponent({
    name: "NewBlogHome",
    setup() {
        const articles = useArticles();
        return () => h("div", { class: "page blog" }, [
            h(BlogHero),
            h("div", { class: "blog-page-wrapper" }, [
                h("main", { class: "blog-home", id: "main-content" }, [
                    // h(DropTransition, { appear: true, delay: 0.16 }, () => h(ProjectPanel)),
                    h(DropTransition, { appear: true, delay: 0.24 }, () => h(ArticleList, { items: articles.value.items })),
                ]),
                h(DropTransition, { appear: true, delay: 0.16 }, () => h(InfoPanel)),
            ]),
            // h(DropTransition, { appear: true, delay: 0.28 }, () => h(MarkdownContent)),
        ]);
    },
});
```

然后再 `.vuepress/config.ts` 中覆盖博客 layout 即可：

```ts
  alias: {
    "@MyComponent": path.resolve(__dirname, "./components/MyComponent.vue"),
    "@theme-hope/components/HomePage": path.resolve(
      __dirname,
      "./components/myBlogHome.js"
    ),
  },
```

#### 自定义分类页面

分类页面定义在 `BlogCategory.js` 中

#### 自定义布局

```js
import Changelog from "./layouts/Changelog.vue";
import Layout from "./layouts/Layout.vue";

export default defineClientConfig({
  // 你可以在这里覆盖或新增布局
  layouts: {
    // 比如这里我们将 vuepress-theme-hope 的默认布局改为自己主题下的 layouts/Layout.vue
    Layout,
    // 同时我们新增了一个 Changelog 布局
    Changelog,
  },
});
```



[指南](https://vuepress-theme-hope.github.io/v2/zh/cookbook/advanced/extend.html)

#### 文章配置

MD 文件都必须配置 yaml 文件头 如：

```yaml
title: Disabling layout and features
icon: config
order: 3
category:
  - Guide
tag:
  - your_tags
author: [kevin, other author]
navbar: true
sidebar: heading   # 仅显示当前文章的 toc
headerDepth: 2     # toc 中最大显示的深度
layout: SpecialLayout   # 自定义布局
breadcrumb: false   # 路径导航功能
pageInfo: false     # 在文章标题下显示文章更新日期、作者等信息
contributors: false
editLink: false
lastUpdated: false
prev: false
next: false
comment: false
footer: false     # 默认启动，添加备案信息

backtotop: false
```

#### icon

[主题中精选的 icon](https://vuepress-theme-hope.github.io/v2/zh/guide/interface/icon.html#%E4%BD%BF%E7%94%A8-fontawesome)



### 写作技巧

#### 使用组件

[详细组件查看](https://vuepress-theme-hope.github.io/v2/zh/guide/markdown/components.html#pdf)

以下列举部分组件：

 **PDF 阅读器** 

```
<PDF url="/assets/sample.pdf" />
```

 **bilibili** 

```
<BiliBili bvid="BV1kt411o7C3" low-quality no-danmaku />
```

 **youtube** 

```
<YouTube id="0JJPfz5dg20" />
```

 **audio player** 

```
<AudioPlayer
  src="/assets/assets/sample.mp3"
  title="A Sample Audio"
  poster="/logo.svg"
/>
```

#### 使用容器

比如使用一个 details 容器，只需要：

```
::: details
详情容器
:::
```

除此外，还支持 warning, tip, info, note, danger。

#### 图片编写

博客中的图片需要以 `![title](path)` 的形式呈现，才能够出发 `imglazyload`。使用 `![Alt](/example.jpg "图片标题" =200x300)` 时， 图片将会被解析成 `<img src="/example.jpg" title="图片标题" width="200" height="300" />` 如果 写 `=200x` 那么高度就不会被定义

 **推荐图片编写方式：** 

```md
![Alt](/example.jpg "图片标题" =200x300)
```

#### 代码编写

代码可以使用这种方式来高亮其中的某些行：

````
```ts {1,6-8}
import type { UserConfig } from "@vuepress/cli";
import { defaultTheme } from "@vuepress/theme-default";

export const config: UserConfig = {
  title: "你好， VuePress",

  theme: defaultTheme({
    logo: "https://vuejs.org/images/logo.png",
  }),
};
```
````

文章目录

博客的文件名需要包含中文符号，才能够触发 vuepress-active-heading 下的部分设置。



#### 文章解析问题

数学公式存在以下错误，会导致文章解析出错：

+ `\alpha` 单独使用 `{}` ，如 `{{\alpha}_t}`
+ `\text{ 中包含了下划线: _}`

#### 使用模板

可以在 `.vuepress/Component` 下面添加模板，该文件下面的模板会被自动注册。注意： 这里的模板必须只能有一个单独的标签：

```vue
<template>
    <p>正确的写法，多个标签会发生错误</p> 
    <p></p>
</template>

<template>
	<div> 正确的写法，必须用一个标签包含起来
        <p></p> 
        <p></p>
    </div>    
</template>
```

在 markdown 中可以以 vue 的方式直接使用模板：

```md
<MyComponent />
```

对于 vuepress-theme-hops，需要在 client 中注册组件：

```ts
export default defineClientConfig({
  enhance: ({ app, router, siteData }) => {
    app.component("ObjectDetection", ObjectDetection);
    app.use(ElementPlus);
;
  },
});
```

并且在 config 中配置好路径：

```js
  alias: {
    "@ObjectDetection": path.resolve(__dirname, "./components/ai/ObjectDetection.vue"),
  },
```

### 部署

[部署官方文档](https://v2.vuepress.vuejs.org/zh/guide/deployment.html#github-pages)

`pnpm run docs:build` 之后，将所有 `.vuepress/dist` 下面的内容上传到服务器。

可以使用 `github.io` 。参考：[github page](https://docs.github.com/en/pages/quickstart)

如果是部署到自己的服务器，把服务器地址指向 dist 文件目录即可，比如 nginx：

```conf
server {
  listen 80;
  listen [::]:80;
  root /home/blog/dist;
  server_name wujiawen.xyz;
  client_max_body_size 1024m;
  location / {

  }
```

dev 时候的加载速度是远慢于 build 之后部署的速度的。因此千万不要用 `npm run dev` 去跑一个服务器。

###  increase-memory-limit

项目较大 build 报错：

```
npm install -g increase-memory-limit
```

每次 `pnpm run docs:build` 之前，都运行一次 ` increase-memory-limit`

## vue ts

typescript 入门教程 [link](http://ts.xcatliu.com/)

传入参数

```vue
<template>
  <div :class="msg">
    231312312
    <h1>{{ msg }}</h1>
    </div>
</template>
<script setup lang="ts">
import { ref } from 'vue'
let msg=ref('');
msg.value = "this is msg";
</script>
```

定义语法

```
const a= ref(0)
const b = ref(null)
const c = ref({})
const d = ref(false)
const e = ref([])
```

