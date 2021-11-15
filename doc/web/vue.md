创建项目：3步

```
vue init webpack demo
```

- Vue build ==> 打包方式，回车即可；
- Install vue-router ==> 是否要安装 vue-router，项目中肯定要使用到 所以Y 回车；
- Use ESLint to lint your code ==> 是否需要 js 语法检测 目前我们不需要 所以 n 回车；
- Set up unit tests ==> 是否安装 单元测试工具 目前我们不需要 所以 n 回车；
- Setup e2e tests with Nightwatch ==> 是否需要 端到端测试工具 目前我们不需要 所以 n 回车；

```
npm i # 安装依赖
npm run dev # 启动项目
```

## 指令：

### v-html

将内容解析为html

```html
<p>Using mustaches: {{ rawHtml }}</p>
<p>Using v-html directive: <span v-html="rawHtml"></span></p>  #使用v-html将值作为html来渲染在页面上
```

```html
const RenderHtmlApp = {
  data() {
    return {
      rawHtml: '<span style="color: red">This should be red.</span>'
    }
  }
}
```

### v-bind

将操作html的属性：[link](https://v3.cn.vuejs.org/guide/template-syntax.html#attribute)  它可以缩写成 :

```html
<div v-bind:id="dynamicId"></div>
```

### v-if 

当作条件判断，false将不会渲染该元素

```html
<p v-if="seen">现在你看到我了</p>
```

### v-on 

监听dom事件 缩写：@

```html
<a v-on:click="doSomething"> ... </a>
```



## Date Property

Vue 在创建新组件实例的过程中调用此函数。它应该返回一个对象，然后 Vue 会通过响应性系统将其包裹起来，并以 `$data` 的形式存储在组件实例中。

```js
const app = Vue.createApp({
  data() {
    return { count: 4 }
  }
})

const vm = app.mount('#app')

console.log(vm.$data.count) // => 4
console.log(vm.count)       // => 4

// 修改 vm.count 的值也会更新 $data.count
vm.count = 5
console.log(vm.$data.count) // => 5

// 反之亦然
vm.$data.count = 6
console.log(vm.count) // => 6
```



## 方法

Vue 自动为 `methods` 绑定 `this`，以便于它始终指向组件实例。这将确保方法在用作事件监听或回调时保持正确的 `this` 指向。在定义 `methods` 时应避免使用箭头函数，因为这会阻止 Vue 绑定恰当的 `this` 指向。

```js
const app = Vue.createApp({
  data() {
    return { count: 4 }
  },
  methods: {
    increment() {
      // `this` 指向该组件实例
      this.count++
    }
  }
})

const vm = app.mount('#app')

console.log(vm.count) // => 4

vm.increment()

console.log(vm.count) // => 5
```

使用

```js
<button @click="increment">Up vote</button>
```



##  计算属性

所以，对于任何包含响应式数据的复杂逻辑，你都应该使用**计算属性**。

**意思就是在html代码里面不应该写太多的逻辑，逻辑部分抽离到计算属性中。**

```html
<div id="computed-basics">
  <p>Has published books:</p>
  <span>{{ publishedBooksMessage }}</span>
</div>
```



```js
Vue.createApp({
  data() {
    return {
      author: {
        name: 'John Doe',
        books: [
          'Vue 2 - Advanced Guide',
          'Vue 3 - Basic Guide',
          'Vue 4 - The Mystery'
        ]
      }
    }
  },
  computed: {
    // 计算属性的 getter
    publishedBooksMessage() {
      // `this` 指向 vm 实例
      return this.author.books.length > 0 ? 'Yes' : 'No'
    }
  }
}).mount('#computed-basics')
```

其实计算属性可以通过使用**函数**来达到相同的效果，只不过计算属性会缓存，不会每次调用都调用一次函数。

```html
<p>{{ calculateBooksMessage() }}</p>
```

```js
// 在组件中
methods: {
  calculateBooksMessage() {
    return this.author.books.length > 0 ? 'Yes' : 'No'
  }
}
```

### 计算属性的 Setter

计算属性默认只有 getter，不过在需要时你也可以提供一个 setter：

```js
// ...
computed: {
  fullName: {
    // getter
    get() {
      return this.firstName + ' ' + this.lastName
    },
    // setter
    set(newValue) {
      const names = newValue.split(' ')
      this.firstName = names[0]
      this.lastName = names[names.length - 1]
    }
  }
}
// ...
```

现在再运行 `vm.fullName = 'John Doe'` 时，setter 会被调用，`vm.firstName` 和 `vm.lastName` 也会相应地被更新。

## 侦听器

没咋看懂，后面需要复习。



## Class 与 Style 绑定

class使用数据的数组绑定

```html
<div :class="[activeClass, errorClass]"></div>
```

```js
data() {
  return {
    activeClass: 'active',
    errorClass: 'text-danger'
  }
}
```

绑定一个返回对象的[计算属性](https://v3.cn.vuejs.org/guide/computed.html)。计算属性就是数据和方法的结合。

```html
<div :class="classObject"></div>
```

```js
data() {
  return {
    isActive: true,
    error: null
  }
},
computed: {
  classObject() {
    return {
      active: this.isActive && !this.error,
      'text-danger': this.error && this.error.type === 'fatal'
    }
  }
}
```



style使用动态绑定

```html
<div :style="styleObject"></div>
```

1

```js
data() {
  return {
    styleObject: {
      color: 'red',
      fontSize: '13px'
    }
  }
}
```



## v-if vs v-show

一般来说，`v-if` 有更高的切换开销，而 `v-show` 有更高的初始渲染开销。因此，如果需要非常频繁地切换，则使用 `v-show` 较好；如果在运行时条件很少改变，则使用 `v-if` 较好。

v-if false直接不渲染

```html
<h1 v-if="awesome">Vue is awesome!</h1>
```

v-show 总是渲染，不过使用css来达到隐藏的效果

```html
<h1 v-show="ok">Hello!</h1>
```



## v-for 渲染一个数组

```html
<ul id="array-rendering">
  <li v-for="item in items">
    {{ item.message }}
  </li>
</ul>
```

```js
Vue.createApp({
  data() {
    return {
      items: [{ message: 'Foo' }, { message: 'Bar' }]
    }
  }
}).mount('#array-rendering')
```

在 `v-for` 块中，我们可以访问所有父作用域的 property。`v-for` 还支持一个可选的第二个参数，即当前项的索引。

```html
<ul id="array-with-index">
  <li v-for="(item, index) in items">
    {{ parentMessage }} - {{ index }} - {{ item.message }}
  </li>
</ul>
```

```js
Vue.createApp({
  data() {
    return {
      parentMessage: 'Parent',
      items: [{ message: 'Foo' }, { message: 'Bar' }]
    }
  }
}).mount('#array-with-index')
```

你也可以用 `v-for` 来遍历一个对象的 property。它只会显示对象的值，不会显示对象的索引

```html
<ul id="v-for-object" class="demo">
  <li v-for="value in myObject">
    {{ value }}
  </li>
</ul>
```

```js
Vue.createApp({
  data() {
    return {
      myObject: {
        title: 'How to do lists in Vue',
        author: 'Jane Doe',
        publishedAt: '2016-04-10'
      }
    }
  }
}).mount('#v-for-object')
```

当然也可以把键显示出来

```html
<li v-for="(value, name) in myObject">
  {{ name }}: {{ value }}
</li>
```

索引也可以使用

```html
<li v-for="(value, name, index) in myObject">
  {{ index }}. {{ name }}: {{ value }}
</li>
```



## 监听事件

点击加一，逻辑直接写在v-on中

```html
<div id="basic-event">
  <button @click="counter += 1">Add 1</button>
  <p>The button above has been clicked {{ counter }} times.</p>
</div>
```

```js
Vue.createApp({
  data() {
    return {
      counter: 0
    }
  }
}).mount('#basic-event')
```

为v-on定义一个方法

```html
<div id="event-with-method">
  <!-- `greet` 是在下面定义的方法名 -->
  <button @click="greet">Greet</button>
</div>
```

```js
Vue.createApp({
  data() {
    return {
      name: 'Vue.js'
    }
  },
  methods: {
    greet(event) {
      // `methods` 内部的 `this` 指向当前活动实例
      alert('Hello ' + this.name + '!')
      // `event` 是原生 DOM event
      if (event) {
        alert(event.target.tagName)
      }
    }
  }
}).mount('#event-with-method')
```



## 表单输入绑定

创建双向数据绑定

```html
<input v-model="message" placeholder="edit me" />
<p>Message is: {{ message }}</p>
```

绑定数组到多选框

```html
<div id="v-model-multiple-checkboxes">
  <input type="checkbox" id="jack" value="Jack" v-model="checkedNames" />
  <label for="jack">Jack</label>
  <input type="checkbox" id="john" value="John" v-model="checkedNames" />
  <label for="john">John</label>
  <input type="checkbox" id="mike" value="Mike" v-model="checkedNames" />
  <label for="mike">Mike</label>
  <br />
  <span>Checked names: {{ checkedNames }}</span>
</div>
```

```js
Vue.createApp({
  data() {
    return {
      checkedNames: []
    }
  }
}).mount('#v-model-multiple-checkboxes')
```



## 组件：

比较难，还得慢慢看

## 组件注册：

### 全局注册：

```js
Vue.createApp({...}).component('my-component-name', {
  // ... 选项 ...
})
```

使用：

```html
<div id="app">
  <component-a></component-a>
  <component-b></component-b>
  <component-c></component-c>
</div>
```

### 局部注册：

使用和上面的相同。

```js
const ComponentA = {
  /* ... */
}

const app = Vue.createApp({
  components: {
    'component-a': ComponentA,
  }
})
```



## props

```html
<blog-post title="My journey with Vue"></blog-post>  # 静态传值
<blog-post :title="post.title"></blog-post>  # 动态传值  数值、布尔值、数组、对象等 需要使用动态传值
```

### 单项数据流



## vue-cli使用：

build和config都是为webpack做配置。注：node是js的运行环境

build/build.js

```
// 删除之前大打包过的文件
rm(path.join(config.build.assetsRoot, config.build.assetsSubDirectory), err => {
// 重新打包
webpack(webpackConfig, (err, stats) => {
```

config/index.js

定义一些变量

```
host: 'localhost', // can be overwritten by process.env.HOST
port: 8080, // can be overwritten by process.env.PORT, if port is in use, a free one will be determined
autoOpenBrowser: false,
```

App.vue是最外层的父路由，所有子路由都在App.vue下

### src/router/index.js  

- routes中存的就是路由的数组，
- path就是你要访问你所创建的页面的路径，这里所配置的路由为''/，也就是根路径所以你直接访问localhost:8080就会出现一个App.vue中插入一个HelloWorld.vue的页面（这个相当于路由嵌套）
- name就是给当前路由命名，可以在其他页面通过$route.name访问到当前页面路由的name

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



### 新建vue

在template里面新建文件 demo1.vue，内容如下：

```html
<template>
    <div>
        test
    </div>
</template>
```

然后配置它的路由，先引入这个文件，用import，然后填写要访问这个文件的路由路径，这边写为/test，所有访问这个路由的url为：localhost:8080/#/test

注：vue脚手架默认的路由嵌套就是所有页面都嵌套在App.vue页面下

```js
import Vue from 'vue'
import Router from 'vue-router'
import HelloWorld from '@/components/HelloWorld'
import demo1 from '@/components/demo1'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'HelloWorld',
      component: HelloWorld
    },
    {
      path: '/test',
      name: 'demo1',
      component: demo1
    }
  ]
})
```

