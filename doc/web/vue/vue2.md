创建项目：3步

```
vue create app
```

```
npm run dev # 启动项目
```

安装插件

npm相关命令：

```
npm list -g --dept 0 // 查看安装的插件
npm ls  // 查看本地的c
npm install --registry=https://registry.npm.taobao.org

npm config get registry  // 查看npm当前镜像源
npm config set registry https://registry.npm.taobao.org/  // 设置源
```

## 指令：

```html
<p>Using v-html directive: <span v-html="rawHtml"></span></p>
<div v-bind:id="dynamicId"></div>
<p v-if="seen">现在你看到我了</p>
<a v-on:click="doSomething"> ... </a>
```

## Date Property

`Vue` 在创建新组件实例的过程中调用此函数。它应该返回一个对象，然后 `Vue` 会通过响应性系统将其包裹起来，并以 `$data` 的形式存储在组件实例中。

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

模块的暴露：

把模块暴露出去，让其他模块能够引用

```js
export default {
 name: 'App'
}
```



## props

```html
<blog-post title="My journey with Vue"></blog-post>  # 静态传值
<blog-post :title="post.title"></blog-post>  # 动态传值  数值、布尔值、数组、对象等 需要使用动态传值
```

### 单项数据流



## `vue-router`

实现页面跳转的功能。相当于 `requestmapping`

vue一般都是单页面，只有一个`index.html`，怎么实现页面的跳转呢，就用到了`vue-router`。

```js
import Vue from 'vue'
import Router from 'vue-router'
import HelloWorld from '@/components/HelloWorld' // 引入组件
import demo1 from '@/components/demo1'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',  // 这里是跳转的路径，访问这个路径就可以跳转到这个组件
      name: 'HelloWorld', // 这里是路由的名字
      component: HelloWorld  // 这里是路由跳转的组件
    },
    {
      path: '/test',
      name: 'demo1',
      component: demo1
    }
  ]
})
```

## `vuex`

全局数据的使用：`<h1>数量：{{$store.state.count}}</h1>`

计算属性的使用：`<h2>商品总价：{{$store.getters.totalPrice}}</h2>`

mutation 修改数据方法的使用：需要结合点击事件和方法来使用

```js
<button @click="changeEvent">添加数量</button>
methods: {
    changeEvent:function() {
      // this.$store.commit('setCount')
      this.$store.commit('setCountNum', 10)
    }
  }
```

异步获取数据：调用获取数据的方法，直接修改数据

```
mounted: function() {
    this.$store.dispatch('getDz')
  }
```



###  

