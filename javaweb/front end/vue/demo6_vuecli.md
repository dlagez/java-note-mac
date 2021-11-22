## vue-cli使用：

### 3.x

首选定义一个页面`about.vue`

```html
<template>
  <div class="about">
    <h1>This is an about page</h1>
  </div>
</template>

```

`router/index.js` 在这个注册上面的页面定义一个访问路径即可

```js
import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import User from '../views/User.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/about',
    name: 'About',
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () => import(/* webpackChunkName: "about" */ '../views/About.vue')
  },
  {
    path: '/user',
    name: 'User',
    component: User
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router

```



### 页面的嵌套：

```js
<template>
  <div class="home">
    <h1>数量：{{$store.state.count}}</h1>
    <h2>商品价格：100</h2>
    <h2>商品总价：{{$store.getters.totalPrice}}</h2>
    <button @click="changeEvent">添加数量</button>
  

  <h1>段子</h1>
  <p> {{$store.state.dzList}}</p>
  </div>
</template>

<script>
// @ is an alias to /src 直接在这里导入子组件
import HelloWorld from '@/components/HelloWorld.vue'

export default {
  name: 'Home',
  components: {
    HelloWorld
  }, 
  methods: {
    changeEvent:function() {
      // this.$store.commit('setCount')
      this.$store.commit('setCountNum', 10)
    }
  },
  mounted: function() {
    this.$store.dispatch('getDz')
  }
}
</script>

```



## 2.x

HelloWorld.vue 定义一个组件

```html
<template>
  <div>
  </div>
</template>
<script>
export default {
  name: 'HelloWorld',
  data () {
    return {
      msg: 'Welcome to Your Vue.js App'
      }
    }
  }
</script>

```

`router.index.js` 定义HelloWorld路由，也就是定义跳转路径

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

`main.js` 注册我们定义的路由

```js
import Vue from 'vue'
import App from './App'
import router from './router'  // 引入我们定义的路由 
import axios from 'axios'
//其他vue组件中就可以this.$axios调用使用
Vue.prototype.$axios = axios



/* eslint-disable no-new */
new Vue({
  el: '#app',
  router, // 使用我们定义的路由
  components: { App },
  template: '<App/>'
})

```

`App.vue` 使用路由

```html
<template>
  <div id="app">
    <img src="./assets/logo.png">
    <router-view/>  // 使用我们定义的路由
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>
```

直接使用 `<router-view/>` **访问链接**就直接显示组件了，如果想点击显示组件的话可以这样。

```html
 <template>
  <div id="app">
    <img src="./assets/logo.png">
    <router-link to="/">首页</router-link>
    <router-link to="/HelloWorld">HelloWorld</router-link>
    <router-link to="/demo1">demo1</router-link>
    <router-view/>
  </div>
</template>
```

