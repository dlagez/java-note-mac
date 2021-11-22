1.创建项目：

```
vue init webpack vue_element
```

安装依赖：

```
npm install  // 好像是安装nn_module
npm install vue-router
npm i element-ui -S
```

使用router

login.vue

```html
<template>
    <div>
        <h1> {{ msg }}</h1>
        <el-form ref="form" :model="form">
            <el-form-item label="账号">
                <el-input type="text" placeholder="请输入用户名" v-model="form.name"></el-input>
            </el-form-item>
            <el-form-item>
                <el-input type="password" placeholder="请输入密码" v-model="form.password"></el-input>
            </el-form-item>
            <el-form-item>
                <el-button type="primary" @click="onSubmit">登陆</el-button>
            </el-form-item>
        </el-form>
    </div>
</template>

<script>
    export default {
        name: "Login",
        data() {
            return {
                msg: "Welcome to Your Vue.js App",
                form: {
                    name: '',
                    password: ''
                }
            }
        }, 
        methods: {
            onSubmit() {
                alert("Hello")
            }
        }
    }
</script>
```

index.js 定义router

```js
import Vue from 'vue'
import Router from 'vue-router'
import Login from '../views/Login'

Vue.use(Router)

export default new Router({
    routes: [{
        // 登陆页
        path: '/login',
        name: 'Login',
        component: Login

    }]
})
```

main.js 在项目中导如router

```js
// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import VueRouter from 'vue-router'
import router from './router'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css';

Vue.use(VueRouter)
Vue.use(ElementUI);

Vue.config.productionTip = false

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  render: h => h(App)  // 实例化element-ui
})

```

app.vue 使用router

```js
<template>
  <div id="app">
    <router-view/>
  </div>
</template>

<script>


export default {
  name: 'App',

}
</script>

<style>

</style>

```

