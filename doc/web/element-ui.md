安装：

```
npm i element-ui -S

// 这里踩坑。sass版本太高，要指定版本
npm install sass-loader@7.3.1 --save-dev
npm install node-sass@4.14.1 --registry=https://registry.npm.taobao.org

卸载
npm uninstall sass-loader
```

使用：

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
  render: h => h(App)  // element-ui c
})

```

sass使用，直接写css即可

```css
// scoped 表示样式只在当前组件生效
<style lang="sass"  scoped> 
    .login-box{
        width: 350px;
        margin: 150px auto;
        border: 1px solid black;
    }
</style>
```

