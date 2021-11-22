安装：

```
npm install axios
```

在xue中使用axios

vue 3.x

不设置全局axios，单独页面引入：

请求本地的8080会报跨域的问题，这里先试一试网上的api，可以获取到数据。

```html
<template>
  <div class="home">
    <img alt="Vue logo" src="../assets/logo.png">
    <HelloWorld msg="Welcome to Your Vue.js App"/>
    <button @click="query">点击查询</button>
  </div>
</template>

<script>
// @ is an alias to /src
import HelloWorld from '@/components/HelloWorld.vue'
import axios from 'axios'


export default {
  name: 'Home',
  components: {
    HelloWorld
  },
  methods: {
    query: function()  {
      // http://localhost:8081/query
      axios.get("https://api.github.com/users").then(function(resopnse) {
        console.log(resopnse.data)
      })
    }
  }
}
</script>

```



vue 2.x

main.js中添加

```js
import axios from 'axios'
//其他vue组件中就可以this.$axios调用使用
Vue.prototype.$axios = axios
```

```js
  methods: {
    getJoke: function() {
      var that = this;
      this.$axios.get("https://autumnfish.cn/api/joke", {
        crossDomian: true
      })
      .then(function(resp) {
        that.joke = resp.data
      })
    },
    getStudent: function() {
      this.$axios.get("http://localhost:8081/query",
      {crossDomain: true, async:true}
      ).then(function (resp){
        
      })
    }
  },
```



### 什么是`Axios`

`Axios` 实现`AJAX`异步通信，特点如下：

- 从浏览器创建 `XMLHttpRequests`
- 支持 `Promise API`
- 拦截请求和相应
- 转换请求数据和响应数据
- 取消请求
- 自动转换`JSON`数据
- 客户端支持防御`XSRF`

## 全局配置axios

`ref:http://www.axios-js.com/zh-cn/docs/`

main.js

```js
// 为axios配置请求的根路径
axios.default.baseURL = 'http://api.com'
// 将axios挂载为app的全局属性之后，每个组件可以通过this直接访问
app.config.gloabalProperties.$http = axios

this.$http.get('/users')
```

带参数 get

```js
axios.get('/user', {
    params: {
      ID: 12345
    }
  })
  .then(function (response) {
    console.log(response);
  })
  .catch(function (error) {
    console.log(error);
  });
```

带参数 post

```js
axios.post('/user', {
    firstName: 'Fred',
    lastName: 'Flintstone'
  })
  .then(function (response) {
    console.log(response);
  })
  .catch(function (error) {
    console.log(error);
  });
```

获取图片

```js
// 获取远端图片
axios({
  method:'get',
  url:'http://bit.ly/2mTM3nY',
  responseType:'stream'
})
  .then(function(response) {
  response.data.pipe(fs.createWriteStream('ada_lovelace.jpg'))
});
```

