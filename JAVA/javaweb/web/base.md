基本使用

main.js

```js
// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import axios from 'axios'
//其他vue组件中就可以this.$axios调用使用
Vue.prototype.$axios = axios


Vue.config.productionTip = false

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})
```

vue

```vue
<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
    <input type="button" value="get joke" @click="getJoke">
    <!-- <input type="button" value="post请求" class="post"> -->
    <p> {{ joke }}</p>
  </div>
</template>

<script>
export default {
  name: 'HelloWorld',
  data () {
    return {
      msg: 'Welcome to Your Vue.js App',
      joke: "one joke"
    }
  },
  methods: {
    getJoke:function() {
      var that = this;
      // https://autumnfish.cn/api/joke
      this.$axios.get("localhost:8081/query")
      .then(function(response) {
        // console.log(response)
        console.log(response.data)
        that.joke = response.data
      }, function (err) {

      })
    }
  }
}
</script>




<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h1, h2 {
  font-weight: normal;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>

```

