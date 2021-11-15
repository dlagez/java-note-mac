获取笑话案例

```js
<template>
  <div>
    <input type="button" value="获取笑话" @click="getJoke"> 
    <br>{{joke}}
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
    getJoke: function() {
      var that = this;
      this.$axios.get("https://autumnfish.cn/api/joke")
      .then(function(resp) {
        that.joke = resp.data
      })
    }
  }
}
</script>

```

