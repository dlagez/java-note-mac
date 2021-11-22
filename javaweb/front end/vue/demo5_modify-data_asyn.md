这个是`store`里面定义了user模块的数据和方法。

```js
const user = {
    // 数据
    state:() => ({
        username: "roc",
        age: 30
    }),
    // 修改数据的方法，修改数据都要在这里定义方法
    mutations: {
        setUsername:function(state) {
            state.username = "roc zhang"
        },
        setAge:function(state, num) {
            state.age = num
        }
    },
    // 异步请求数据
    actions: {
        asyncSetAge: function(context) {
            console.log(context)
            setTimeout(() => {
                context.commit('setAge', 50)
            }, 3000);
        }
    },
    // 计算属性
    getters: {
        description:function(state) {
            return state.username + '的年龄：' + state.age;
        }
    }
}

export default user
```

再 `store` 的`index.js` 里面注册模块。

```js
import { createStore } from 'vuex'
import user from './user'

export default createStore({
  // 设置全局数据的地方
  state: {
    count: 0,
    dzList: 'joke'
  },
  getters: {
    totalPrice:function(state) {
      return state.count*100
    }
  },
  // 修改状态的方法
  mutations: {
    setCount: function(state) {
      state.count++;
    },
    setCountNum: function(state, num) {
      state.count+=num;
    },
    setDzList:function(state, arr) {
      state.dzList = arr;
    }
  },
  // 异步修改状态
  actions: {
    getDz:function(context) { // 这里的content相当于store
      let api = 'https://autumnfish.cn/api/joke'
      fetch(api).then(res=>res.text()).then(result=>{
        console.log(result)
        context.commit('setDzList', result)
      })
    }
  },
  modules: { // 再这里注册模块
    user
  }
})

```

使用：

```js
<template>
    <div>
        <h1>用户名：{{$store.state.user.username}}</h1>
        <h1>年龄：{{$store.state.user.age}}</h1>
        <h1>描述：{{$store.getters.description}}</h1>
        <button @click="changeAge">异步修改年龄</button>
    </div>
</template>

<script>
    export default {
        mounted() {
            console.log(this.$store)
        },
        methods: {
            changeAge: function() {
                this.$store.dispatch('asyncSetAge') // 通过这个方法调用异步求情
            }
        }
    }   
</script>

```

