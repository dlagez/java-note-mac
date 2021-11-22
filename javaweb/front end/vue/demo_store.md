store的使用：

```js
import { createStore } from 'vuex'
import user from './user'

export default createStore({
  // 设置全局数据的地方
  state: {
    count: 0,
    dzList: 'joke'
  },
  // 相当于计算属性
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
  modules: {
    user
  }
})

```

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

`modules`使用：就是把这个`store`复制一份，用在一个小的模块上面。使用`export`将模块导出后可以在`modules`里面使用，除了获取数据要加上 `<h1>年龄：{{$store.state.user.age}}</h1>` 模块名之外，没有上面需要改变的。

```js
const user = {
    state:() => ({
        username: "roc",
        age: 30
    }),
    mutations: {
        setUsername:function(state) {
            state.username = "roc zhang"
        },
        setAge:function(state, num) {
            state.age = num
        }
    },
    actions: {
        asyncSetAge: function(context) {
            console.log(context)
            setTimeout(() => {
                context.commit('setAge', 50)
            }, 3000);
        }
    },
    getters: {
        description:function(state) {
            return state.username + '的年龄：' + state.age;
        }
    }
}

export default user
```

