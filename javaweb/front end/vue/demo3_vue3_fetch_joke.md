`store.index.js`

在`actions`里面使用异步方法，异步方法也将使用`mutations`来修改`state`里面的数据

```js
import { createStore } from 'vuex'

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
     // 使用这个方法将异步请求的数据传入state里面
    setDzList:function(state, arr) {
      state.dzList = arr;
    }
  },
  // 异步修改状态
  actions: {
    getDz:function(context) { // 这里的content相当于store
      let api = 'https://autumnfish.cn/api/joke'
      // 第一个then返回的数据进行格式化
      fetch(api).then(res=>res.text()).then(result=>{
        console.log(result)
        context.commit('setDzList', result)
      })
    }
  },
  modules: {
  }
})

```

直接显示全局数据即可：

```html
  <h1>段子</h1>
  <p> {{$store.state.dzList}}</p>
  </div>
```

