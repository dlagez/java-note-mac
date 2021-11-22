自定义一个store ，这个store属于`user module`

store/index.js

```js
const user = {
    state:() => ({
        username: "roc",
        age: 30
    }),
    mutations: {
        setUsername:function(state) {
            this.state.username = "roc zhang"
        },
        setAge:function(state) {
            this.state.age = 18;
        }
    },
    actions: {
        asyncSetAge: function(context) {
            console.log(context)
            setTimeout(() => {
                context.commit('setAge')
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

在module里面使用

store/index.js

```
  modules: {
    user
  }
```

然后再任意一个地方就可以使用里面的数据和方法

```html
<h1>{{$store.state.user.username}}</h1>
```

