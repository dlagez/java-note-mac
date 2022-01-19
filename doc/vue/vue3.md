### create a project

```
Vue CLI v4.5.15
? Please pick a preset:
  Default ([Vue 2] babel, eslint)
  Default (Vue 3) ([Vue 3] babel, eslint)
> Manually select features
```

```
Vue CLI v4.5.15
? Please pick a preset: Manually select features
? Check the features needed for your project: (Press <space> to select, <a> to toggle all, <i> to invert selection)
>(*) Choose Vue version
 (*) Babel
 ( ) TypeScript
 ( ) Progressive Web App (PWA) Support
 (*) Router
 (*) Vuex
 ( ) CSS Pre-processors
 ( ) Linter / Formatter
 ( ) Unit Testing
 ( ) E2E Testing
```

- 利用babel就可以让我们在当前的项目中随意的使用这些新最新的es6，甚至es7的语法。说白了就是把各种`javascript`千奇百怪的语言统统专为浏览器可以认识的语言。
- `TypeScript` 是一种给 JavaScript 添加特性的语言扩展。增加的功能包括：
- `Progressive Web App (PWA) Support `一是给项目添加一些webapp支持

```
Vue CLI v4.5.15
? Please pick a preset: Manually select features
? Check the features needed for your project: Choose Vue version, Babel, Router, Vuex, CSS Pre-processors
? Choose a version of Vue.js that you want to start the project with 3.x
? Use history mode for router? (Requires proper server setup for index fallback in production) (Y/n) n
```





指令：

- v-bind  将这个元素节点的 `title` attribute 和当前活跃实例的 `message` property 保持一致，它一般与默认属性一起使用，比如：`v-bind:id`



语法：Skf12345

取消了data，method等函数。使用setup() 替代。数据需要return出去才能再页面渲染。

```js
export default {
  name: 'Home',
  components: {
    HelloWorld
  },
  setup() {
    console.log('setup')
    const ebooks = ref()  // 使用ref实现数据的绑定
    // 在这里写函数会在页面渲染完之后再执行。可能会拿到数据比较晚会出问题。比如操作数据会出错，因为数据还没有拿到。
    onMounted(() => {
      axios.get("http://localhost:8088/ebookLikeReq?name=vue").then((response) => {
        const data = response.data
        ebooks.value = response.data.content
        console.log(response)
      });
    })
    return {
      ebooks
    }
  }
}
</script>
```

使用 {} 定义一个对象。