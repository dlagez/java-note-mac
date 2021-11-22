语法：

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