安装：

```
npm install axios
```

使用：

```js
<script>
// @ is an alias to /src
import HelloWorld from '@/components/HelloWorld.vue'
import axios from 'axios'

export default {
  name: 'Home',
  components: {
    HelloWorld
  },
  setup() {
    console.log('setup')
    axios.get("http://localhost:8088/ebookLikeReq?name=vue").then((response) => {
      console.log(response)
    })
  }
}
</script>
```

### 使用ref实现数据的绑定

```js
<script>
// @ is an alias to /src
import {onMounted, ref} from "vue";
import HelloWorld from '@/components/HelloWorld.vue'
import axios from 'axios'

export default {
  name: 'Home',
  components: {
    HelloWorld
  },
  setup() {
    console.log('setup')
    const ebooks = ref()
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

