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

