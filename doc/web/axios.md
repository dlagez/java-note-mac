在xue中使用axios

main.js中添加

```js
import axios from 'axios'
//其他vue组件中就可以this.$axios调用使用
Vue.prototype.$axios = axios
```

