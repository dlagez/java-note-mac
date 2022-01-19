### `router-link`

```html
  <div id="nav">
    <router-link to="/">Home</router-link> |
    <router-link to="/about">About</router-link>
  </div>
```

类似于上面样子的使用，相当于一个链接，点击哪个链接就会显示哪个页面，每个`link`都会链接到一个`views`页面

### `router-view`

will display the component that corresponds to the url

就很简单了，他会显示上面点击的router