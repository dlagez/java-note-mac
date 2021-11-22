前端矢量库的使用

`https://www.iconfont.cn/`

### 在我的项目里面使用Symbol引入方式

#### 第一步：拷贝项目下面生成的symbol代码：

```js
//at.alicdn.com/t/font_8d5l8fzk5b87iudi.js
```

#### 第二步：加入通用css代码（引入一次就行）：

```js
<style type="text/css">
    .icon {
       width: 1em; height: 1em;
       vertical-align: -0.15em;
       fill: currentColor;
       overflow: hidden;
    }
</style>
```

#### 第三步：挑选相应图标并获取类名，应用于页面：

```js
<svg class="icon" aria-hidden="true">
    <use xlink:href="#icon-xxx"></use>
</svg>
```



### Font clss 方式

首先在html里面引入

```
<link rel="stylesheet" href="//at.alicdn.com/t/font_2948129_qzkvua9pqi.css">
```

然后使用即可

```
<span class="iconfont icon-liebiao"></span>
```

