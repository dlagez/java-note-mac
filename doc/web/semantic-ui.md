## 基本使用：

下载`js css`文件来使用

访问这个网站下载zip文件 `https://semantic-ui.com/introduction/getting-started.html`

将下面的两个文件复制到你的项目中。

![image-20220115223738156](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220115223738156.png)



下面是一个页面中使用semantic-ui

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
  <link rel="stylesheet" href="/css/semantic.min.css">
  <script src="https://cdn.jsdelivr.net/npm/jquery@3.2/dist/jquery.min.js"></script>
  <script src="/js/semantic.min.js"></script>
</head>
<body>

<button class="ui red large button">Hello World</button>

</body>
</html>
```

需要注意两点：

- 在springboot中引入了`js`和`css`文件需要重新启动项目。不然会找不到`js`和`css`
- 在引入`semantic-ui`的`js`文件之前需要先引入`jquery`文件

在上面的网页中可以找到semantic的各种示例，只需要将示例的代码复制到页面中即可。



示例：

![image-20220115224651953](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220115224651953.png)



在页面中使用即可。

```html
<body>

<button class="ui red large button">Hello World</button>

<button class="ui primary button">
    Save
</button>
<button class="ui button">
    Discard
</button>
</body>
```



效果：

![image-20220115224813256](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220115224813256.png)



再找个表格试试看

```html
<div class="ui form">
    <div class="fields">
        <div class="field">
            <label>First name</label>
            <input type="text" placeholder="First Name">
        </div>
        <div class="field">
            <label>中间名</label>
            <input type="text" placeholder="中间名">
        </div>
        <div class="field">
            <label>姓氏</label>
            <input type="text" placeholder="Last Name">
        </div>
    </div>
</div>
```



效果：

![image-20220115225326448](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220115225326448.png)



## 各种组件

### button

通过class可以改变button的外观，注意`inverted`，它表示反色，就是按钮周围一圈是指定的颜色，鼠标移上去才和其他的按钮一样全部为指定颜色。

```html
<button class="ui button">按钮</button>
<button class="ui basic button">按钮</button>
<button class="ui positive button">按钮</button>
<button class="ui negative button">按钮</button>

<button class="ui red button">按钮</button>
<button class="ui orange button">按钮</button>  #  改变颜色
<button class="ui inverted orange button">按钮</button>  # 反色

<br><br>

<button class="ui button">normal</button>
<button class="mini ui button">normal</button>
<button class="tiny ui button">normal</button>
<button class="medium ui button">normal</button>
<button class="medium ui negative button">normal</button>  # 组合使用
```

效果：

![image-20220115231120529](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220115231120529.png)



### 动画：

使用`animated`作为一个整体来封装里面的内容，这个div就有了动画效果了。默认是显示，鼠标移上去就把隐藏的内容显示出来了。

```html
<div class="ui animated button">
    <div class="visible content">显示</div>
    <div class="hidden content">隐藏</div>
</div>
```



```html
<!--vertical 垂直切换的动画  fade 表示渐进切换动画， 不分垂直还是水平切换 -->
<div class="ui vertical fade animated button">
    <div class="visible content">显示</div>
    <div class="hidden content">
        <i class="right arrow icon"></i>
    </div>
</div>
```



### button变形

由于没有引入icon图标所以图标显示会出现异常，下同。

```html
<button class="ui button">标准</button>

<button class="ui active button">激活</button>
<button class="ui disabled button">禁用</button>

<br><br>
<button class="ui right floated button">向右</button>
<button class="ui left floated button">向左</button>

<button class="ui fluid button">整行</button>

<button class="ui compact button">紧凑</button>

<button class="circular ui teal icon button">
    <i class="settings icon"></i>
</button>
```

效果

![image-20220115233819625](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220115233819625.png)

### button的组合

```html
<!--按钮的组合-->
<!--attached 表示按钮紧紧依附-->
<div class="ui top attached button">TOP</div>
<div class="ui attached segment"> <p>pppppp</p></div>
<div class="ui bottom attached button">BOTTOM</div>

<!--按钮左右紧紧依附-->
<button class="ui left attached button">左</button>
<button class="ui right attached button">右</button>

<!--  vertical 垂直排列的按钮组-->
<div class="ui vertical buttons">
    <div class="ui button">A</div>
    <div class="ui button">A</div>
    <div class="ui button">A</div>
</div>
<!--  buttons 按钮组-->
<div class="ui icon buttons">
    <button class="ui button">
        <i class="play icon"></i>
    </button>
    <button class="ui button">
        <i class="pause icon"></i>
    </button>
    <button class="ui button">
        <i class="shuffle icon"></i>
    </button>
</div>
```

效果

![image-20220115234847620](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220115234847620.png)
