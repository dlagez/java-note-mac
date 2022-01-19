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



## npm使用



## 示例：

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



### 容器的使用：

```html
<body>
<!--普通容器-->
<div class="ui container">
    <p>测试测试测试测试测试测试测试测测试测试测试测试测试测试测试测试
        测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试
        测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试</p>
</div>
<!--文字容器，它会凸显文字，文字会比普通文字大，且居中-->
<div class="ui text container">
    <p>测试测试测试测试测试测试测试测测试测试测试测试测试测试测试测试
        测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试
        测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试</p>
</div>
<!-- 左右没有边距 -->
<div class="ui fluid container">
    <p>测试测试测试测试测试测试测试测测试测试测试测试测试测试测试测试
        测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试
        测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试
        测试测试测试测试测试测试测试测测试测试测试测试测试测试测试测试
        测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试
        测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试</p>
</div>
<!--文字居左显示-->
<div class="ui right aligned container">
    <p>测试测试测试测试测试测试测试测测试测试测试测试测试测试测试测试
        测试测试测试测试测试测试测试测</p>
</div>
<!--文字居右显示-->
<div class="ui left aligned container">
    <p>测试测试测试测试测试测试测试测测试测试测试测试测试测试测试测试
        测试测试测试测试测试测试测试测</p>
</div>
<!--文字居中显示-->
<div class="ui center aligned container">
    <p>测试测试测试测试测试测试测试测测试测试测试测试测试测试测试测试
        测试测试测试测试测试测试测试测</p>
</div>
</body>
```

效果：

![image-20220116120357469](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220116120357469.png)





### 输入框

```html
<body>
<!--普通容器-->
<div class="ui input">
    <input type="text" placeholder="标准输入框">
</div>
<br>
<div class="ui focus input">
    <input type="text" placeholder="标准输入框">
</div>
<div class="ui disabled input">
    <input type="text" placeholder="标准输入框">
</div>
<br>
<div class="ui transparent input">
    <input type="text" placeholder="标准输入框">
</div>
<br>
<div class="ui mini input">
    <input type="text" placeholder="标准输入框">
</div>
<br>
<div class="ui small input">
    <input type="text" placeholder="标准输入框">
</div>
<br>
<div class="ui large input">
    <input type="text" placeholder="标准输入框">
</div>

</body>
```

效果：

![image-20220116121436984](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220116121436984.png)

### 组合输入框

```html
<div class="ui error input">
    <input type="text" placeholder="错误">
</div>
<!--带图标的输入框-->
<div class="ui icon input">
    <input type="text" placeholder="账号">
    <i class="user icon"></i>
</div>

<div class="ui left icon input">
    <i class="lock icon"></i>
    <input type="text" placeholder="密码">
</div>

<div class="ui icon loading input">
    <input type="text" placeholder="搜索">
</div>

<div class="ui labeled input">
    <label for="amount" class="ui label">数量</label>
    <input type="text" placeholder="0.00">
</div>

<!--label相当于和input输入框并排显示-->
<div class="ui right labeled input">
    <div class="ui label">数量</div>
    <input type="text" id="1" placeholder="0.00">
    <label for="1" class="ui label">0.00</label>
</div>
<!--label的for指向input的id，绑定了关系，点击label，会聚焦到input输入框里面
 i标签会显示再input输入框里面-->
<div class="ui left icon right labeled input">
    <i class="tags icon"></i>
    <input type="text" id="name" placeholder="标签">
    <label for="name" class="ui label">标签</label>
</div>

<!--这里button是按钮，和上面的label不同-->
<div class="ui action input">
    <input type="text" placeholder="搜索">
    <button class="ui button">搜索</button>
</div>
```

效果：

![image-20220116130905292](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220116130905292.png)



### 图片：

标准图片如果比较大，会占满容器直到显示器真实大小。我们也可以控制图片显示的大小。

```html
<!--圆形显示-->
<img src="/images/wechat.jpg" alt="" class="ui circular small image">

<!--圆角显示-->
<img src="/images/wechat.jpg" alt="" class="ui rounded image">

<!--占满整个容器的图像，会随浏览器放大缩小-->
<img src="/images/wechat.jpg" alt="" class="ui fluid image">


<!--头像形式的图片-->
<img src="/images/wechat.jpg" alt="" class="ui avatar image">
<!--链接样式的图标-->
<a href="https://baidu.com" class="ui tiny image">
    <img src="/images/wechat.jpg" alt="">
</a>

<img src="/images/wechat.jpg" alt="" class="ui large image">
<br>
<img src="/images/wechat.jpg" alt="" class="ui small image">
<br>
<img src="/images/wechat.jpg" alt="" class="ui tiny image">
<br>
<!--原始大小-->
<img src="/images/wechat.jpg" alt="" class="ui image">
```

图片比较大，就不放效果图了。

### 标题

两种写法都可以

```html
    <h1 class="ui header">H1</h1>
    <h2 class="ui header">H1</h2>
    <h3 class="ui header">H1</h3>

    <br><br>

    <div class="ui tiny header">tiny</div>
    <div class="ui small header">small</div>
    <div class="ui large header">large</div>
```

组合写法



