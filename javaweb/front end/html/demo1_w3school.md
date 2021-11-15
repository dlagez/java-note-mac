### a标签

第一个不加http就不能跳转，会跳到本地 

```
file:///C:/code/demo_html/www.baidu.com
```

```
<a href="www.baidu.com">This is a link</a>
<a href="http://www.baidu.com">This is a link</a>
```

打开新窗口

```
<a href="http://www.w3school.com.cn/" target="_blank">Visit W3School!</a>
```

### img 标签

显示原始图像，不改变宽高

```
<img src="https://scpic.chinaz.net/files/pic/pic9/202111/apic36392.jpg">
```

调整宽高

```
        <img src="https://scpic.chinaz.net/files/pic/pic9/202111/apic36392.jpg" width="103" height="142">
```

#### 替换文本属性（Alt）

alt 属性用来为图像定义一串预备的可替换的文本。替换文本属性的值是用户定义的。

```
<img src="boat.gif" alt="Big Boat">
```

当浏览器无法载入图像时，替换文本属性可告诉读者他们失去的信息。此时，浏览器将显示这个替代性的文本而不是图像。为页面上的图像都加上替换文本属性是个好习惯，这样有助于更好的显示信息，并且对于那些使用纯文本浏览器的人来说是非常有用的。

#### 图像再文本中设置对齐方式

```
<p>图像 <img src="/i/eg_cute.gif" align="middle""> 在文本中</p>
```

图像再段落中浮动到左右侧

```
<p>
<img src ="/i/eg_cute.gif" align ="left"> 
带有图像的一个段落。图像的 align 属性设置为 "left"。图像将浮动到文本的左侧。
</p>
```



### 标签的通用属性

| class   | *classname*        | 规定元素的类名（classname）              |
| ------- | ------------------ | ---------------------------------------- |
| id      | *id*               | 规定元素的唯一 id                        |
| style   | *style_definition* | 规定元素的行内样式（inline style）       |
| title   | *text*             | 规定元素的额外信息（可在工具提示中显示） |
| align   | "center"           | 居中                                     |
| bgcolor | "yellow" <body>    | 设置背景颜色                             |

### 如何使用样式：

#### 外部样式表

```
<head>
<link rel="stylesheet" type="text/css" href="mystyle.css">
</head>
```

### 内部样式表

```
<head>

<style type="text/css">
body {background-color: red}
p {margin-left: 20px}
</style>
</head>
```

### 内联样式

当特殊的样式需要应用到个别元素时，就可以使用内联样式。 使用内联样式的方法是在相关的标签中使用样式属性。样式属性可以包含任何 CSS 属性。以下实例显示出如何改变段落的颜色和左外边距。

```
<p style="color: red; margin-left: 20px">
This is a paragraph
</p>
```

### 

### 命名锚语法：

### 实例

首先，我们在 HTML 文档中对锚进行命名（创建一个书签）：

```
<a name="tips">基本的注意事项 - 有用的提示</a>
```

然后，我们在同一个文档中创建指向该锚的链接：

```
<a href="#tips">有用的提示</a>
```

您也可以在其他页面中创建指向该锚的链接：

```
<a href="http://www.w3school.com.cn/html/html_links.asp#tips">有用的提示</a>
```

在上面的代码中，我们将 # 符号和锚名称添加到 URL 的末端，就可以直接链接到 tips 这个命名锚了。



### 表格

行由 tr 定义，列由 td 定义

定义边框

```
<p>
<img src ="/i/eg_cute.gif" align ="left"> 
带有图像的一个段落。图像的 align 属性设置为 "left"。图像将浮动到文本的左侧。
</p>
```

### 无序列表

无序列表始于 <ul> 标签。每个列表项始于 <li>。

### 有序列表

有序列表始于 <ol> 标签。每个列表项始于 <li> 标签。



### class 类

使用 .class 来定义类的样式

```html
<!DOCTYPE html>
<html>
<head>
<style>
.cities {
    background-color:black;
    color:white;
    margin:20px;
    padding:20px;
}	
</style>
</head>

<body>

<div class="cities">
<h2>London</h2>

<p>London is the capital city of England. It is the most populous city in the United Kingdom, with a metropolitan area of over 13 million inhabitants.</p>

<p>Standing on the River Thames, London has been a major settlement for two millennia, its history going back to its founding by the Romans, who named it Londinium.</p>
</div> 

</body>
</html>
```

使用 #id 来定义id的样式

```html
<!DOCTYPE html>
<html>
<head>
<style>
#myHeader {
  background-color: lightblue;
  color: black;
  padding: 40px;
  text-align: center;
}
</style>
</head>
<body>

<h1 id="myHeader">My Header</h1>

</body>
</html>
```



## 通过 ID 和链接实现 HTML 书签

HTML 书签用于让读者跳转至网页的特定部分。

如果页面很长，那么书签可能很有用。

要使用书签，您必须首先创建它，然后为它添加链接。

然后，当单击链接时，页面将滚动到带有书签的位置。

### 实例

首先，用 `id` 属性创建书签：

```
<h2 id="C4">第四章</h2>
```

然后，在同一张页面中，向这个书签添加一个链接（“跳转到第四章”）：

### 实例

```
<a href="#C4">跳转到第四章</a>
```



### 在网页中显示网页

```
<iframe src="URL"></iframe>
```