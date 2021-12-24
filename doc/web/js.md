## 主要记录一下DOM

当网页被加载时，浏览器会创建页面的文档对象模型（*D*ocument *O*bject *M*odel）。

<img src="C:\Users\pzhang36\AppData\Roaming\Typora\typora-user-images\image-20211224224617021.png" alt="image-20211224224617021" style="zoom: 67%;" />

获取元素操作

| 方法                                    | 描述                   |
| :-------------------------------------- | :--------------------- |
| document.getElementById(*id*)           | 通过元素 id 来查找元素 |
| document.getElementsByTagName(*name*)   | 通过标签名来查找元素   |
| document.getElementsByClassName(*name*) | 通过类名来查找元素     |

```
通过 id 查找 HTML 元素
var myElement = document.getElementById("intro");
通过标签名查找 HTML 元素
var x = document.getElementsByTagName("p");
通过类名查找 HTML 元素
var x = document.getElementsByClassName("intro");
通过 CSS 选择器查找 HTML 元素
var x = document.querySelectorAll("p.intro");
```



实例：通过获取输入框的数据来验证输入数据的合法性，最后根据规则来相应。

```html
<!DOCTYPE html>
<html>
<body>

<h1>JavaScript Validation</h1>
<p>请输入 1 到 10 之间的数字：</p>
<input id="numb">
<button type="button" onclick="myFunction()">Submit</button>
<p id="demo"></p>

<script>
function myFunction() {
  // 获取 id = "numb" 的输入字段的值
  let x = document.getElementById("numb").value;
  // 如果 x 不是数字或小于 1 或大于 10
  let text;
  if (isNaN(x) || x < 1 || x > 10) {
    text = "输入无效";
  } else {
    text = "输入没问题";
  }
  document.getElementById("demo").innerHTML = text;
}
</script>
</body>
</html>

```

