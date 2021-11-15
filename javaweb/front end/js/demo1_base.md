## 基础

通过点击事件来改变 p 标签内容

```html
<button type="button"
onclick="document.getElementById('demo').innerHTML = Date()">
点击这里来显示日期和时间
</button>
<p id="demo"></p>
```

通过点击事件改变 img 的 src

```html
<button onclick="document.getElementById('myImage').src='/i/eg_bulbon.gif'">开灯</button>
```

改变样式css

```html
document.getElementById("demo").style.fontSize = "25px";
```

JavaScript 能够隐藏 HTML 元素

```
document.getElementById("demo").style.display="none";
```

button 的点击事件

```html

<h1>A Web Page</h1>
<p id="demo">一个段落</p>
<button type="button" onclick="myFunction()">试一试</button>
<script>
function myFunction() {
   document.getElementById("demo").innerHTML = "段落被更改。";
}
</script>

```

## 变量

### var 全局变量，可以在声明之前使用它

```
var carName;   # 他的值是 undefined
var x = 7;     # 类型只有数值和字符串

```

### let 块变量，注意局部变量的使用 ，不能再声明之前使用它

```
{ 
  let x = 10;
}
// 此处不可以使用 x
```



```
let i = 7;
for (let i = 0; i < 10; i++) {
  // 一些语句
}
// 此处 i 为 7
```

#### 全局作用域

如果在块外声明声明，那么 var 和 let 也很相似。它们都拥有*全局作用域*：

```
var x = 10;       // 全局作用域
let y = 6;       // 全局作用域
```

通过 let 关键词定义的全局变量不属于 window 对象：

```
let carName = "porsche";
// 此处的代码不可使用 window.carName
```

#### 重新声明

允许在程序的任何位置使用 var 重新声明 JavaScript 变量：

#### 实例

```
var x = 10;

// 现在，x 为 10
 
var x = 6;

// 现在，x 为 6
```



### const  变量必须在声明时赋值：

不能更改，它没有定义常量值。它定义了对值的常量引用。

```
const PI = 3.141592653589793;
PI = 3.14;      // 会出错
PI = PI + 10;   // 也会出错
```

```
// 您可以创建 const 对象：
const car = {type:"porsche", model:"911", color:"Black"};

// 您可以更改属性：
car.color = "White";

// 您可以添加属性：
car.owner = "Bill";
```



## 数据类型

**字符串值，数值，布尔值，数组，对象。**

```
var length = 7;                             // 数字
var lastName = "Gates";                      // 字符串
var cars = ["Porsche", "Volvo", "BMW"];         // 数组
var x = {firstName:"Bill", lastName:"Gates"};    // 对象 
```

- 当数值和字符串相加时，JavaScript 将把数值视作字符串。
- JavaScript 从左向右计算表达式。不同的次序会产生不同的结果：



### typeof 运算符

```
typeof ""                  // 返回 "string"
typeof "Bill"              // 返回 "string"
typeof "Bill Gates"          // 返回 "string"
```



### 函数

toCelsius 引用的是函数对象，而 toCelsius() 引用的是函数结果。

```
<script>
function toCelsius(f) {
    return (5/9) * (f-32);
}
document.getElementById("demo").innerHTML = toCelsius;
</script>

直接输出 function toCelsius(f) { return (5/9) * (f-32); }
```

```
<script>
function toCelsius(f) {
    return (5/9) * (f-32);
}
document.getElementById("demo").innerHTML = toCelsius(33);
</script>

直接输出 函数计算后的结果
```

