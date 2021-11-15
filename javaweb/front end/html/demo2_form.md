### input 元素

| 类型   | 描述                                 |
| ------ | ------------------------------------ |
| text   | 定义常规文本输入。                   |
| radio  | 定义单选按钮输入（选择多个选择之一） |
| submit | 定义提交按钮（提交表单）             |

### Action 属性

*action 属性*定义在提交表单时执行的动作。

```
<form action="action_page.php">
```

### Method 属性

*method 属性*规定在提交表单时所用的 HTTP 方法（*GET* 或 *POST*）：

```
<form action="action_page.php" method="GET">
```

### Name 属性

如果要正确地被提交，每个输入字段必须设置一个 name 属性。

### Autocomplete 属性

`autocomplete` 属性规定表单是否应打开自动完成功能。

启用自动完成功能后，浏览器会根据用户之前输入的值自动填写值。

### select 元素（下拉列表）

*<select>* 元素定义*下拉列表*：您能够通过添加 selected 属性来定义预定义选项。

```
<select name="cars">
<option value="volvo">Volvo</option>
<option value="saab">Saab</option>
<option value="fiat">Fiat</option>
<option value="audi">Audi</option>
</select>
```

button 元素

```
<button type="button" onclick="alert('Hello World!')">Click Me!</button>
```



### input 输入类型

```
<input type="text"> 定义供文本输入的单行输入字段：
<input type="password"> 定义密码字段：
<input type="submit"> 定义提交表单数据至表单处理程序的按钮。
<input type="radio"> 定义单选按钮。
<input type="checkbox"> 定义复选框。
<input type="button> 定义按钮。
<input type="date"> 用于应该包含日期的输入字段。
<input type="color"> 用于应该包含颜色的输入字段。
<input type="range"> 用于应该包含一定范围内的值的输入字段。
<input type="email"> 用于应该包含电子邮件地址的输入字段。
<input type="search"> 用于搜索字段（搜索字段的表现类似常规文本字段）。
```

### input 属性

- *value* 属性规定输入字段的初始值：
- *readonly* 属性规定输入字段为只读（不能修改）：
- *disabled* 属性规定输入字段是禁用的。
- *size* 属性规定输入字段的尺寸（以字符计）：
- *maxlength* 属性规定输入字段允许的最大长度：