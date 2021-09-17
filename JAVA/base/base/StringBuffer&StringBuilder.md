### String 

String类是一个不可变类，即一旦一个String对象被创建之后。是不可被改变的。

我们平时使用的代码

```java
String  a = "123"
a = "456"
```

是因为首先使得a引用指向"123"，第二句将a引用改变，指向"456"。"123"对象没变，也没有消失。但是已经没有引用指向它了。

### StringBuffer

线程安全。他的操作方法都加上了synchronized关键字.

继承AbstractStringBuilder，这个类有两个字段，一个是字节数组byte[] value，一个是数量。说白了StringBuffer本质上是一个字符数组。

StringBuffer的操作append和insert都是在value字符数组上操作的。所以效率很高。

#### 扩容：

```java
int newCapacity = (oldCapacity << 1) + 2;
```

两倍加二

### StringBuilder

