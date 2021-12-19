# lambda表达式的使用方法

lambda的简要介绍：

![image-20211218211800987](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211218211800987.png)

在哪里使用lambda表达式：在函数式接口上使用lambda表达式。

## 使用方法：

我现在有一个需求，读取一个文件的第一行，平时我们定义的方法是直接使用`BufferedReader`方法读取数据。

```java
    public static String processFile() throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader("/Volumes/roczhang/temp/a.txt"));) {
            return br.readLine();
        }
    }
```



## 改造成通用方法

现在我们需要改造一下。使他变成一个通用的读取文件的方法。

我们需要把`processFile`的行为参数化。把行为传递给`processFile`以便它可以利用`BufferedReader`执行不同的行为。

### 第一步：

首先定义一个参数化的行为（接口）

```java
package java8.demo2_processFile;

import java.io.BufferedReader;
import java.io.IOException;

public interface BufferReaderProcessor {
    String process(BufferedReader b) throws IOException;
}
```

### 第二步：

然后改造方法，将这个行为通过参数传递给方法，然后使用这个行为来操作`BufferedReader`

```java
public static String processFile2(BufferReaderProcessor p) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader("/Volumes/roczhang/temp/a.txt"));) {
            return p.process(br);
        }
    }
```

### 第三步：

使用的话配合lambda表达式即可。比如我们想读取两行数据就不用重新定义一个函数了。直接在lambda表达式里面实现即可。对于比较简单的操作可以直接写在lambda表达式里面，太过于复杂代码就不太好读懂了。

```java
String s = processFile2((BufferedReader br) -> br.readLine());
System.out.println("s: " + s);

String s2 = processFile2((BufferedReader br) -> br.readLine() + br.readLine());
System.out.println("s: " + s2);
```



完整的代码如下。

```java
package java8.demo2_processFile;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class demo {
    public static void main(String[] args) throws IOException {
        System.out.println(processFile());

        String s = processFile2((BufferedReader br) -> br.readLine());
        System.out.println("s: " + s);


        String s2 = processFile2((BufferedReader br) -> br.readLine() + br.readLine());
        System.out.println("s: " + s2);
    }

    public static String processFile() throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader("/Volumes/roczhang/temp/a.txt"));) {
            return br.readLine();
        }
    }

    public static String processFile2(BufferReaderProcessor p) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader("/Volumes/roczhang/temp/a.txt"));) {
            return p.process(br);
        }
    }
}
```

