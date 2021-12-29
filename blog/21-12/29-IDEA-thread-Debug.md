使用`IDEA`多线程`debug`

懒汉单例模式：被外部类调用的时候内部类才会被加载。



创建一个单例模式，但是这个单例模式会有线程安全问题

```java
package design_patterns.Demo4_LazySimpleSingleton;

public class LazySimpleSingleton {
    private LazySimpleSingleton() {};

    private static LazySimpleSingleton lazy = null;

    // 这里在多线程的时候会出现问题
    public static LazySimpleSingleton getInstance() {
        if (lazy == null) {
            lazy = new LazySimpleSingleton();
        }
        return lazy;
    }
}
```

创建一个线程用来获取单例

```java
pckage design_patterns.Demo4_LazySimpleSingleton;

public class ExectorThread implements Runnable {
    @Override
    public void run() {
        LazySimpleSingleton instance = LazySimpleSingleton.getInstance();
        System.out.println(Thread.currentThread().getName() + ":" + instance);
    }
}
```

编写测试类

```java
package design_patterns.Demo4_LazySimpleSingleton;

public class test_LazySimpleSingleton {
    // 饿汉单例模式可能会出现线程安全问题
    public static void main(String[] args) {
        Thread t1 = new Thread(new ExectorThread());
        Thread t2 = new Thread(new ExectorThread());
        t1.start();
        t2.start();
        System.out.println("End");
    }
}
```



上面的代码是存在安全隐患的，测试类执行之后有很小的概率出现下面的问题：

![image-20211229104443743](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211229104443743.png)

单例模式获取的对象不一样，在多线程获取单例的测试中获取到了两个对象。

我测试了大概十次会出现一次线程安全问题，我们用`idea`的线程模式调试来重现这个问题。

#### 第一步：

将线程类打断点，并右击断电设置成`Thread`模式

![image-20211229104753259](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211229104753259.png)

#### 第二步：

将单例模式的类也打上断点

![image-20211229105045832](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211229105045832.png)

#### 第三步：

将测试类也打上断点

![image-20211229105135969](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20211229105135969.png)

最后点击debug调试即可











