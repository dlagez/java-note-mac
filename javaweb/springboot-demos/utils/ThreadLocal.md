### ThreadLocal简介

线程安全的解决思路

- 互斥同步：synchroniezd和ReentrantLock
- 非阻塞同步：CAS、Atomic
- 无同步方案：栈封闭，本地存储（ThredLocal），可重入代码

著作权归https://pdai.tech所有。 链接：https://www.pdai.tech/md/java/thread/java-thread-x-threadlocal.html

> 该类提供了线程局部 (thread-local) 变量。这些变量不同于它们的普通对应物，因为访问某个变量(通过其 get 或 set 方法)的每个线程都有自己的局部变量，它独立于变量的初始化副本。ThreadLocal 实例通常是类中的 private static 字段，它们希望将状态与某一个线程(例如，用户 ID 或事务 ID)相关联。

我们都知道如果在多线程中操作同一变量会造成线程安全问题，ThreadLocal是一个在多线程中为每一个线程创建单独的变量副本的类。每一个ThreadLocal都有自己的线程变量副本，所以不会有多个线程操作共享变量而导致数据不一致的情况。

### ThreadLocal的理解

为什么要使用ThreadLocal这个类？直接使用锁不好吗？

举例：我们定义一个数据库连接的管理类，包括建立连接和打开连接两个方法。这两个方法没有使用锁。

类里面有一个私有静态连接对象。

```java
class ConnectionManager {
    private static Connection connect = null;

    public static Connection openConnection() {
        if (connect == null) {
            connect = DriverManager.getConnection();
        }
        return connect;
    }

    public static void closeConnection() {
        if (connect != null)
            connect.close();
    }
}
```

如果在多线程中使用就会存在安全问题：

- 建立连接在多线程中可能建立多个连接connection
- 有可能一个线程在使用connect进行数据操作，一个线程connecton关闭连接了。

解决方案有两个：

- 加锁
- 提供一个方法，方法类自动连接和关闭数据库连接。

加锁先不讨论，我们先来看第二个方法。

```java
class ConnectionManager {
    private Connection connect = null;

    public Connection openConnection() {
        if (connect == null) {
            connect = DriverManager.getConnection();
        }
        return connect;
    }

    public void closeConnection() {
        if (connect != null)
            connect.close();
    }
}

class Dao {
    public void insert() {
        ConnectionManager connectionManager = new ConnectionManager();
        Connection connection = connectionManager.openConnection();

        // 使用connection进行操作

        connectionManager.closeConnection();
    }
}
```

提供的方法没有线程安全问题，但是又有另外一个问题，方法中频繁的开启关闭数据库连接，倒置服务器压力较大。

ThreadLocal可以解决这样的问题：

### ThreadLocal原理：

#### 如何实现线程隔离

首先创建一个ThreadLocal对象，ThreadLocal<T> 和ArrayList一样，可以装任意类型。

```java
public T get() {
    Thread t = Thread.currentThread();
    ThreadLocalMap map = getMap(t);
    if (map != null) {
        ThreadLocalMap.Entry e = map.getEntry(this);
        if (e != null) {
            @SuppressWarnings("unchecked")
            T result = (T)e.value;
            return result;
        }
    }
    return setInitialValue();
}
```

get方法比较重要，

- 首先得到当前线程对象t。然后线程t中获取到ThreadLocalMap的成员属性threadLocals
- 如果当前线程的threadLocals已经初始化(即不为null) 并且存在以当前ThreadLocal对象为Key的值, 则直接返回当前线程要获取的对象(本例中为Connection);
- 如果当前线程的threadLocals已经初始化(即不为null)但是不存在以当前ThreadLocal对象为Key的的对象, 那么重新创建一个Connection对象, 并且添加到当前线程的threadLocals Map中,并返回
- 如果当前线程的threadLocals属性还没有被初始化, 则重新创建一个ThreadLocalMap对象, 并且创建一个Connection对象并添加到ThreadLocalMap对象中并返回。

很好理解：获取当前对象，将当前对象的变量存到ThreadLocalMap里面，每个线程都有一个，各用各的，互不干扰。

### ThreadLocalMap对象是什么

本质上来讲, 它就是一个Map, 但是这个ThreadLocalMap与我们平时见到的Map有点不一样

- 它没有实现Map接口;
- 它没有public的方法, 最多有一个default的构造方法, 因为这个ThreadLocalMap的方法仅仅在ThreadLocal类中调用, 属于静态内部类
- ThreadLocalMap的Entry实现继承了WeakReference<ThreadLocal<?>>
- 该方法仅仅用了一个Entry数组来存储Key, Value; Entry并不是链表形式, 而是每个bucket里面仅仅放一个Entry;

ref：https://www.pdai.tech/md/java/thread/java-thread-x-threadlocal.html
