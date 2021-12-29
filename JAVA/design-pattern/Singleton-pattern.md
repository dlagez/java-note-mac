面试设计模式系列之单例模式



面试官：简单说一下单例模式

你：

单例模式分为两种，

- 一种是饿汉单例模式，
- 一种是懒汉单例模式。

他们都有各自的优缺点：

饿汉单例模式在类加载的时候就立即初始化了，并且在初始化的过程中就创建了单例。它是绝对线程安全。没有加任何的锁，执行效率比较高。但是在类加载的时候就已经初始化了，通过static关键字修饰单例，不管用不用都占着空间，浪费内存。

懒汉单例模式在初始化的时候不会创建单例，而是在需要的用到单例的时候才去创建单例，这样做的好处是如果没有用到该单例的时候不会占用内存，但是懒汉单例在创建的时候会导致线程安全问题，同时使用两个线程去创建单例会得到两个不同的单例。而锁会导致性能问题，多个线程创建单例时会导致线程都处于阻塞状态，导致cpu的负载增大。

懒汉单例模式可以使用枚举类来实现。枚举类其实是通过类名和类对象找到一个唯一的枚举对象。因此枚举对象不可能被类加载器加载多次。

我们都可以使用反射和序列化的方法来进行懒汉单例模式的破坏。创建两个单例对象，一个对象为空`null`，一个为单例对象。将单例对象序列化到磁盘中，然后反序列化出来，将空对象指向这个反序列化出来的单例。把他们两进行对比可以发现他们不是同一对象。	



### 饿汉单例模式：

饿汉单例模式在类加载的时候就立即初始化，并且创建单例模式。它绝对线程安全

优点：没有加任何锁、执行效率比较高，用户体验比懒汉式单例模式更好

缺点：类加载的时候就初始化，不管用与不用都占着空间，浪费了内存。

```java
public class HungrySingleton {
    // 先静态后动态
    // 先属性后方法
    // 先上后下
    private static final HungrySingleton hungrySingleton = new HungrySingleton();

    private HungrySingleton() {}

    public static HungrySingleton getInstance() {
        return hungrySingleton;
    }
}
```

利用静态代码块

```java
public class HungryStaticSingleton {
    private static final HungryStaticSingleton hungrySingleton;

    static {
        hungrySingleton = new HungryStaticSingleton();
    }

    private HungryStaticSingleton() {}

    public static HungryStaticSingleton getInstance() {
        return hungrySingleton;
    }
}
```

### 懒汉单例模式：

简单的懒汉单例模式，会出现线程安全问题

```java
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

#### 加锁的懒汉单例模式

它比简单懒汉单例模式更安全，即使在多线程模式下也不会出现线程安全问题。但是加锁了之后，在线程较多的情况下，cpu的压力会增大，因为会有大批的线程阻塞。

```java
public class LazyDoubleCheckSingleton {
    private volatile static LazyDoubleCheckSingleton lazy = null;

    private LazyDoubleCheckSingleton() {}

    public static LazyDoubleCheckSingleton getInstance() {
        if (lazy == null) {
            synchronized (LazyDoubleCheckSingleton.class) {
                if (lazy == null) {
                    lazy = new LazyDoubleCheckSingleton();
                   //分配内存给这个对象
                   // 初始化对象
                   // 设置lazy指向刚分配的内存地址
                }
            }
        }
        return lazy;
    }
}
```

完美的懒汉单例模式

在其构造方法中做一些限制。他的加载不是直接创建对象，而是通过一个LazyHolder来进行对象的创建，在创建单例的时候可以检查对象是否已经创建。而且没有线程安全问题。

```java
public class LazyInnerClassSingleton {
    private LazyInnerClassSingleton() {
        if (LazyHolder.LAZY != null) {
            throw new RuntimeException("不允许创建多个实例");
        }
    }
    public static final LazyInnerClassSingleton getInstance() {
        return LazyHolder.LAZY;
    }

    // 默认不加载
    private static class LazyHolder {
        private static final LazyInnerClassSingleton LAZY = new LazyInnerClassSingleton();
    }
}
```



#### 单例模式的破坏：

序列化破坏单例

用的是饿汉单例模式

```java
public class SeriableSingleton implements Serializable {
  // 序列化就是把内存中的状态通过转换成字节码形式
  // 从而转换一个I/O流，写入到其他地方
  // 内存中的状态会永久保存下来
  
  // 反序列化就是将已经持久化的字节码内容转换为I/O流
  // 通过i/O流的读取，进而将读取的内容转换成java对象
  // 转换的过程中会重新创建对象new
  public final static SeriableSingleton INSTANCE = new SeriableSingleton();

    private SeriableSingleton() {}

    public static SeriableSingleton getInstance() {
        return INSTANCE;
    }
}
```



```java
public class test_SeriableSingleton {
    public static void main(String[] args) {
        SeriableSingleton s1 = null;
        SeriableSingleton s2 = SeriableSingleton.getInstance(); // 先创建一个单例

        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream("SeriableSingleton.obj");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(s2);  // 将单例序列化到内存
            oos.flush();
            oos.close();

          	//读取内存中的单例
            FileInputStream fis = new FileInputStream("SeriableSingleton.obj"); 
            ObjectInputStream ois = new ObjectInputStream(fis);
            s1 = (SeriableSingleton) ois.readObject();
            ois.close();

            System.out.println(s1);
            System.out.println(s2);
            System.out.println(s1 == s2);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

显示的结果

从结果可以看出，反序列化后的对象和手动创建出来的对象是不一样的。即使他们是同一个对象，但是经过序列化和反序列化之后就不一样了。

```
design_patterns.Demo4_LazySimpleSingleton.SeriableSingleton@9807454
design_patterns.Demo4_LazySimpleSingleton.SeriableSingleton@49476842
false
```

#### 防止序列化

上面的反序列化破坏单例模式也可以避免。添加一下代码即可。

```java
public class SeriableSingleton implements Serializable {
    public final static SeriableSingleton INSTANCE = new SeriableSingleton();

    private SeriableSingleton() {}

    public static SeriableSingleton getInstance() {
        return INSTANCE;
    }
    private Object readResolve() {
    		return INSTANCE;
    }
}

```

结果：

```
design_patterns.Demo4_LazySimpleSingleton.SeriableSingleton@49476842
design_patterns.Demo4_LazySimpleSingleton.SeriableSingleton@49476842
true
```



### 注册式单例模式：

又称登记式单例模式，有两种：一种为枚举式单例模式，另一种是容器式单例模式。

#### 枚举式单例模式：

在静态代码块中就给INSTANCE进行了赋值，是饿汉式单例模式的实现。

枚举类型其实是通过类名和类对象找到一个唯一的枚举对象。

```java
/**
 * 注册式单例模式
 */
public enum EnumSingleton {

    INSTANCE;

    private Object data;

    public Object getData()  {
        return data;
    }

    public void setData(Object data) {
        this.data = data;
    }

    public static EnumSingleton getInstance() {
        return INSTANCE;
    }
}
```



测试代码：

```java
public class test_EnumSingleton {
    public static void main(String[] args) {
        try {
            EnumSingleton instance1 = null;

            EnumSingleton instance2 = EnumSingleton.getInstance();
            instance2.setData(new Object());

            FileOutputStream fos = new FileOutputStream("EnumSingleton.obj");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(instance2);
            oos.flush();
            oos.close();

            FileInputStream fis = new FileInputStream("EnumSingleton.obj");
            ObjectInputStream ois = new ObjectInputStream(fis);
            instance1 = (EnumSingleton) ois.readObject();
            ois.close();

            System.out.println(instance1.getData());
            System.out.println(instance2.getData());
            System.out.println(instance1.getData() == instance2.getData());
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
```

结果：很明显，单例模式没有被破坏，即使使用了序列化从新读取对象之后，读取出来的对象和序列化之前的对象是同一个对象。

```
java.lang.Object@378bf509
java.lang.Object@378bf509
true
```



#### 容器式单例

容器式单例模式，使用一个`ConcurrentHashMap`来存储单例，所以它可以存储很多个单例，这里是以单例的className来区分单例的，每个className仅仅职能创建一个单例。

```java
public class ContainerSingleton {
    private ContainerSingleton() {}
    
    private static Map<String, Object> ioc = new ConcurrentHashMap<String, Object>();
    
    public static Object getBean(String className) {
        synchronized (ioc) {
            if (!ioc.containsKey(className)) {
                Object obj = null;
                try {
                    obj = Class.forName(className).newInstance();
                    ioc.put(className, obj);
                } catch (ClassNotFoundException e) {
                    e.printStackTrace();
                } catch (InstantiationException e) {
                    e.printStackTrace();
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
                return obj;
            } else {
                return ioc.get(className);
            }
        }
    }
}
```







ref：《Spring5 核心原理》整篇文章就是在做上面的笔记。加上一些自己的思考。