### 工厂模式



面试官：简单说一下工厂模式

你：

​	工厂方法模式是指定义一个创建对象的接口，但让实现这个接口的类来决定实例化哪个类，工厂方法模式让类的实例化推迟到了子类中进行。

​	核心的工厂类不再负责所有产品的创建，而是将具体创建工作交给子类去做。这个核心类仅仅负责给出具体工厂必须实现的接口，而不负责哪一个产品类被实例化这种细节，这使得工厂方法模式可以允许系统在不修改工厂角色的情况下引进新产品。

​	比如创建一个课程工厂接口，然后使用java课程实现这个课程工厂接口，使用这个专门的java课程工厂来创建java课程，如果以后开设了另外一门课叫python课程，我们只要定义一个python课程工厂实现这个课程工厂接口即可。



### 简单工厂模式：

简单工厂模式适用于工厂类负责创建的对象较少的情况。且客户端只需要传入工厂类的参数，对于如何创建对象不需要关心。

客户端可以免除直接创建产品对象的责任，而仅仅“消费”产品

创建对象的过程并不复杂，但从代码设计的角度来讲不易于拓展。

```java
public class CourseFactory {
    public ICourse create(Class<? extends ICourse> clazz) {
        try {
            if (null != clazz) {
                return clazz.newInstance();
            }
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }
        return null;
    }
}
```

```java
CourseFactory courseFactory = new CourseFactory();
ICourse pythonCourse = courseFactory.create(PythonCourse.class);
pythonCourse.record();
```



### 工厂方法模式：

工厂方法模式是指定义一个创建对象的接口，但让实现这个接口的类来决定实例化哪个类，工厂方法模式让类的实例化推迟到了子类中进行。

工厂方法模式主要解决产品的拓展问题。



> 核心的工厂类不再负责所有产品的创建，而是将具体创建工作交给子类去做。这个核心类仅仅负责给出具体工厂必须实现的接口，而不负责哪一个产品类被实例化这种细节，这使得工厂方法模式可以允许系统在不修改工厂角色的情况下引进新产品。

![../_images/loger.jpg](https://cdn.jsdelivr.net/gh/dlagez/img@master/loger.jpg)



相比于简单工厂，随着产品链的丰富，如果每个课程的创建逻辑有区别，则工厂的职责会变得越来越多，有点像万能工厂，不便于维护，根据单一职责原则，我们应该将职能拆分，专人干专事。

首先定义一个接口，用来表示工厂类的方法：

```java
public interface ICourseFactory {
    ICourse create();
}
```

创建java工厂，并实现工厂接口。

```java
public class JavaCourseFactory implements ICourseFactory{
    @Override
    public ICourse create() {
        return new JavaCourse();
    }
}
```

创建java课程类时直接使用工厂类创建即可。

```java
ICourseFactory courseFactory = new JavaCourseFactory();
```

工厂方法模式适用于以下场景：

- 创建对象需要大量重复的代码
- 客户端不依赖于产品类实例如何被创建、如何被实现等。
- 一个类通过其子类来指定创建哪个对象。

### 抽象工厂模式：

是指提供一个创建一系列相关或者互相依赖对象的接口，无需制定他们的具体实现。

适用于一系列相关的产品对象，比如一个课程里面有视频，笔记对象等等。



创建视频和笔记接口

```java
public interface INote {
    void edit();
}
public interface IVideo {
    void record();
}
```



创建课程工厂

```java
public interface CourseFactory {
    INote createNote();

    IVideo createVideo();
}
```

实现接口，用来给工厂返回创建了的对象

```java
public class JavaNote implements INote{
    @Override
    public void edit() {
        System.out.println("编写java笔记");
    }
}

public class JavaVideo implements IVideo{

    @Override
    public void record() {
        System.out.println("录制java视频");
    }
}
```



最后实现工厂类：

```java
public class JavaCourseFactory implements CourseFactory{
    @Override
    public INote createNote() {
        return new JavaNote();
    }

    @Override
    public IVideo createVideo() {
        return new JavaVideo();
    }
}
```



优缺点：规定了所有可能被创建的产品集合，产品族中拓展新产品比较困难，需要修改抽象工厂的接口。



ref：

- https://design-patterns.readthedocs.io/zh_CN/latest/creational_patterns/factory_method.html
- 《Spring 5 核心原理》



### 单例模式



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