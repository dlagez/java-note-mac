面试设计模式系列之工厂模式



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