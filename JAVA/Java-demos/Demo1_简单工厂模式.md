首先定义一个基础的课程接口。所有的课程都需要集成它。

```java
package Demo1_simpleFactoryPattern;

/**
 * 基础的课程类
 */
public interface ICourse {
    // 定义一个方法，让子类去实现。
    public void record();
}
```

java课程类：

```java
package Demo1_simpleFactoryPattern;

/**
 * 基础课程的实现类
 */
public class JavaCourse implements ICourse{
    @Override
    public void record() {
        System.out.println("录制java课程");
    }
}
```

Python课程类：

```java
package Demo1_simpleFactoryPattern;

public class PythonCourse implements ICourse{
    @Override
    public void record() {
        System.out.println("录制Python课程");
    }
}
```

我们来测试一下。

```java
package Demo1_simpleFactoryPattern;

/**
 * 测试类
 */
public class Demo_test {
    public static void main(String[] args) {
        // 普通方法创建课程
        // 首先是这个创建课程的方法是通过new关键字来创建的。
        // 应用层代码需要依赖JavaCourse。
        ICourse course = new JavaCourse();
        course.record();
    }
}
```

在测试类中，我们需要创建一个JavaCourse类，通过new关键字创建的类，把应用层的类和JavaCourse类仅仅的耦合在一起，只要JavaCourse类除了问题，应用层的代码就会直接报错。为了降低这种耦合度。我们使用简单工厂类，通过反射的方式，拿到对象的Class之后，实例化出一个对象。

```java
package Demo1_simpleFactoryPattern;

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

可以看到这个工厂类需要传入一个类的class信息。通过class信息实例化一个类。用到了反射的知识。

在来看测试：

```java
package Demo1_simpleFactoryPattern;

/**
 * 测试类
 */
public class Demo_test {
    public static void main(String[] args) {
        // 普通方法创建课程
        // 首先是这个创建课程的方法是通过new关键字来创建的。
        // 应用层代码需要依赖JavaCourse。
        ICourse course = new JavaCourse();
        course.record();

        // 工厂类创建课程，通过类反射获取对象。降低了耦合度。
        CourseFactory courseFactory = new CourseFactory();
        ICourse pythonCourse = courseFactory.create(PythonCourse.class);
        pythonCourse.record();
    }
}
```

