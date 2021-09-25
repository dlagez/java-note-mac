和简单工厂模式类似

把工厂抽象了出来，添加了一个工厂接口

```java
package Demo2_FactoryMethodPattern;

public interface ICourseFactory {
    ICourse create();
}
```

java课程实现自己的工厂类。

```java
package Demo2_FactoryMethodPattern;

public class JavaCourseFactory implements ICourseFactory{
    @Override
    public ICourse create() {
        return new JavaCourse();
    }
}
```