static

用于内存管理，用来共享给定类的相同变量或方法，可以使用static修饰variables, methods, blocks, and nested classes。

静态关键字属于类，而不是类的实例。静态关键字用于一个常量变量或类的每个实例都相同的方法。

当一个类的成员使用static关键字修饰，不需要定义任何对象即可使用该成员。

```java
// Java program to demonstrate that a static member
// can be accessed before instantiating a class

class Test
{
	// static method
	static void m1()
	{
		System.out.println("from m1");
	}

	public static void main(String[] args)
	{
		// calling m1 without creating
		// any object of class Test
		m1();
	}
}
```

