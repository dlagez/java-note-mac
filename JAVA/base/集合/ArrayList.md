### List扩容实现步骤

总的来说就是分两步：

1、扩容

 把原来的数组复制到另一个内存空间更大的数组中

2、添加元素

 把新元素添加到扩容以后的数组中

扩容速度为1.5倍

```
int newCapacity = oldCapacity + (oldCapacity >> 1);
```

oldCapacity >> 1 右移运算符 原来长度的一半 再加上原长度也就是每次扩容是原来的1.5倍

### 线程安全问题：

```java
private static final long serialVersionUID = 8683452581122892189L;
private static final int DEFAULT_CAPACITY = 10;
private static final Object[] EMPTY_ELEMENTDATA = new Object[0];
private static final Object[] DEFAULTCAPACITY_EMPTY_ELEMENTDATA = new Object[0];
transient Object[] elementData;
private int size;
private static final int MAX_ARRAY_SIZE = 2147483639;
```

ArrayList的实现主要就是用了一个Object的数组，用来保存所有的元素，以及一个size变量用来保存当前数组中已经添加了多少元素。

在线程判断数组容量的方法上没有加锁

```java
public void ensureCapacity(int minCapacity) {
        if (minCapacity > this.elementData.length && (this.elementData != DEFAULTCAPACITY_EMPTY_ELEMENTDATA || minCapacity > 10)) {
            ++this.modCount;
            this.grow(minCapacity);
        }

    }
```

这肯定会由线程安全问题。