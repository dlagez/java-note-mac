ArrayList是一个大小可以调整的动态数组。适应查询为主的场景。

它不是一个线程安全的集合。并发修改时可能会抛出ConcurrentModificationException

### 字段

```java
// 序列化版本标识，序列化和反序列化时使用
private static final long serialVersionUID = 8683452581122892189L;
// 默认的数据容量
private static final int DEFAULT_CAPACITY = 10;
//  用于ArrayList空实例的共享空数组实例
private static final Object[] EMPTY_ELEMENTDATA = {};
//  默认大小空间实例的共享空数组实例
private static final Object[] DEFAULTCAPACITY_EMPTY_ELEMENTDATA = {};
// 存放元素的数组 没有设置私有是为了方便内部类访问
transient Object[] elementData; // non-private to simplify nested class access
// 数组元素个数
private int size;
```

### 构造方法

如果可以预估容量，就使用带初始容量的构造方法，可以避免数组扩容带来的性能消耗

```java
public ArrayList(int initialCapacity) {
    if (initialCapacity > 0) {
        this.elementData = new Object[initialCapacity];
    } else if (initialCapacity == 0) {
        this.elementData = EMPTY_ELEMENTDATA;
    } else {
        throw new IllegalArgumentException("Illegal Capacity: "+
                                           initialCapacity);
    }
}
```

如果初始化时没有带初始容量，则会创建一个容量默认为10的ArrayList

```java
public ArrayList() {
    this.elementData = DEFAULTCAPACITY_EMPTY_ELEMENTDATA;
}
```

还可以从其他数据结构转换。

```java
public ArrayList(Collection<? extends E> c) {
    Object[] a = c.toArray();
    if ((size = a.length) != 0) {
        if (c.getClass() == ArrayList.class) {
            elementData = a;
        } else {
            elementData = Arrays.copyOf(a, size, Object[].class);
        }
    } else {
        // replace with empty array.
        elementData = EMPTY_ELEMENTDATA;
    }
}
```

### 添加

单个添加

```java
public boolean add(E e) {
    modCount++;
    add(e, elementData, size);
    return true;
}
```

扩容机制

```java
private int newCapacity(int minCapacity) {
    // overflow-conscious code
    int oldCapacity = elementData.length;
    // oldCapacity >> 1 除以2 相当于扩容1.5倍。
    int newCapacity = oldCapacity + (oldCapacity >> 1);
    // 防止整型溢出
    if (newCapacity - minCapacity <= 0) {
        if (elementData == DEFAULTCAPACITY_EMPTY_ELEMENTDATA)
          // 如果整型溢出了，就会选择最小容量或者默认容量  
          return Math.max(DEFAULT_CAPACITY, minCapacity);
        if (minCapacity < 0) // overflow
            throw new OutOfMemoryError();
        return minCapacity;
    }
    return (newCapacity - MAX_ARRAY_SIZE <= 0)
        ? newCapacity
        : hugeCapacity(minCapacity);
}
```

批量添加，和单个添加类似，最主要的是扩容机制

```java
public boolean addAll(Collection<? extends E> c) {
  // 先把集合转换成数组  
  Object[] a = c.toArray();
    modCount++;
    int numNew = a.length;
    if (numNew == 0)
        return false;
    Object[] elementData;
    final int s;
    if (numNew > (elementData = this.elementData).length - (s = size))
        elementData = grow(s + numNew);
    System.arraycopy(a, 0, elementData, s, numNew);
    size = s + numNew;
    return true;
}
```

### 排序

```java
public class ArrayList_test {
    public static void main(String[] args) {
        List<String> strList = new ArrayList<String>(4);
        strList.add("1");
        strList.add("2");
        strList.add("3");

        // 可以使用以下三种排序方式
        Collections.sort(strList);
        Collections.sort(strList, String::compareTo);
        strList.sort(String::compareTo);
    }
}
```

