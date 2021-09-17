HashMap：它根据键的hashcode值存储数据，遍历的顺序不确定，他是非线程安全的

ConcurrentHashMap：线程安全。

### 存储结构-字段

从结构实现来讲，hashmap是数组+链表+红黑树实现的。

![img](HashMap.assets/e4a19398.png)

1.hashmap有一个很重要的字段，就是Node[] table 即哈希桶数组，很明显它是一个Node数组。而下面的Node是一个链表。当前节点存储了下一个节点的句柄。

```java
static class Node<K,V> implements Map.Entry<K,V> {
    final int hash;
    final K key;
    V value;
    Node<K,V> next;

    Node(int hash, K key, V value, Node<K,V> next) {
    }

    public final K getKey()        { return key; }
    public final V getValue()      { return value; }
    public final String toString() { return key + "=" + value; }

    public final int hashCode() {
    }

    public final V setValue(V newValue) {
    }

    public final boolean equals(Object o) {...}
}
```

Node是hashmap的一个内部类，实现了Map.Entry接口，本质就是一个映射（键值对），上图的每个黑色圆点就是一个Node对象。

2.hashmap就是用哈希表来存储的，为了解决冲突，hashmap使用了链地址法。简单来说就是数组加链表的组合。每个数组元素上面放一个链表结构，当有hash冲突的时候，就把数据放在对应数组元素的链表上。



#### 字段：

```
int threshold;             // 所能容纳的key-value对极限 
final float loadFactor;    // 负载因子
int modCount;  
int size;  
```

threshold = length * loadFactor 也就是说数组长度固定后，负载因子越大，所能容纳的键值对个数越多。

size是hashmap中实际存在的键值对数量。

哈希桶数组table的长度length大小必须为2的n次方

### 功能实现-方法

jdk 1.8

```
static final int hash(Object key) {
    int h;
    // h = key.hashCode() 第一步 区hashCode值
    // h ^ （h >>> 16） 高位参与运算
    return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
}
```

这里的Hash算法本质上就是三步：**取key的hashCode值、高位运算、取模运算**。

通过hashcode的高16位异或低16位实现的。数组table的length比较小的时候，也能保证考虑到高低Bit都参与到Hash的计算中，同时不会有太大的开销。

![img](HashMap.assets/45205ec2.png)

#### 分析HashMap的put方法

![img](HashMap.assets/d669d29c.png)

```java
final V putVal(int hash, K key, V value, boolean onlyIfAbsent,
               boolean evict) {
    Node<K,V>[] tab; Node<K,V> p; int n, i;
  // table为空则创建  
  if ((tab = table) == null || (n = tab.length) == 0)
        n = (tab = resize()).length;
  // 计算index 并对null做处理  
  if ((p = tab[i = (n - 1) & hash]) == null)
        tab[i] = newNode(hash, key, value, null);
    else {
        Node<K,V> e; K k;
        //  节点key存在，直接覆盖value
        if (p.hash == hash &&
            ((k = p.key) == key || (key != null && key.equals(k))))
            e = p;
       // 判断该节点位红黑树
        else if (p instanceof TreeNode)
            e = ((TreeNode<K,V>)p).putTreeVal(this, tab, hash, key, value);
      // 该链为链表  
      else {
            for (int binCount = 0; ; ++binCount) {
                if ((e = p.next) == null) {
                    p.next = newNode(hash, key, value, null);
                  // 链表长度大于8转换为红黑树进行处理  
                  if (binCount >= TREEIFY_THRESHOLD - 1) // -1 for 1st
                        treeifyBin(tab, hash);
                    break;
                }
              // key已经存在直接覆盖value
                if (e.hash == hash &&
                    ((k = e.key) == key || (key != null && key.equals(k))))
                    break;
                p = e;
            }
        }
        if (e != null) { // existing mapping for key
            V oldValue = e.value;
            if (!onlyIfAbsent || oldValue == null)
                e.value = value;
            afterNodeAccess(e);
            return oldValue;
        }
    }
    ++modCount;
  // 超过最大容量就扩容
    if (++size > threshold)
        resize();
    afterNodeInsertion(evict);
    return null;
}
```

### 扩容机制

```java
void resize(int newCapacity) {   //传入新的容量
  Entry[] oldTable = table;    //引用扩容前的Entry数组
  int oldCapacity = oldTable.length;         
  if (oldCapacity == MAXIMUM_CAPACITY) {  //扩容前的数组大小如果已经达到最大(2^30)了
    threshold = Integer.MAX_VALUE; //修改阈值为int的最大值(2^31-1)，这样以后就不会扩容了
    return;
  }

  Entry[] newTable = new Entry[newCapacity];  //初始化一个新的Entry数组
  transfer(newTable);                         //！！将数据转移到新的Entry数组里
  table = newTable;                           //HashMap的table属性引用新的Entry数组
  threshold = (int)(newCapacity * loadFactor);//修改阈值
}
```

jdk1.8

```java
final Node<K,V>[] resize() {
    Node<K,V>[] oldTab = table;
    int oldCap = (oldTab == null) ? 0 : oldTab.length;
    int oldThr = threshold;
    int newCap, newThr = 0;
    if (oldCap > 0) {
        // 如果容量大于最大容量就不再扩充了，就只能随便你碰撞去了
        if (oldCap >= MAXIMUM_CAPACITY) {
            threshold = Integer.MAX_VALUE;
            return oldTab;
        }
      // 没有超过
        else if ((newCap = oldCap << 1) < MAXIMUM_CAPACITY &&
                 oldCap >= DEFAULT_INITIAL_CAPACITY)
          // 新的容量是旧容量的两倍  
          newThr = oldThr << 1; // double threshold
    }
    else if (oldThr > 0) // initial capacity was placed in threshold
      newCap = oldThr;
    else {               // zero initial threshold signifies using defaults
        newCap = DEFAULT_INITIAL_CAPACITY;
        newThr = (int)(DEFAULT_LOAD_FACTOR * DEFAULT_INITIAL_CAPACITY);
    }
    // 计算信息的resize上限
    if (newThr == 0) {
        float ft = (float)newCap * loadFactor;
        newThr = (newCap < MAXIMUM_CAPACITY && ft < (float)MAXIMUM_CAPACITY ?
                  (int)ft : Integer.MAX_VALUE);
    }
    threshold = newThr;
    @SuppressWarnings({"rawtypes","unchecked"})
    Node<K,V>[] newTab = (Node<K,V>[])new Node[newCap];
    table = newTab;
    if (oldTab != null) {
        for (int j = 0; j < oldCap; ++j) {
            Node<K,V> e;
            if ((e = oldTab[j]) != null) {
                oldTab[j] = null;
                if (e.next == null)
                    newTab[e.hash & (newCap - 1)] = e;
                else if (e instanceof TreeNode)
                    ((TreeNode<K,V>)e).split(this, newTab, j, oldCap);
                else { // 链表优化重hash的代码块
                    Node<K,V> loHead = null, loTail = null;
                    Node<K,V> hiHead = null, hiTail = null;
                    Node<K,V> next;
                    do {
                        next = e.next;
                      // 原索引
                        if ((e.hash & oldCap) == 0) {
                            if (loTail == null)
                                loHead = e;
                            else
                                loTail.next = e;
                            loTail = e;
                        }
                      // 源索引+oldCap·
                        else {
                            if (hiTail == null)
                                hiHead = e;
                            else
                                hiTail.next = e;
                            hiTail = e;
                        }
                    } while ((e = next) != null);
                    if (loTail != null) {
                        loTail.next = null;
                        newTab[j] = loHead;
                    }
                    if (hiTail != null) {
                        hiTail.next = null;
                        newTab[j + oldCap] = hiHead;
                    }
                }
            }
        }
    }
    return newTab;
}
```

在扩容的时候，在旧数组中同一调Entry链上的元素，通过重新计算索引位置后，有可能被放到了新数组的不同位置上。

![img](HashMap.assets/b2330062.png)

我们使用的是2次幂的扩展(指长度扩为原来2倍)，元素的位置要么是在原位置，要么是在原位置再移动2次幂的位置。

![img](HashMap.assets/4d8022db.png)

元素在重新计算hash之后，因为n变为2倍，那么n-1的mask范围在高位多1bit(红色)，因此新的index就会发生这样的变化：

![img](HashMap.assets/d773f86e.png)

因此，我们在扩充HashMap的时候，不需要像JDK1.7的实现那样重新计算hash，只需要看看原来的hash值新增的那个bit是1还是0就好了，是0的话索引没变，是1的话索引变成“原索引+oldCap”，可以看看下图为16扩充为32的resize示意图：

![img](HashMap.assets/3cc9813a.png)

这样设计非常的巧妙，

- 省去了重新计算hash值的时间
- 新增的1bit是0还是1是随机的，所以在resize的过程中，均匀的把之前的冲突节点分散开了。
- resize之后链表不会倒置。

## 小结

1. 扩容是一个特别耗性能的操作，所以当程序员在使用HashMap的时候，估算map的大小，初始化的时候给一个大致的数值，避免map进行频繁的扩容。
2. 负载因子是可以修改的，也可以大于1，但是建议不要轻易修改，除非情况非常特殊。
3. HashMap是线程不安全的，不要在并发的环境中同时操作HashMap，建议使用ConcurrentHashMap。
4. JDK1.8引入红黑树大程度优化了HashMap的性能。
5. 还没升级JDK1.8的，现在开始升级吧。HashMap的性能提升仅仅是JDK1.8的冰山一角。