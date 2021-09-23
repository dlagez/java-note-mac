首先看这个类的变量：

```java
    private static final boolean jsfPresent = ClassUtils.isPresent("javax.faces.context.FacesContext", RequestContextHolder.class.getClassLoader());
    private static final ThreadLocal<RequestAttributes> requestAttributesHolder = new NamedThreadLocal("Request attributes");
    private static final ThreadLocal<RequestAttributes> inheritableRequestAttributesHolder = new NamedInheritableThreadLocal("Request context");

```

使用了两个ThreadLocal<RequestAttributes>来存储当前线程下的request。

对于ThreadLocal可以看相应的笔记。

看看这个ThreadLocal类是怎么获取request的。在类中直接调用了ThreadLocal的get()方法，该方法会判断，如果当前线程的ThreadLocalMap存在，就直接使用这个ThreadLocalMap，ThreadLocalMap里面存放着当前线程的信息。如果当前线程的ThreadLocalMap不存在就创建一个新的ThreadLocalMap，并保存。

```java
@Nullable
public static RequestAttributes getRequestAttributes() {
    RequestAttributes attributes = (RequestAttributes)requestAttributesHolder.get();
    if (attributes == null) {
        attributes = (RequestAttributes)inheritableRequestAttributesHolder.get();
    }

    return attributes;
}
```

他会返回一个T对象，就是我们RequestContextHolder的变量ThreadLocal<RequestAttributes>，将RequestAttributes返回。使用ThreadLocal实现线程隔离。

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

### 具体的使用：

在代码中，我们并不会使用RequestAttributes，可以看看下面的RequestAttributes类，里面的方法太少了，我们没有使用框架之间都会使用servlet包下的HttpServletRequest。所以我们要将它转换成ServletRequestAttributes。它是Spring提供操作HttpServletRequest的类。

```java
ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
```

```java
public interface RequestAttributes {
    int SCOPE_REQUEST = 0;
    int SCOPE_SESSION = 1;
    String REFERENCE_REQUEST = "request";
    String REFERENCE_SESSION = "session";

    @Nullable
    Object getAttribute(String var1, int var2);

    void setAttribute(String var1, Object var2, int var3);

    void removeAttribute(String var1, int var2);

    String[] getAttributeNames(int var1);

    void registerDestructionCallback(String var1, Runnable var2, int var3);

    @Nullable
    Object resolveReference(String var1);

    String getSessionId();

    Object getSessionMutex();
}
```

我们来看一下ServletRequestAttributes，在ServletRequestAttributes类里面有一个HttpServletRequest request变量，看到这个变量我们应该熟悉怎么去操作它了。

```java
public class ServletRequestAttributes extends AbstractRequestAttributes {
    public static final String DESTRUCTION_CALLBACK_NAME_PREFIX = ServletRequestAttributes.class.getName() + ".DESTRUCTION_CALLBACK.";
    protected static final Set<Class<?>> immutableValueTypes = new HashSet(16);
    private final HttpServletRequest request;
    @Nullable
    private HttpServletResponse response;
    @Nullable
    private volatile HttpSession session;
    private final Map<String, Object> sessionAttributesToUpdate;
```