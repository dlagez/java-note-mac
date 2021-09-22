Spring Security 为我们提供了一套加密规则和密码比对规则

```java
public interface PasswordEncoder {
    // 加密
    String encode(CharSequence var1);
		// 加密前后对比(一般用来比对前端提交过来的密码和数据库存储密码, 也就是明文和密文的对比)
    boolean matches(CharSequence var1, String var2);
		是否需要再次进行编码, 默认不需要
    default boolean upgradeEncoding(String encodedPassword) {
        return false;
    }
}
```

他有很多的实现类，其中下面的实现类使用BCrypt强哈希方法来加密。推荐使用

```java
public class BCryptPasswordEncoder implements PasswordEncoder 
```



