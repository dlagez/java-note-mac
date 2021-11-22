热部署：

1.添加依赖 2.`Build project automatically`

```
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
            <scope>runtime</scope>
            <optional>true</optional>
        </dependency>
```

取值：

`application.properties`

```properties
test.hello=Hello4
```

`：`后面得`Test`是默认值，如果在properties里面没有找到相应得值。就会使用默认值。

```java
@Value("${test.hello:Test}")
private String hello;
```

