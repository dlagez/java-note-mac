分布式session，使用redis、mongoDB等存储session。

## 使用

首先引入依赖：

```xml
<!-- 实现对 Spring Session 使用 Redis 作为数据源的自动化配置 -->
<dependency>
    <groupId>org.springframework.session</groupId>
    <artifactId>spring-session-data-redis</artifactId>
</dependency>

<!-- 实现对 Spring Data Redis 的自动化配置 -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
    <exclusions>
        <!-- 去掉对 Lettuce 的依赖，因为 Spring Boot 优先使用 Lettuce 作为 Redis 客户端 -->
        <exclusion>
            <groupId>io.lettuce</groupId>
            <artifactId>lettuce-core</artifactId>
        </exclusion>
    </exclusions>
</dependency>
<!-- 引入 Jedis 的依赖，这样 Spring Boot 实现对 Jedis 的自动化配置 -->
<dependency>
    <groupId>redis.clients</groupId>
    <artifactId>jedis</artifactId>
</dependency>
```

设置连接参数：

```yml
spring:
  redis:
    host: 127.0.0.1
    port: 6379
    password:
    database: 0
    timeout: 0
    jedis:
      pool:
        max-active: 8
        max-idle: 8
        min-idle: 0
        max-wait: -1
```

配置 springboot 使用redis

开启自动化配置 Spring Session 使用 Redis 作为数据源，配置Bean是为了使用特定的JSON序列化方式。因为默认情况下采用 Java 自带的序列化方式。

```java
package com.roc.distributedsession.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.serializer.RedisSerializer;
import org.springframework.session.data.redis.config.annotation.web.http.EnableRedisHttpSession;

@Configuration
@EnableRedisHttpSession  // 自动配置 spring session 使用 redis 作为数据源
public class SessionConfiguration {
    @Bean(name = "springSessionDefaultRedisSerializer")
    public RedisSerializer springSessionDefaultRedisSerializer() {
        return RedisSerializer.json();
    }

}
```

已经配置好了，进行简单的测试：

```java
@RestController
@RequestMapping("/session")
public class SessionController {

    @GetMapping("/set")
    public String set(HttpSession session,
                    @RequestParam("key") String key,
                    @RequestParam("value") String value) {
        session.setAttribute(key, value);
        return "access success";
    }

    @GetMapping("/get_all")
    public Map<String, Object> getAll(HttpSession session) {
        Map<String, Object> result = new HashMap<>();

        for (Enumeration<String> enumeration = session.getAttributeNames();
             enumeration.hasMoreElements();
        ) {
            String key = enumeration.nextElement();
            Object value = session.getAttribute(key);
            result.put(key, value);
        }
        return result;
    }
}
```

访问 `"http://127.0.0.1:8080/session/get"` 接口，注意这里没有定义这个接口。这里只是随便使用了一个地址访问了服务器。服务器会产生一个`session`，查看这个`session`。

```bash
127.0.0.1:6379> scan 0
1) "0"
2) 1) "spring:session:sessions:dbb96a11-8a69-44fe-907e-8bb530536f84"
   2) "spring:session:expirations:1637050680000"
   3) "spring:session:sessions:expires:dbb96a11-8a69-44fe-907e-8bb530536f84"
```

我们查看其sessions的结构

```bash
127.0.0.1:6379> HGETALL spring:session:sessions:dbb96a11-8a69-44fe-907e-8bb530536f84
1) "lastAccessedTime"
2) "1637048853592"
3) "maxInactiveInterval"
4) "1800"
5) "creationTime"
6) "1637048790077"
```

现在往`session`里面设置几个 `Attribute`，访问连接即可。

```
http://localhost:8080/session/set?key=roc&value=zhang
```

查看`session`的内容，和上面的`session`不同，上个`session`过期了

```bash
127.0.0.1:6379> HGETALL spring:session:sessions:09128401-94ed-44e3-b7c0-757a07deb636
1) "lastAccessedTime"
2) "1637050999800"
3) "maxInactiveInterval"
4) "1800"
5) "creationTime"
6) "1637050999800"
7) "sessionAttr:roc"
8) "\"zhang\""
```

