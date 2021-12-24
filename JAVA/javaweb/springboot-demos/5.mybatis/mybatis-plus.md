使用：

### 1.引入依赖：

```xml
<!-- 实现对数据库连接池的自动化配置 -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-jdbc</artifactId>
</dependency>
<dependency> <!-- 本示例，我们使用 MySQL -->
    <groupId>mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
</dependency>

<!-- 实现对 MyBatis Plus 的自动化配置 -->
<dependency>
    <groupId>com.baomidou</groupId>
    <artifactId>mybatis-plus-boot-starter</artifactId>
    <version>3.2.0</version>
</dependency>
```

### 2.扫描mapper所在的包

因为不是springboot官方实现的

```java
@SpringBootApplication
@MapperScan(basePackages = "com.example.mybatisplus.mapper")
public class MybatisPlusApplication {

    public static void main(String[] args) {
        SpringApplication.run(MybatisPlusApplication.class, args);
    }

}
```

### 3.配置文件的设置

- 相比 `mybatis` 配置项来说，`mybatis-plus` 增加了更多配置项，也因此我们无需在配置 [`mybatis-config.xml`](https://github.com/YunaiV/SpringBoot-Labs/blob/master/lab-12-mybatis/lab-12-mybatis-xml/src/main/resources/mybatis-config.xml) 配置文件。

```yml
spring:
  # datasource 数据源配置内容
  datasource:
    url: jdbc:mysql:///test2?useSSL=false&useUnicode=true&characterEncoding=UTF-8
    driver-class-name: com.mysql.cj.jdbc.Driver
    username: root
    password: password

# mybatis-plus 配置内容
mybatis-plus:
  configuration:
    map-underscore-to-camel-case: true # 虽然默认为 true ，但是还是显示去指定下。
  global-config:
    db-config:
      id-type: auto # ID 主键自增
      logic-delete-value: 1 # 逻辑已删除值(默认为 1)
      logic-not-delete-value: 0 # 逻辑未删除值(默认为 0)
  mapper-locations: classpath*:mapper/*.xml
  type-aliases-package: com.example.mybatisplus.dataobject

logging:
  level:
    com:
      example:
        mybatisplus:
          mapper: debug
```

### 4.实体类：

 [`@TableName`](https://mybatis.plus/guide/annotation.html#tablename) 注解，设置了 UserDO 对应的表名是 `users` 。毕竟，我们要使用 MyBatis-Plus 给咱自动生成 CRUD 操作。

增加了 `deleted` 字段，并添加了 [`@TableLogic`](https://mybatis.plus/guide/annotation.html#tablelogic) 注解，设置该字段为逻辑删除的标记。

```java三
package com.example.mybatisplus.dataobject;

import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;

import java.util.Date;

@TableName(value = "users") //设置了 UserDO 对应的表名是 users
public class UserDo {

    private Integer id;

    private String username;

    private String password;

    private Date createTime;

    @TableLogic
    private Integer deleted;

    getter、setter、construct、toString.....

}
```

建表

```sql
CREATE TABLE `users` (  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '用户编号',  `username` varchar(64) COLLATE utf8mb4_bin DEFAULT NULL COMMENT '账号',  `password` varchar(32) COLLATE utf8mb4_bin DEFAULT NULL COMMENT '密码',  `create_time` datetime DEFAULT NULL COMMENT '创建时间',  `deleted` bit(1) DEFAULT NULL COMMENT '是否删除。0-未删除；1-删除',  PRIMARY KEY (`id`),  UNIQUE KEY `idx_username` (`username`)) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
```

### 5 UserMapper

继承了 `com.baomidou.mybatisplus.core.mapper.BaseMapper<T>` 接口，这样常规的 CRUD 操作，MyBatis-Plus 就可以替我们自动生成。

```java
@Repository
public interface UserMapper extends BaseMapper<UserDo> {

    default UserDo selectByUsername(@Param("username") String name) {
        return selectOne(new QueryWrapper<UserDo>().eq("username", name));
    }

    List<UserDo> selectByIds(@Param("ids") Collection<Integer> ids);

    default IPage<UserDo> selectPageByCreateTime(IPage<UserDo> page, @Param("createTime")Date createTime) {
        return selectPage(page, new QueryWrapper<UserDo>().gt("create_time", createTime));
    }
}
```

在 [`resources/mapper`](https://github.com/YunaiV/SpringBoot-Labs/tree/master/lab-12-mybatis/lab-12-mybatis-plus/src/main/resources/mapper) 路径下，创建 [`UserMapper.xml`](https://github.com/YunaiV/SpringBoot-Labs/blob/master/lab-12-mybatis/lab-12-mybatis-plus/src/main/resources/mapper/UserMapper.xml) 配置文件。代码如下：

因为在yml配置文件中设置了，所以要建在mapper文件夹下

```
mapper-locations: classpath*:mapper/*.xml
```

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.example.mybatisplus.mapper.UserMapper">
    <sql id="FIELDS">
        id, username, password, create_time
    </sql>

    <select id="selectByIds" resultType="UserDo">
        SELECT
            <include refid="FIELDS"/>
        FROM users
        WHERE id IN
            <foreach collection="ids" item="id" separator="," open="(" close=")" index="">
                #{id}
            </foreach>
    </select>
</mapper>
```

重点要看分页插件。