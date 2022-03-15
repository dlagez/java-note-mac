引入依赖：

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-jdbc</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>

<dependency>
    <groupId>mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
    <scope>runtime</scope>
</dependency>
```

配置参数：

```yml
spring:
  # datasource ???????
  datasource:
    url: jdbc:mysql:///jpa?useSSL=false&useUnicode=true&characterEncoding=UTF-8
    driver-class-name: com.mysql.jdbc.Driver
    username: root
    password: password
  # JPA ??????? JpaProperties ?
  jpa:
    show-sql: true # ?? SQL ??????????
    # Hibernate ??????? HibernateProperties ?
    hibernate:
      ddl-auto: none
```

> ddl-auto

- `create`：每次运行程序时，都会重新创建表，故而数据会丢失
- `create-drop`：每次运行程序时会先创建表结构，然后待程序结束时清空表
- `upadte`：每次运行程序，没有表时会创建表，如果对象发生改变会更新表结构，原有数据不会清空，只会更新（推荐使用）
- `validate`：运行程序会校验数据与数据库的字段类型是否相同，字段不同会报错
- `none`: 禁用DDL处理

继承CrudRepository即可。

```java
public interface UserRepository01 extends CrudRepository<UserDO, Integer> {
    Page<UserDO> findAll(Pageable pageable);
}
```

分页查询：

```java
@Test
public void testFindPage() {
    System.out.println("begin");
    Pageable pageable = PageRequest.of(0, 5);
    Page<UserDO> page = userRepository01.findAll(pageable);
    System.out.println(page.getTotalElements());
    System.out.println(page.getTotalPages());

    for (UserDO userDO : page.getContent()) {
        System.out.println(userDO.toString());
    }
    System.out.println("end");
}
```



`ref:https://docs.spring.io/spring-data/jpa/docs/2.6.0/reference/html/#repositories`

## 使用细节：

### 有选择地公开 CRUD 方法

 中间存储库接口用`@NoRepositoryBean`. 确保将该注释添加到 Spring Data 不应在运行时为其创建实例的所有存储库接口。

```java
@NoRepositoryBean
interface MyBaseRepository<T, ID> extends Repository<T, ID> {

  Optional<T> findById(ID id);

  <S extends T> S save(S entity);
}

interface UserRepository extends MyBaseRepository<User, Long> {
  User findByEmailAddress(EmailAddress emailAddress);
}
```

### 使用多数据源

`Persoon`使用`@Eneity`进行注释，所以它属于`jpa`，`User`使用`@Document`注释它属于 `Spring Data MongoDB`

```java
interface PersonRepository extends Repository<Person, Long> { … }

@Entity
class Person { … }

interface UserRepository extends Repository<User, Long> { … }

@Document
class User { … }
```

### 分页查询

```java
Page<User> findByLastname(String lastname, Pageable pageable);

Slice<User> findByLastname(String lastname, Pageable pageable);

List<User> findByLastname(String lastname, Sort sort);

List<User> findByLastname(String lastname, Pageable pageable);
```

`page` 查询会计算数据库的总数 使用`count`方法。如果数据库存储的数据量很大，这将非常消耗性能。

`slice` 只会返回下一个 `Slice` 是否可获得，也就是说它不会查询所有的数量，而只会查询下一个页面还有没有数据。相比于查询数据库的总数据量，只查询下个页面可以节约`cpu`的性能。

### 使用排序

```java
Sort sort = Sort.by("firstname").ascending()
  .and(Sort.by("lastname").descending());
```

### 限制查询结果

比如需要查询前十大的数字

```java
User findFirstByOrderByLastnameAsc();

User findTopByOrderByAgeDesc();

Page<User> queryFirst10ByLastname(String lastname, Pageable pageable);

Slice<User> findTop3ByLastname(String lastname, Pageable pageable);

List<User> findFirst10ByLastname(String lastname, Sort sort);

List<User> findTop10ByLastname(String lastname, Pageable pageable);
```

### 处理空值

其实就是两种方法：

- 一种使用`@Nullable`注释，它允许传入空值，允许返回空值。
- 一种使用`Optional<User>`来接收返回值，它允许空值。



```java
package com.acme;                                                       

import org.springframework.lang.Nullable;

interface UserRepository extends Repository<User, Long> {

  User getByEmailAddress(EmailAddress emailAddress);                    

  @Nullable
  User findByEmailAddress(@Nullable EmailAddress emailAdress);          

  Optional<User> findOptionalByEmailAddress(EmailAddress emailAddress); 
}
```

