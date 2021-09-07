

## MyBatis + XML

https://github.com/dlagez/dlagez-springboot-demos/tree/master/mybatis

### 1.引入依赖

在 pom.xml 文件中，引入相关依赖。

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.1.3.RELEASE</version>
        <relativePath/> <!-- lookup parent from repository -->
    </parent>
    <modelVersion>4.0.0</modelVersion>

    <artifactId>lab-12-mybatis</artifactId>

    <dependencies>
        <!-- 实现对数据库连接池的自动化配置 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-jdbc</artifactId>
        </dependency>
        <dependency> <!-- 本示例，我们使用 MySQL -->
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
            <version>5.1.48</version>
        </dependency>

        <!-- 实现对 MyBatis 的自动化配置 -->
        <dependency>
            <groupId>org.mybatis.spring.boot</groupId>
            <artifactId>mybatis-spring-boot-starter</artifactId>
            <version>2.1.1</version>
        </dependency>

        <!-- 方便等会写单元测试 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>

    </dependencies>

</project>
```

### 2.Application

创建 [`Application.java`](https://github.com/YunaiV/SpringBoot-Labs/blob/master/lab-12-mybatis/lab-12-mybatis-xml/src/main/java/cn/iocoder/springboot/lab12/mybatis/Application.java) 类，配置 `@MapperScan` 注解，扫描对应 Mapper 接口所在的包路径。代码如下：



```java
@SpringBootApplication
@MapperScan(basePackages = "cn.iocoder.springboot.lab12.mybatis.mapper")
public class Application {
}
```

因为这里是做示例。实际项目中，可以考虑创建一个 MyBatisConfig 配置类，将 `@MapperScan` 注解添加到其上。

### 3.应用配置文件

```yml
spring:
  datasource:
    url: jdbc:mysql:///mybatis?useSSL=false&useUnicode=true&characterEncoding=UTF-8
    driver-class-name: com.mysql.cj.jdbc.Driver
    username: root
    password: password

mybatis:
  config-location: classpath:mybatis-config.xml # 配置 MyBatis 配置文件路径
  mapper-locations: classpath:mapper/*.xml # 配置 Mapper XML 地址
  type-aliases-package: com.example.mybatis.pojo # 配置数据库实体包路径
```

### 4 MyBatis 配置文件

在 [`resources`](https://github.com/YunaiV/SpringBoot-Labs/tree/master/lab-12-mybatis/lab-12-mybatis-xml/src/main/resources) 目录下，创建 [`mybatis-config.xml`](https://github.com/YunaiV/SpringBoot-Labs/blob/master/lab-12-mybatis/lab-12-mybatis-xml/src/main/resources/mybatis-config.xml) 配置文件。配置如下：

```java
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE configuration PUBLIC "-//mybatis.org//DTD Config 3.0//EN" "http://mybatis.org/dtd/mybatis-3-config.dtd">
<configuration>

    <settings>
        <!-- 使用驼峰命名法转换字段。 -->
        <setting name="mapUnderscoreToCamelCase" value="true"/>
          <!--    打印sql语句    -->
        <setting name="logImpl" value="STDOUT_LOGGING"></setting>
    </settings>

    <typeAliases>
        <typeAlias alias="Integer" type="java.lang.Integer"/>
        <typeAlias alias="Long" type="java.lang.Long"/>
        <typeAlias alias="HashMap" type="java.util.HashMap"/>
        <typeAlias alias="LinkedHashMap" type="java.util.LinkedHashMap"/>
        <typeAlias alias="ArrayList" type="java.util.ArrayList"/>
        <typeAlias alias="LinkedList" type="java.util.LinkedList"/>
    </typeAliases>

</configuration>
```

因为在数据库中的表的字段，我们是使用下划线风格，而数据库实体的字段使用驼峰风格，所以通过 `mapUnderscoreToCamelCase = true` 来自动转换。

### 5.UserDO

在pojo文件夹下面创建pojo类，与数据库的表相对应。

```java
public class SysUser {
    private Long id;
    private String userName;
    private String userPassword;
    private String userEmail;
    private String userInfo;
    private byte[] headImg;
    private Date createTime;
```

### 6.UserMapper.java

创建mapper接口，在resource下创建mapper文件夹和mapper接口对应，UserMapper.java 对应UserMapper.xml

他是这样对应的。我们配置文件设置了xml的文件夹存放地址。 mapper-locations: classpath:mapper/*.xml

说明在resouces目录下mapper文件夹存放了xml文件

```java
@Repository
public interface UserMapper {
	SysUser selectById(Long id);
}
```

### 7.在resources/mapper路径下，创建UserMapper.xml配置文件

注意namespace属性。当 app 接口和 XM 文件关联的时候，命名空间口amespace 值就需要配置成接口的全限定名称

例如：我的接口名为UserMapper，在相应的namespace里面需要使用 namespace="com.example.mybatis.mapper.UserMapper"

接口和xml文件就是这样关联起来的。

里面的方法是怎么关联的：通过id关联，名字相同就关联

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.example.mybatis.mapper.UserMapper">
    <select id="selectById" resultType="sysUser">
        select * from sys_user where id = #{id}
    </select>
</mapper>
```

到现在为止，mybatis的配置已经完成，运行测试即可。

### 注意

返回结果有多个的时候，使用List来存储他们

```
List<SysUser> selectAll();
```



### Select技巧

多表查询：通过用户名来查询角色信息。设计到三个表，用户表、角色表、用户角色表。

```sql
<select id="selectRolesByUserId" resultType="sysrole">
    select
        r.id,
        r.role_name,
        r.enabled,
        r.create_by,
        r.create_time
    from sys_user u
    inner join sys_user_role ur on u.id = ur.user_id
    inner join sys_role r on ur.role_id = r.id
    where u.id = #{userId}
</select>
```

Inner join 是链接查询，当user表的id字段和userRole表的user_id字段相等时，他们就是结果。



返回值类型多了怎么办？

上面的要求不变，增加一些需求，比如结果中增加user表的两个字段。

User.name user.info

解决方法：在SysRole类中添加User字段。相当于给SysRole表格添加了用户信息。

```java
public class SysRole {
    private Long id;
    private String roleName;
    private Integer enabled;
    private Long create_by;
    private Date createTime;

    private SysUser sysUser;

    public SysUser getSysUser() {
        return sysUser;
    }

    public void setSysUser(SysUser sysUser) {
        this.sysUser = sysUser;
    }
```

此时在mapper中查询时要怎么设置user的数据呢。

```xml
<select id="selectRolesByUserId" resultType="sysrole">
    select
        r.id,
        r.role_name,
        r.enabled,
        r.create_by,
        r.create_time,
        u.user_name as "user.userName",
        u.user_email as "user.userEmail"
    from sys_user u
    inner join sys_user_role ur on u.id = ur.user_id
    inner join sys_role r on ur.role_id = r.id
    where u.id = #{userId}
</select>
```

设置别名的时候，使用的是"user.属性名", user是SysRole中的属性。userName和userEmail是SysUser对象中的属性。通过这种方式直接将值赋值给user字段中的属性。

### insert插入技巧

接口中定义方法

```java
// insert方法
int insert(SysUser sysUser);
```

在xml中定义对应的sql语句

```sql
<insert id="insert">
    insert into sys_user (id, user_name, user_password, user_email, user_info, head_img, create_time)
    values (#{id}, #{userName}, #{userPassword}, #{userEmail}, #{userInfo}, #{headImg, jdbcType=BLOB},
    #{createTime,jdbcType=TIMESTAMP})
</insert>
```

其中要注意的是：为了防止类型错误，对于一些特殊的数据类型，需要执行数据库中对应的数据类型，

#{headImg, jdbcType=BLOB} 指定数据库中的类型为二进制数据流类型。

java中一般使用java.util.Date类型，date、time、datetime对应的数据库类型分别为DATE、TIME、TIMESTAMP

### update技巧

接口中定义方法

```
int updateById(SysUser sysUser);
```

xml中定义sql语句

```sql
<update id="updateById">
    update sys_user
    set user_name = #{userName},
        user_password = #{userPassword},
        user_email = #{userEmail},
        user_info = #{userInfo},
        head_img = #{headImg, jdbcType=BLOB},
        create_time = #{createTime, jdbcType=TIMESTAMP}
    where id = #{id}
</update>
```

测试即可



接口方法中有多个参数

```
List<SysRole> selectRolesByUserIdAndRoleEnabled(Long userId, Integer enabled);
```

xml

```xml
<!--  测试多个参数  -->
<select id="selectRolesByUserIdAndRoleEnabled" resultType="sysrole">
    select r.id, r.role_name, r.enabled, r.create_by, r.create_time
    from sys_user u
    inner join sys_user_role ur on u.id = ur.user_id
    inner join sys_role r on ur.role_id = r.id
    where u.id = #{userId} and r.enabled = #{enabled}
</select>
```

注意：这里参数的取值是通过#{userId}取值。使用接口参数的名字。



## 动态sql

### If用法

if 标签通常用于 WHE 语句中，通过判断参数值来决定是否使用某个查询条件，它也常用于 UPDATE 语句中判断是否更新某 个字段 还可以在 IN SE 语句中用来判断是否插入某个字段的值。

需求：通过名字和邮箱来查找用户，参数可以是一个名字，或者一个邮箱，或者两者都要。

接口方法

```
List<SysUser> selectByUser(SysUser sysUser);
```

xml

```xml
<!--     测试动态sql    -->
<select id="selectByUser" resultType="sysuser">
    select id,user_name, user_password, user_email, user_info, head_img, create_time
    from sys_user
    where 1 = 1
    <if test="userName != null and userName != ''">
        and user_name like concat('%', #{userName}, "%")
    </if>
    <if test="userEmail != null and userEmail != ''">
        and user_email = #{userEmail}
    </if>
</select>
```

if标签有一个必填的属性test，test 的属性值是一个符合 OGNL 要求的判断表达式，表达式的结果可以是 tr ue fals ，除此之外所有的非 值都为 true ，只有 false

- 判断条件 property =nu ll property == null 适用于任何类型的宇段 ，用于判断属性值是否为空。
- 判断条件 property !=null 或  property=='''： 仅适用于 String 类型的宇段用于判断是否为空字符串
- 可以使用and和or连接



需求：通过id来更新用户，传入的参数是一个SysUser，没有值的字段不能进行更新。

接口

```
int updateByIdSelective(SysUser sysUser);
```

xml

```xml
<update id="updateByIdSelective">
    update sys_user
    set
        <if test="userName != null and userName != ''" >
            user_name = #{userName},
        </if>
        <if test="userPassword != null and userPassword != ''" >
            user_password = #{userPassword},
        </if>
        <if test="userEmail != null and userEmail != ''">
            user_email = #{userEmail},
        </if>
        <if test="userInfo != null and userInfo != ''">
            user_info = #{userInfo},
        </if>
        <if test="headImg != null">
            head_img = #{headImg, jdbcType=BLOB},
        </if>
        <if test="createTime != null">
            create_time = #{createTime, jdbcType=TIMESTAMP},
        </if>
        id = #{id}
    where id = #{id}
</update>
```

注意两点：id=#{id} 为什么要这个无用的语句。因为当所有字段为空时

```
update sys_user set where id = #{id}
```

语句是错误的。

如果没有这个条件

```sql
update sys_user set user_name = #{userName}, where id = #{id}
```

这个语句显然也是错误的。



insert使用动态sql

注意，在into语句后面使用if也要在values里面使用if

```sql
insert into sys_user(
	user_name, user_password,
  <if test="userEmail != null and userEmail != ''">
  	user_email,
  </if>
  user_info, head_img, create_time
)
values (
	#{username}, #{userPassword},
  <if test="userEmail != null and userEmail != ''">
  	#{userEmail},
  </if>
  #{userInfo}, #{headImg, jdbcType=BLOB},
  #{createTime, jdbcType=TIMESTAMP}
)
```

### choose用法

上一节的 if 标签提供了基本的条件判断，但是它无法实现 if. . . else if ... else ... 的逻辑，要想实现这样的逻辑，就需要用到 choose when otherwise 标签。 choose 元素中包含 when和 otherwise 两个标签，一个 choose 中至少有一个when，有0个或者1个otherwise。

要求：：当参数 id 有值的时候优先使id 查询，当 id 没有值时就去判断用户名是否有值，如果有值就用用户名查询 ，如果用 户名也没有值，就使 sql查询无结果

接口

```java
SysUser selectByIdOrUserName(SysUser sysUser);
```

xml

```xml
<select id="selectByIdOrUserName" resultType="sysuser">
    select id, user_name, user_password, user_email, user_info, head_img, create_time
    from sys_user
    where 1 = 1
    <choose>
        <when test="id != null">
            and id = #{id}
        </when>
        <when test="userName != null and userName != ''">
            and user_name = #{userName}
        </when>
        <otherwise>
            and 1 = 2
        </otherwise>
    </choose>
</select>
```

注意：这里的when成立时，后面的when就没有用了。

### where set trim用法

where 标签的作用：如果该标签包含的元素中有返回值，就插入一个 where ；如果 where后面 字符串是以 AND OR 开头的，就将它们剔除。

```xml
<select id="selectByUser" resultType="sysuser">
    select id,user_name, user_password, user_email, user_info, head_img, create_time
    from sys_user
    <where>
        <if test="userName != null and userName != ''">
            and user_name like concat('%', #{userName}, "%")
        </if>
        <if test="userEmail != null and userEmail != ''">
            and user_email = #{userEmail}
        </if>
    </where>
</select>
```

不用写 where 1=1了



set 标签的作用：如果该标签包含的元素中有返回值，就插入一个 set ：如果 set 后面的字符串是 以逗号结尾的，就将这个逗号剔除

```xml
<update id="updateByIdSelective">
    update sys_user
    <set>
        <if test="userName != null and userName != ''" >
            user_name = #{userName},
        </if>
        <if test="userPassword != null and userPassword != ''" >
            user_password = #{userPassword},
        </if>
        <if test="userEmail != null and userEmail != ''">
            user_email = #{userEmail},
        </if>
        <if test="userInfo != null and userInfo != ''">
            user_info = #{userInfo},
        </if>
        <if test="headImg != null">
            head_img = #{headImg, jdbcType=BLOB},
        </if>
        <if test="createTime != null">
            create_time = #{createTime, jdbcType=TIMESTAMP},
        </if>
        id = #{id}
    where id = #{id}
    </set>
</update>
```



### foreach用法

foreach 包含以下属性。

- collection 必填，值为要选代循环的属性名。这个属性值的情况有很多。
- item ：变量名，值为从法代对象中取出的每一个值。
- index ：索引的属性名，在集合数组情况下值为当前索引值 当选代循环的对象是 Map类型时，这个值为 Map 的key （键值）。
- open：整个循环内容开头的字符串
- close 整个循环内容结尾的字符串。
- separator ：每次循环的分隔符

接口

```java
List<SysUser> selectByIdList(List<Long> idList);
```

xml

```xml
<select id=";selectByIdList" resultType="sysuser">
    select id, user_name, user_password, user_email, user_info, head_img, create_time
    from sys_user
    where id in
    <foreach collection="list" open="(" close=")" separator="," item="id" index="i">
        #{id}
    </foreach>
</select>
```

collection属性值的三种情况：

- 如果传入的参数类型为List时，collection的默认属性值为list,同样可以使用@Param注解自定义keyName
- 如果传入的参数类型为array时，collection的默认属性值为array,同样可以使用@Param注解自定义keyName;
- 如果传入的参数类型为Map时，collection的属性值可为三种情况：（1.遍历map.keys;2.遍历map.values;3.遍历map.entrySet()）

当参数类型为集合的时候，默认会转换为map类型，并添加一个key为collection的值。

如果参数类型时List集合，那么就添加一个key为list的值。

当collection="list"时，就能得到这个集合，并对它进行循环操作。

通俗来讲就是接口的参数类型为list，就会把这个集合变成一个map，这个map的键为list，collection=‘list'就能得到这个集合。

对于open、close、separator。

```sql
<foreach collection="list" open="(" close=")" separator="," item="id" index="i">
    #{id}
</foreach>
```

结果是这样的(1, 2, 3, 4)，遍历出来的东西得用括号括住，所以得加open

### foreach批量新增

接口

```
int insertList(List<SysUser> userList);
```

xml

```xml
<insert id="insertList">
    insert into sys_user(
        user_name, user_password, user_email, user_info, head_img, create_time
    )
    values
        <foreach collection="list" item="user" separator=",">
            (#{user.userName}, #{user.userPassword}, #{user.userEmail}, #{user.userInfo}, #{user.headImg, jdbcType=BLOB},
            #{user.createTime,jdbcType=TIMESTAMP})
        </foreach>
</insert>
```

解析出来的sql语句是这样的

```
insert into sys_user( user_name, user_password, user_email, user_info, head_img, create_time ) values (?, ?, ?, ?, ?, ?) , (?, ?, ?, ?, ?, ?) 
```

### foreach实现动态的update

```java
// foreach实现动态更新
int updateByMap(Map<String, Object> map);
```

xml

```xml
<!--  foreach实现动态更新  -->
<update id="updateByMap">
    update sys_user
    set
    <foreach collection="_parameter" item="val" index="key" separator=",">
        ${key} = #{val}
    </foreach>
    where id = #{id}
</update>
```

这里方controller方法

```java
@GetMapping("/testUser15")
public void test15() {
    HashMap<String, Object> map = new HashMap<>();
    map.put("user_email", "test@outlook.com");
    map.put("user_password", "password");
    map.put("id", 1003L);
    int i = userMapper.updateByMap(map);
    if (i != 0) {
        System.out.println("update success!");
    }

}
```

这里map的key就是index的值。

解析出来的sql语句就是这样的

```sql
update sys_user set user_email = ? , user_password = ? , id = ? where id = ? 
```

### OGNL用法

常用的表达是如下：

1. e1 or e2
2. e1 and e2
3. e1 == e2 e1 eq e2
4. e1 != e2 e1 neq e2