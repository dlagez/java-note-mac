## 1.使用方法

它是一个条件构造器，用来构建查询的sql语句。

官网的使用介绍：https://mp.baomidou.com/guide/wrapper.html#abstractwrapper

比如

### eq：

这里的eq相当于=号，column eq val

```java
eq(R column, Object val)
eq(boolean condition, R column, Object val)
```

- 等于 =
- 例: `eq("name", "老王")`--->`name = '老王'`

我使用过的实例：首先新建一个QueryWrapper，然后传入条件即可。

```java
@Override
public UmsAdmin getAdminByUsername(String username) {
    UmsAdmin admin = adminCacheService.getAdmin(username);
    if (admin != null) return admin;
    // 如果查询出来的用户有多个
    QueryWrapper<UmsAdmin> wrapper = new QueryWrapper<>();
    wrapper.lambda().eq(UmsAdmin::getUsername, username);
    List<UmsAdmin> list = list(wrapper);
    if (list != null && list.size() > 0) {
        UmsAdmin admin1 = list.get(0);
        adminCacheService.setAdmin(admin1);
        return admin1;
    }
    return null;
}
```



它的类图：



wapper介绍 ：

Wrapper ： 条件构造抽象类，最顶端父类，抽象类中提供4个方法西面贴源码展示
AbstractWrapper ： 用于查询条件封装，生成 sql 的 where 条件
AbstractLambdaWrapper ： Lambda 语法使用 Wrapper统一处理解析 lambda 获取 column。
LambdaQueryWrapper ：看名称也能明白就是用于Lambda语法使用的查询Wrapper
LambdaUpdateWrapper ： Lambda 更新封装Wrapper
QueryWrapper ： Entity 对象封装操作类，不是用lambda语法
UpdateWrapper ： Update 条件封装，用于Entity对象更新操作