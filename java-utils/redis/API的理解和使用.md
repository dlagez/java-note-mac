### 基本命令

- 查看所有的键：keys*  。时间复杂度为O（n），会遍历所有的键
- 键总数：dbsize  。直接获取redis内置的键总数变量 

- 检查键是否存在：exists key
- 删除键：del key。可同时删除多个键
- 键过期：expire key seconds  
- 查看键剩余过期时间：ttl key。-2表示键不存在，-1表示键没有设置过期时间。
- 键的数据结构类型：type key 。

五种：string（字符串），hash（哈希），list（列表），set（集合），zset（有序集合）

### 字符串：

字符串类型是Redis最基础的数据结构。首先键都是字符串类型，而且 其他几种数据结构都是在字符串类型基础上构建的，所以字符串类型能为其 他四种数据结构的学习奠定基础。

- 设置值：set key value [ex seconds] [px milliseconds] [nx|xx]
  1. ·ex seconds：为键设置秒级过期时间。 
  2. ·px milliseconds：为键设置毫秒级过期时间。 
  3. ·nx：键必须不存在，才可以设置成功，用于添加。 
  4. ·xx：与nx相反，键必须存在，才可以设置成功，用于更新。

- 获取值：get key
  1. 不存在则返回nil

- 批量设置值：mset key value [key value...]
- 批量获取值：mget key [key]
- 计数：自增：incr key。自减：decr。自增指定数字：incrby。自减指定数字：decrby
  - 值不是整数：返回错误
  - 值是整数：返回自增后的结果
  - 键不存在，按照值为0自增，返回结果为1

#### 典型的使用场景

1. 缓存功能：Redis+MySql组成
2. 计数：防作弊、按照不同维 度计数，数据持久化到底层数据源等。
3. 共享Session：使用Redis将用户的Session进行集中管理，查询登录信息都直接从Redis中集中获取
4. 限速：为了短信接口不被频繁访问

### 哈希：

在Redis中，哈希类型是指键值本身又是一个键值对结构，

- 设置值：hset key field value

  一个key可以设置多个field

- 获取值；hget key field

- 删除：hdel key field 

- 计算field的个数：hlen key

- 批量获取field-value：hmget key field [field]

- 批量设置field-value：hmset key field value [field value]

- 判断field是否存在：hexists：key field

- 获取所有的field：hkeys key

- 获取所有的value：hvals key

- 获取所有的field-value：hgetall key

使用场景：

- 和数据库表的对应，一个field对应一个字段。

### 列表：

### 集合：

### 有序集合：

### 键管理：

- 键重命名：rename key newkey

  newkey如果已经存在了，那么他的值将会被覆盖。

- renamenx key newkey  如果newkey已经存在，则重命名失败。

#### 键过期：

- ·expire key seconds：键在seconds秒后过期。
- ·expireat key timestamp：键在秒级时间戳timestamp后过期。

```bash
127.0.0.1:6379> expire hello 10
(integer) 1
127.0.0.1:6379> ttl hello
(integer) 4
127.0.0.1:6379> ttl hello
(integer) 0
127.0.0.1:6379> ttl hello
(integer) -2
127.0.0.1:6379>
```

- 如果expire key的键不存在，返回结果为0：
- 如果过期时间为负值，键会立即被删除，犹如使用del命令一样：
- persist命令可以将键的过期时间清除：

### 迁移键：

#### dump+restore

```
dump key
restore key ttl value
```

dump+restore可以实现在不同的Redis实例之间进行数据迁移的功能，整 个迁移的过程分为两步：

 1）在源Redis上，dump命令会将键值序列化，格式采用的是RDB格式。 

2）在目标Redis上，restore命令将上面序列化的值进行复原，其中ttl参 数代表过期时间，如果ttl=0代表没有过期时间。

#### migrate

```
migrate host port key|"" destination-db timeout [copy] [replace] [keys key [key ...]
```

- 第一，整个过程是原子执行的，不需要在多个Redis实例上开启 客户端的，只需要在源Redis上执行migrate命令即可
- 第二，migrate命令的 数据传输直接在源Redis和目标Redis上完成的。
- 第三，目标Redis完成restore 后会发送OK给源Redis，源Redis接收后会根据migrate对应的选项来决定是否 在源Redis上删除对应的键。



1. ·host：目标Redis的IP地址。
2. ·port：目标Redis的端口。
3. ·key|""：如果当前需要迁移多 个键，此处为空字符串""。
4. ·destination-db：目标Redis的数据库索引，例如要迁移到0号数据库，
5. ·timeout：迁移的超时时间（单位为毫秒）。
6. ·[copy]：如果添加此选项，迁移后并不删除源键。
7. ·[replace]：如果添加此选项，migrate不管目标Redis是否存在该键都会 正常迁移进行数据覆盖。
8. ·[keys key[key...]]：迁移多个键，例如要迁移key1、key2、key3，此处填 写“keys key1 key2 key3”。

实例：

```
 migrate 127.0.0.1 6379 hello 0 1000
```

#### 遍历键

scan : 那么每次执行scan，可以想象成只扫描一个字典中的一部分键，直到将 字典中的所有键遍历完毕。scan的使用方法如下：

```
scan cursor [match pattern] [count number]
```

scan 0 会返回这一次遍历的键，并返回下次需要scan的值

```
127.0.0.1:6379> scan 0
1) "6"
2) 1) "w"
2) "i"
3) "e"
4) "x"
5) "j"
6) "q"
7) "y"
8) "u"
9) "b"
10) "o"
```

## 数据库管理

切换数据库：select dbIndex

假设databases=16，select0操作将切换到第一个数据库，select15选择最 后一个数据库，但是0号数据库和15号数据库之间的数据没有任何关联，甚 至可以存在相同的键：



flushdb/flushall命令用于清除数据库，两者的区别的是flushdb只清除当 前数据库，flushall会清除所有数据库。