ref：`https://redis.io/topics/data-types-intro`

## 数据类型：

 The following is the list of all the data structures supported by Redis

- Binary-safe strings.  字符串
- Lists: collections of string elements sorted according to the order of insertion. They are basically *linked lists*.  列表
- Sets: collections of unique, unsorted string elements.  集合
- Sorted sets, similar to Sets but where every string element is associated to a floating number value, called *score*. The elements are always taken sorted by their score, so unlike Sets it is possible to retrieve a range of elements (for example you may ask: give me the top 10, or the bottom 10).  有序集合
- Hashes, which are maps composed of fields associated with values. Both the field and the value are strings. This is very similar to Ruby or Python hashes. 哈希，散列
- Bit arrays (or simply bitmaps): it is possible, using special commands, to handle String values like an array of bits: you can set and clear individual bits, count all the bits set to 1, find the first set or unset bit, and so forth.
- HyperLogLogs: this is a probabilistic data structure which is used in order to estimate the cardinality of a set. Don't be scared, it is simpler than it seems... See later in the HyperLogLog section of this tutorial.
- Streams: append-only collections of map-like entries that provide an abstract log data type. They are covered in depth in the [Introduction to Redis Streams](https://redis.io/topics/streams-intro).

### 对于redis的键：

- 很长的键值不是一个好的主意：浪费内存，进行键值比较时浪费时间。
- 太短的键值也不好，
- Try to stick with a schema. For instance "object-type:id" is a good idea, as in "user:1000". 设置键的模式

### Redis Strings

type redis-cli 即可进入互动

```
set roc handsome
get roc
set roc handsome2  # 再次执行会覆盖上面的值

set counter 100  # 即使是字符串
incr counter  # 使用incr命令可以将字符串解析为整数，并加一

> mset a 10 b 20 c 30  # 再一个命令可以设置多个值
OK
> mget a b c
1) "10"
2) "20"
3) "30"

exists roc  # 查询键是否存在
del roc  # 删除键和值
type roc  # 查询键对应值的类型
expire roc 5  # 设置指定的键多少秒后过期
ttl roc  # 查询键值s
```

