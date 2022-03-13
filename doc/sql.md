## cookbook

ref <sql cookbook>

数据库表及其数据：

```sql
DROP TABLE IF EXISTS emp;

CREATE TABLE emp (
  empno decimal(4,0) NOT NULL,
  ename varchar(10) default NULL,
  job varchar(9) default NULL,
  mgr decimal(4,0) default NULL,
  hiredate date default NULL,
  sal decimal(7,2) default NULL,
  comm decimal(7,2) default NULL,
  deptno decimal(2,0) default NULL
);

DROP TABLE IF EXISTS dept;

CREATE TABLE dept (
  deptno decimal(2,0) default NULL,
  dname varchar(14) default NULL,
  loc varchar(13) default NULL
);

INSERT INTO emp VALUES ('7369','SMITH','CLERK','7902','1980-12-17','800.00',NULL,'20');
INSERT INTO emp VALUES ('7499','ALLEN','SALESMAN','7698','1981-02-20','1600.00','300.00','30');
INSERT INTO emp VALUES ('7521','WARD','SALESMAN','7698','1981-02-22','1250.00','500.00','30');
INSERT INTO emp VALUES ('7566','JONES','MANAGER','7839','1981-04-02','2975.00',NULL,'20');
INSERT INTO emp VALUES ('7654','MARTIN','SALESMAN','7698','1981-09-28','1250.00','1400.00','30');
INSERT INTO emp VALUES ('7698','BLAKE','MANAGER','7839','1981-05-01','2850.00',NULL,'30');
INSERT INTO emp VALUES ('7782','CLARK','MANAGER','7839','1981-06-09','2450.00',NULL,'10');
INSERT INTO emp VALUES ('7788','SCOTT','ANALYST','7566','1982-12-09','3000.00',NULL,'20');
INSERT INTO emp VALUES ('7839','KING','PRESIDENT',NULL,'1981-11-17','5000.00',NULL,'10');
INSERT INTO emp VALUES ('7844','TURNER','SALESMAN','7698','1981-09-08','1500.00','0.00','30');
INSERT INTO emp VALUES ('7876','ADAMS','CLERK','7788','1983-01-12','1100.00',NULL,'20');
INSERT INTO emp VALUES ('7900','JAMES','CLERK','7698','1981-12-03','950.00',NULL,'30');
INSERT INTO emp VALUES ('7902','FORD','ANALYST','7566','1981-12-03','3000.00',NULL,'20');
INSERT INTO emp VALUES ('7934','MILLER','CLERK','7782','1982-01-23','1300.00',NULL,'10');

INSERT INTO dept VALUES ('10','ACCOUNTING','NEW YORK');
INSERT INTO dept VALUES ('20','RESEARCH','DALLAS');
INSERT INTO dept VALUES ('30','SALES','CHICAGO');
INSERT INTO dept VALUES ('40','OPERATIONS','BOSTON');
```

### 检索数据：

#### 合并result的列

将表中的两列检索出来，并将检索出来的这两列合并成一列数据。

在`mysql`中可以使用`concat`函数来完成上述要求。

```sql
select concat(ename, ' WORK AS A', job) as msg
from emp
where deptno = 10;
```

结果：可以看到查询出来了一列，这一列有查询的两列数据和一些字符串组成。

![image-20220108144635164](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220108144635164.png)



#### 在查询时使用`if else` 

```sql
select ename, sal,
       case when sal <= 2000 then 'underpaid'
            when sal >= 4000 then 'overpaid'
            else 'OK'
       end as status
from emp
```

结果：需要注意的时，查询语句中的else可以不写，但是when后面没有匹配到的数据就是显示成NULL。

![image-20220108145243585](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220108145243585.png)



#### 限制返回的行数

```sql
select * from emp;
select * from emp limit 5; # 返回前五行
select * from emp limit 5, 2;  # 5表示起始行数，2表示取多少条数据。
```

#### 查询出特定列的值为空的行

```sql
select *
from emp
where comm is null;
```

#### 把空行转化成特定值

```sql
select coalesce(comm, 0) from emp;
```

#### order by 

```sql
select *
from emp
where deptno = 20
order by sal;
```

```sql
select *
from emp
where deptno = 20
order by 3 desc;  # 从左往右数第三个列作为排序的列
```

#### 根据字符串子集排序

比如根据job的最后两个字符排序。

`substr(job, length(job) - 1)`这里需要解释一下，`substr`时字符串截取函数，第一个参数是需要截取的字符串，第二个参数索引，表示从第几个字符开始截取.

比如`job`的第一个`record`是 `CLERK`，那么他的长度是5，`substr`第二个参数就是4，所以从第四个字符开始截取（包含第四个字符），结果就是`RK`，也可以当成固定写法`lenth() - 1`表示截取一个字符串倒数两个字符。

```sql
select ename, job
from emp
order by substr(job, length(job) - 1);
```

#### 排序时处理空值

comm是带有空值的列。

```sql
select ename, comm, sal
from emp
order by 2 desc ;

select ename, comm, sal
from emp
order by comm ;
```

#### 根据条件动态查询

```SQL
select ename, sal, job, comm, case when job='SALESMAN' then comm else sal end as orderd
from emp
order by 5;
```



## postgres

默认用户名：postgres 密码password

进入到plsql数据库：

```
psql -U 用户名 -d 数据库名
```

执行sql文件

```
 \i /temp.sql
```



### 创建数据库

```sql
create database demo
```

### 查询所有数据库

```sql
SELECT datname FROM pg_database
WHERE datistemplate = false;
```

增删改查和sql差不多，这里记笔记的地方就是一些对postgre比较陌生的地方

### 视图 ：VIEW

```sql
CREATE VIEW myview AS
    SELECT name, temp_lo, temp_hi, prcp, date, location
    FROM weather, cities
    WHERE city = name;

SELECT * FROM myview;
```

### Window Functions

聚合函数，类似GROUP BY

```sql
SELECT city, max(temp_hi) 
OVER (PARTITION BY city) 
FROM weather;
```



## redis

ref：`https://redis.io/topics/data-types-intro`

### 数据类型：

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



## sql

### 表操作

查看表结构。

```
DESCRIBE Customers;     # structure of a table
```

### 创建表

### 主键自增

注意，如果中间指定了主键的数，后面自增会

```sql
CREATE TABLE animals (
     id MEDIUMINT NOT NULL AUTO_INCREMENT,
     name CHAR(30) NOT NULL,
     PRIMARY KEY (id)
);
```

包含某一列最大值的行

```sql
SELECT * FROM OrderItems ORDER BY item_price DESC LIMIT 1;
```

使用用户自定义的变量

```sql
SELECT @min_price:=MIN(item_price), @max_price:=MAX(item_price) FROM OrderItems;
SELECT * FROM OrderItems WHERE item_price=@min_price OR item_price=@max_price;
```

主键改变，外键跟着变 

定义外键时加上关键字：ON UPDATE CASCADE

```sql
CREATE TABLE parent (
    id INT NOT NULL,
    PRIMARY KEY (id)
) ENGINE=INNODB;

CREATE TABLE child (
    id INT,
    parent_id INT,
    INDEX par_id (parent_id),
    FOREIGN KEY (parent_id) REFERENCES parent(id) ON UPDATE CASCADE ON DELETE CASCADE
)ENGINE=INNODB;
```

### 更新表：

增加一列

```sql
ALTER TABLE vendors
ADD vend_phone CHAR(20);
```

删除一列

```sql
ALTER TABLE vendors
DROP COLUMN vend_phone;
```

定义外键

```sql
ALTER TABLE orderitems ADD CONSTRAINT fk_orderitems_orders FOREIGN KEY (order_num) REFERENCES orders (order_num);
```



### 检索数据

```sql
SELECT DISTINCT 
       vend_id
FROM
     products
LIMIT 5, 5;
```

### 排序数据

```sql
SELECT prod_id, prod_price, prod_name
FROM products
ORDER BY prod_price DESC, prod_name
LIMIT 1;
```

### 过滤数据

```sql
SELECT prod_name, prod_price
FROM products
WHERE prod_price BETWEEN 5 AND 10;

SELECT prod_name, prod_price
FROM products
WHERE vend_id IN (1002, 1003) AND prod_price >= 10;
```



### 正则表达式

```sql
'1000'                 # 表示查找包含1000的字符串
'[1-4] Ton'            # 匹配几个字符之一，包含 ：4 Ton  的字符串都会被匹配到
'.'                    # 匹配任意的字符
'1000|2000'            # 和or类似
'\\.'                  # 匹配特殊字符
#########重复元字符    
?                      # 使得他前面的字符匹配一次或0次
+                      # 使得他前面的字符匹配一次或多次
*                      # 零或多个匹配
{n}                    # 指定数目的匹配
{n,}                   # 不少于指定数据的匹配
{n, m}                 # 匹配数目的范围 m不超过255
############定位符
^                      # 文本的开始
&                      # 文本的结尾
[[:<:]]                # 词的开始
[[:>:]]                # 词的结尾
```

例子：

```
REGEXP '\\([0-9] sticks?\\)'            # 匹配 TNT (1 stick)   
REGEXP '[[:digit:]]{4}'                 # 匹配连在一起的四个数字
```

### 创建计算字段

```
SELECT prod_id, quantity, item_price, quantity * item_price AS expanded_price
FROM orderitems
WHERE order_num = 20005;
```

### 函数的使用

```sql
############ 字符串处理 ################
CONCAT(vend_name, ' (', vend_country, ')')          # 合并字段  
RTRIM(vend_name)                                    # 删除右侧多余的空格
LTrim()
UPPER(vend_name)                                    # 将字符变成大写
Left()                                              # 返回串左边的字符
Length()
Locate()
Lower()
Right()
Soundex()
SubString()
############ 日期函数 ##############
CurDate()                                            # 返回当前日期
DateDiff()                                           # 计算两个日期之差
```

### 日期处理

```sql
SELECT cust_id, order_num
FROM orders
WHERE DATE(order_date) = '2005-09-01';

SELECT cust_id, order_num
FROM orders
WHERE Date(order_date) BETWEEN '2005-09-01' AND '2005-09-30';

SELECT cust_id, order_num
FROM orders
WHERE YEAR(order_date) = 2005 AND MONTH(order_date) = 9;
```

### 聚集函数和分组

```
AVG()
COUNT()
MAX()
MIN()
SUM()
```

GROUP BY 

- group by 子句可以包含任意数目的列，这使得能对分组进行嵌套
- 如果在group by子句中嵌套了分组，数据将在最后规定的分组上进行汇总
- group by子句中列出的每个列都必须是检索列，或有效的表达式，不能是聚合函数。如果select中使用表达式，则必须在group by子句中指定相同的表达式，不能使用别名。
- 除聚集计算语句外，select语句中的每个列都必须在group by子句中给出。
- group by子句必须出现在where子句之后，order by子句之前。

#### WHRER 和HAVING的区别

where在数据分组前进行过滤，having在数据分组后进行过滤。

```sql
SELECT vend_id, COUNT(*) AS num_prods
FROM products
WHERE prod_price >= 10
GROUP BY vend_id
HAVING COUNT(*) >= 2;
```

```sql
SELECT order_num, SUM(quantity*item_price) AS ordertotal
FROM orderitems
GROUP BY order_num
HAVING SUM(quantity*item_price) >= 50
ORDER BY ordertotal;
```

### 左链接查询

left join会将左边的表所有列作为结果。

```sql
SELECT customers.cust_id, orders.order_num
FROM customers LEFT OUTER JOIN orders ON customers.cust_id = orders.cust_id;
```

```sql
SELECT customers.cust_name,
       customers.cust_id,
       COUNT(orders.order_num) AS num_ord
FROM customers LEFT JOIN orders on customers.cust_id = orders.cust_id
GROUP BY customers.cust_id;
```

### 组合查询

UNOIN的作用于WHERE类似，UNION ALL将不去重。

```sql
SELECT vend_id, prod_id, prod_price
FROM products
WHERE prod_price <= 5
UNION
SELECT vend_id, prod_id, prod_price
FROM products
WHERE vend_id IN (1001, 1002);
```

和下面单条语句结果类似（这里是一样的）

```sql
SELECT vend_id, prod_id, prod_price
FROM products
WHERE prod_price <= 5 OR vend_id IN (1001, 1002);
```

### 全文搜索：

MyISAM支持全文搜索，而InnoDB不支持全文搜索。

Match() 指定被搜索的列，Against() 指定要使用的搜索表达式。Against() 按匹配等级来排序，相关性越高排序越靠前。

```sql
SELECT note_text
FROM productnotes
WHERE MATCH(note_text) AGAINST('rabbit');
```

#### 布尔操作符

AGAINST里面的参数，以-开头的是要排除的选项。*表达多个字符匹配，和正则表达式类似。

```sql
SELECT note_text
FROM productnotes
WHERE MATCH(note_text) AGAINST('heavy -rope*' IN BOOLEAN MODE);
```

### 插入数据

```sql
INSERT INTO customers
    (cust_id, cust_name, cust_address, cust_city, cust_state, cust_zip, cust_country, cust_contact, cust_email) 
VALUES
    (10001, 'Coyote Inc.', '200 Maple Lane', 'Detroit', 'MI', '44444', 'USA', 'Y Lee', 'ylee@coyote.com'),
    (10001, 'Coyote Inc.', '200 Maple Lane', 'Detroit', 'MI', '44444', 'USA', 'Y Lee', 'ylee@coyote.com');
```

### 更新数据

```sql
UPDATE customers
SET cust_name = 'The Fudds',
    cust_email = 'roczhang@outlook.com'
WHERE cust_id = 10005;
```

### 删除数据

```sql
DELETE FROM customers
WHERE cust_id = 10001;
```

注意删除和更新数据要格外小心，不带 where 子句的语句将会作用于整个表。

### 重命名表

```sql
rename table customer2 to customer
```



### 视图

试图不包含数据，它就像是一个函数，每个使用视图的时候它都会执行一次sql语句。

创建视图，它就相当于一个临时的表

```sql
CREATE VIEW productcustomers AS
SELECT cust_name, cust_contact, prod_id
FROM customers, orders, orderitems
WHERE customers.cust_id = orders.cust_id AND orderitems.order_num = orders.order_num;
```

在这个临时表上可以进行查询操作。

```sql
SELECT cust_name, cust_contact
FROM productcustomers
WHERE prod_id = 'TNT2';
```



### 存储过程

创建、使用、删除

```sql
CREATE PROCEDURE productpricing()
BEGIN
    SELECT AVG(prod_price) AS priceaverage
    FROM products;
end;

CALL productpricing();

DROP PROCEDURE productpricing;
```

带输出参数

```sql
# out 表示从存储过程中传出
# in 同理，传入存储过程中
CREATE PROCEDURE productpricing1(
    OUT pl DECIMAL(8, 2),
    OUT ph DECIMAL(8, 2),
    OUT pa DECIMAL(8, 2)
)
BEGIN
    SELECT MIN(prod_price)
    INTO pl
    FROM products;
    SELECT MAX(prod_price)
    INTO ph
    FROM products;
    SELECT AVG(prod_price)
    INTO pa
    FROM products;
end;

# 这里的参数是输出的参数，作为变量可以查询。
CALL productpricing1(@pricelow,
                    @pricehigh,
                    @priceaverage);

SELECT @priceaverage
```

带输入输出参数

```sql
CREATE PROCEDURE ordertotal(
    IN onumber INT,
    OUT ototal DECIMAL(8, 2)
)
BEGIN
    SELECT SUM(item_price * quantity)
    FROM orderitems
    WHERE order_num = onumber
    INTO ototal;
end;

CALL ordertotal(20005, @total);
SELECT @total;
```

复杂的proceduce

```sql
CREATE PROCEDURE ordertotal2(
    IN onumber INT,
    IN taxable BOOLEAN,
    OUT ototal DECIMAL(8, 2)
) COMMENT 'Obtain order total, optionally adding tax'
BEGIN
    DECLARE total DECIMAL(8, 2);
    DECLARE taxrate INT DEFAULT 6;
    SELECT SUM(item_price * quantity)
    FROM orderitems
    WHERE order_num = onumber
    INTO total;
    -- 把营业税增加到合计
    IF taxable THEN
        SELECT total + (total/100 * taxrate) INTO total;
    end if;

    SELECT total INTO ototal;
end;

# 将boolean设置为false，不带税 0为false 1为true
CALL ordertotal2(20005, 0, @total);
SELECT @total;

# 将boolean设置为true， 带税
CALL ordertotal2(20005, 1, @total);
SELECT @total;
```



### 游标

是一个存储在mysql服务器上的数据库查询，在存储了游标之后，可以根据需要滚动或浏览其中的数据。



### 触发器



### 事务

保证成批的mysql操作要么完全执行，要么完全不执行。

```sql
START TRANSACTION;
DELETE FROM orders;
SELECT * FROM orders;
ROLLBACK; / COMMIT; 回滚或者提交
SELECT * FROM orders;
```

