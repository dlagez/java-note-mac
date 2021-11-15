## 表操作

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



## 视图

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



## 存储过程

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



## 游标

是一个存储在mysql服务器上的数据库查询，在存储了游标之后，可以根据需要滚动或浏览其中的数据。



## 触发器



## 事务

保证成批的mysql操作要么完全执行，要么完全不执行。

```sql
START TRANSACTION;
DELETE FROM orders;
SELECT * FROM orders;
ROLLBACK; / COMMIT; 回滚或者提交
SELECT * FROM orders;
```

