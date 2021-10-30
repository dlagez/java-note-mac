### CHARINDEX (Transact-SQL) :[link](https://docs.microsoft.com/en-us/sql/t-sql/functions/charindex-transact-sql?view=sql-server-ver15)

This function searches for one character expression inside a second character expression, returning the starting position of the first expression if found.

```
CHARINDEX ( expressionToFind , expressionToSearch [ , start_location ] )
```

```
USE tempdb;  
GO  
SELECT CHARINDEX ( 'Test',  
       'This is a Test'  
       COLLATE Latin1_General_CS_AS);
```

```
11
```

理解：在第二个参数里面寻找第一个参数，如果存在，则返回在第一个参数的起始位置，索引从0开始。



### SUBSTRING (Transact-SQL): [link](https://docs.microsoft.com/en-us/sql/t-sql/functions/substring-transact-sql?view=sql-server-ver15)

Returns part of a character, binary, text, or image expression in SQL Server.

```
SELECT name, SUBSTRING(name, 1, 1) AS Initial ,
SUBSTRING(name, 3, 2) AS ThirdAndFourthCharacters
FROM sys.databases  
WHERE database_id < 5;
```

| name   | Initial | ThirdAndFourthCharacters |
| :----- | :------ | :----------------------- |
| master | m       | st                       |
| tempdb | t       | mp                       |
| model  | m       | de                       |
| msdb   | m       | db                       |

理解：字符串拆分，第二个参数是起始位置，从1开始，第二个参数是截取字符串的长度。



## 存储过程：

存储过程思想上很简单，就是数据库 SQL 语言层面的代码封装与重用。相当于Java中的方法，传参数进去，执行某种操作。

## SET NOCOUNT ON;

使返回的结果中不包含有关受 Transact-SQL 语句影响的行数的信息。  [参考链接](https://www.cnblogs.com/lmfeng/archive/2011/10/12/2208821.html)

## with as ：

with as 短语，也叫做子查询部分（subquery factoring），主要是定义一个SQL片段，该SQL片段会被整个SQL语句所用到，也有可能在union all的不同部分，作为提供数据的部分。 [参考链接](https://www.cnblogs.com/xmliu/p/7085644.html)

## concat_ws：[官网链接](https://dev.mysql.com/doc/refman/8.0/en/string-functions.html#function_concat-ws)

CONCAT（）函数用于将多个字符串连接成一个字符串。

### 如何指定参数之间的分隔符

使用函数CONCAT_WS（）。使用语法为：CONCAT_WS(separator,str1,str2,…) CONCAT_WS() 代表 CONCAT With Separator ，是CONCAT()的特殊形式。第一个参数是其它参数的分隔符。分隔符的位置放在要连接的两个字符串之间。分隔符可以是一个字符串，也可以是其它参数。

## convert：[官网链接](https://dev.mysql.com/doc/refman/8.0/en/cast-functions.html#function_convert)

在不同字符集之间转换数据。在MySQL中，转码名称与相应的字符集名称相同。例如，该语句将默认字符集中的字符串'abc'转换为utf8mb4字符集中相应的字符串:

CONVERT(expr, type)语法包含一个表达式和一个类型值（用来指定结果的类型），生成指定类型的结果值

## case when：[官网链接](https://dev.mysql.com/doc/refman/8.0/en/case.html)

存储程序的CASE语句实现了一个复杂的条件构造。

下面的第一个是语法，第二个例子，含义：如果when的表达是为真，则执行then的语句，如果when的表达式为假。则执行else语句，如果有else语句的话。注意每个THEN后面的statement_list包含一个或者多个sql表达式。表达式不能为空。（如果else不需要执行语句的话使用BEGIN        END;结尾即可）

```
CASE
    WHEN search_condition THEN statement_list
    [WHEN search_condition THEN statement_list] ...
    [ELSE statement_list]
END CASE
case when ariba_id like '%-%' then SUBSTRING(ariba_id, 0, charindex('-',ariba_id))
```

## UNION：[官网链接](https://dev.mysql.com/doc/refman/8.0/en/union.html)

UNION将多个SELECT语句的结果合并到一个结果集中