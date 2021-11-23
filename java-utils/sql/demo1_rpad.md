ref:`https://www.sqlservercurry.com/2011/10/sql-server-lpad-and-rpad-functions.html`

### sqlserver 实现rpad

返回str，用pad右填充，长度为len。

```sql
> SELECT rpad('hi', 5, 'ab');
 hiaba
> SELECT rpad('hi', 1, '??');
 h
> SELECT rpad('hi', 5);
 hi
```

替换方法: 

```
LEFT(CAST('wbs_ky' as VARCHAR(10)) + REPLICATE('0', 16), 16)
```

其中用到了

### cast

ref: https://docs.microsoft.com/en-us/azure/databricks/spark/latest/spark-sql/language-manual/functions/cast

将第一个参数转换成第二个参数的类型

```
> SELECT cast('10' as int);
 10
```

### RIGHT [Link](https://docs.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/right)

Returns the rightmost len characters from the string str.

```
> SELECT right('Spark SQL', 3);
 SQL
```

返回最右边的数.

### REPLICATE

```
Select Replicate('abc',2) 
abcabc
```

重复第一个参数
