1.6

下面使用别名是错误的

```sql
SELECT sal as salary, comm as commission
FROM emp
WHERE salary < 5000
```

解决方法

```sql
SELECT * FROM
(
	SELECT sal as salary, comm as commission
	from emp
) as x # 这里的别名不能省略，省略了会报错 Every derived table must have its own alias.
WHERE salary < 5000
```

where子句在select子句之前执行，错误的查询中，where的条件别名`sqlary`还没有被定义。

from子句在where子句之前执行，所以外层的where子句能使用from内部定义的别名。