默认用户名：postgres 密码password

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

