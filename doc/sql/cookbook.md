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

