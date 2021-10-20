#### sql

首先这个sql文件里面的代码是一个存储过程。这个存储过程的作用是将联合和转换的数据插入目标表。

首先使用了关键字 with as ：定义了一个SQL片段，该SQL片段会被整个SQL语句所用到，文件中sql片段里面包含了**数据的选择、转换和连接**，也就是包含三个select语句的连接。

- 首先是将数据进行转换和连接。转换函数使用的是convert函数。这个函数的作用是在不同字符集之间转换数据，根据存储过程中的convert函数的语法，CONVERT(expr, type)语法包含一个表达式和一个类型值（用来指定结果的类型），生成指定类型的结果值。
- 这里使用了UNION将多个SELECT语句的结果合并到一个结果集中。使用UNION关键字需要注意UNION 内部的 SELECT 语句必须拥有相同数量的列。
- 用到了case when表达式。

sql文件的最后将插入到ING表中的数据查询了出来。应该用来查看插入到目标表中的数据，这个语句应该和scala代码没有关系。

#### get_pklist_and_col_mapping

首先这个函数的作用应该是定义三个列表，这三个列表里面的数据分别表示：第一个列表应该是选择出提供数据的表的名字（从这个表里面选择数据并转换），第二个列表里面的数据应该是选择列的表达式（定义选择出来的数据列名，并进行数据的转换和过滤），第三个列表应该是包含目标表的名字（将数据插入到这个表里面），它是通过match方法确定的。

get_pklist_and_col_mapping函数里面使用match方法来匹配case。当entity的target_table属性值为对应case就确定了需要选择的列。

每个case里面有三个list列表。

- pkList列表应该是对应三个select目标表。也就是从这三个表格里面选择数据。
- colExpr：定义每一列的表达式，对应sql语句的选择列的语句。
- trgPkList：应该是目标表吧，把数据插入到这些表里面。对应sql语句里面的 INSERT INTO 

#### ing_compass_union_all

这个方法应该是将三个select语句选择出来的数据进行连接。

这个方法传入四个参数，前两个spack，entity应该是操作sql语句的工具，list是表，load_jb_nr是控制条件，它的值为0和其他值的时候执行的语句不同。

这个方法里面定义了四个变量：

- 前三个变量是与表相关的变量，应该就是表的数据。
- 这个主要说一下get_rgn变量， 这个变量表达式应该是对应select语句里面的case when语句，存储程序的CASE语句实现了一个复杂的条件构造。这个get_rgn变量是匹配语句，它与withColumn这个方法联合使用，withColumn方法的作用是通过添加列或替换具有相同名称的现有列来返回新数据集。
- 将三个数据处理完之后，他们选择出来的数据就确定了，然后使用 df1.union(df2).union(df3) 表达式将他们的结果进行连接，对应sql语句的union关键字，将三个select语句选择的数据进行连接。

当load_jb_nr的值不为0的时候，会有一个判断，edl_crt_ts的值要大于等于get_compass_max_edl_ctr_ts函数的输出。

#### get_compass_max_edl_ctr_ts

这个方法应该是选择edl_crt_ts最大值。

#### get_compass_max_load_jb_nr

选择edl_job_sqn_nr的最大值



## 语句分析任务：

分析下面的代码：

```scala
.withColumn("bsn_ar_reorg_cd", expr("if((cc.cc_BUSINESS_AREA is null and io.io_BUSINESS_AREA is null and M.BUSINESS_AREA is null),'BAUnknown',if((cc.cc_BUSINESS_AREA is null and io.io_BUSINESS_AREA is null) ,M.BUSINESS_AREA,if(cc.cc_BUSINESS_AREA is null,io.io_BUSINESS_AREA,cc.cc_BUSINESS_AREA)))"))
```



### expr： [链接](https://sparkbyexamples.com/pyspark/pyspark-sql-expr-expression-function/)

是一个SQL函数，用于执行类似SQL的表达式

### withColumn函数：[链接](https://spark.apache.org/docs/3.2.0/api/scala/org/apache/spark/sql/Dataset.html#withColumn(colName:String,col:org.apache.spark.sql.Column):org.apache.spark.sql.DataFrame)

添加列，或者替换同名的列。

### 语句拆分分析：

提取出其中的sql表达式，就是三个if判断，if条件为true（选定的字段都为空），则使用指定的字段比如'BAUnknown'。

如果非空判断都为false的话。下面的语句会输出类似（'BAUnknown', 'M.BUSINESS_AREA', 'cc.cc_BUSINESS_AREA'）

```sql
if((cc.cc_BUSINESS_AREA is null and io.io_BUSINESS_AREA is null and M.BUSINESS_AREA is null),'BAUnknown',
  		if((cc.cc_BUSINESS_AREA is null and io.io_BUSINESS_AREA is null) ,M.BUSINESS_AREA,
    			if(cc.cc_BUSINESS_AREA is null,io.io_BUSINESS_AREA,cc.cc_BUSINESS_AREA)
  	)
)
```

使用withColumn将"bsn_ar_reorg_cd"和表达式得出的值合并，就是最终的结果。这里可能有个问题，如果表达式的值和"bsn_ar_reorg_cd"同名的话，新加的列会替换同名的列。

### 整个语句分析：

如果整个语句用于选择的，if判断都为flase的话，会将（'bsn_ar_reorg_cd'，'BAUnknown', 'M.BUSINESS_AREA', 'cc.cc_BUSINESS_AREA'）这四个列选择出来。

```scala
.withColumn("bsn_ar_reorg_cd", expr("if((cc.cc_BUSINESS_AREA is null and io.io_BUSINESS_AREA is null and M.BUSINESS_AREA is null),'BAUnknown',if((cc.cc_BUSINESS_AREA is null and io.io_BUSINESS_AREA is null) ,M.BUSINESS_AREA,if(cc.cc_BUSINESS_AREA is null,io.io_BUSINESS_AREA,cc.cc_BUSINESS_AREA)))"))
```

### 





