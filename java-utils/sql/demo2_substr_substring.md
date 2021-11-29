oracle：

语法：

substr(string,a,b)：

a：从第几位开始，第一位a=1，倒数第三位(即sqlserver中的right)a=-3

b：取几个字符

```
1、select substr('HelloWorld',0,3) value from dual; //返回结果：Hel，截取从“H”开始3个字符
 2、select substr('HelloWorld',1,3) value from dual; //返回结果：Hel，截取从“H”开始3个字符
 3、select substr('HelloWorld',2,3) value from dual; //返回结果：ell，截取从“e”开始3个字符
 4、select substr('HelloWorld',0,100) value from dual; //返回结果：HelloWorld，100虽然超出预处理的字符串最长度，但不会影响返回结果，系统按预处理字符串最大数量返回。
 5、select substr('HelloWorld',5,3) value from dual; //返回结果：oWo
```

sqlserver	

SELECT SUBSTRING('12345', 2, 5) AS ExtractString





