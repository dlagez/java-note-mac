### mysql

```
spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql:///blog?allowPublicKeyRetrieval=true&characterEncoding=utf8&useSSL=false
    username: root
    password: password
```



### jpa

```
spring:
    jpa:
      hibernate:
        ddl-auto: update
      show-sql: true
update 用于开发，帮助建表，none用于生产。
ddl-auto:create----每次运行该程序，没有表格会新建表格，表内有数据会清空
ddl-auto:create-drop----每次程序结束的时候会清空表
ddl-auto:update----每次运行程序，没有表格会新建表格，表内有数据不会清空，只会更新
ddl-auto:validate----运行程序会校验数据与数据库的字段类型是否相同，不同会报错
ddl-auto:none----禁止ddl
```



### redis

```
#以下是redis的配置
redis:
  host: 127.0.0.1
  port: 6379
  database: 0
  timeout: 5000
  jedis:
    pool:
      max-active: 8
      max-idle: 20
      min-idle: 8
      max-wait: 5000
```