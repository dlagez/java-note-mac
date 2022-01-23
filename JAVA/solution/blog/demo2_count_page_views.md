ref:`https://github.com/liuyueyi/spring-boot-demo/tree/master/spring-case/124-redis-sitecount

github：readme.md

https://github.com/liuyueyi/spring-boot-demo/blob/master/spring-case/124-redis-sitecount/readme.md

返回的对象

pv：表示这个url的总访问次数，对于每个ip，一天访问页面次数只只能为一次



uv：页面总的ip访问数



rank：当前ip第一次访问本url的排名，我是这样理解的：这个页面有很多人来访问，如果这个ip是第一个访问这个页面的人，那么这个人的排名为第一名。如果这个ip是第二个访问这个页面的人， 那么这个人的排名为第二名。



hot：热度，只要是访问了这个页面（接口）。那么热度就会加一，理解：比如一个ip今天早上8：00访问了这个页面，那么热度加一，pv加一，如果这个ip今天早上8：01再次访问了这个接口，那么热度还是会加一，但是pv不会加一，因为pv同一ip同一天内只能使得pv加一。



访问测试：

```
http://localhost:8080/visit?app=demo&ip=192.168.0.1&uri=http://hhui.top/home
```



![image-20220121154156801](C:\Users\pzhang36\AppData\Roaming\Typora\typora-user-images\image-20220121154156801.png)



实现方案：

- 解决后端存储问题，使用redis实现。有两个接口，一个接口是增加访问量。一个接口是获取访问量
- 解耦问题，使用ajax异步请求
- 传参问题
- 元素获取问题
