要了解什么是GET和POST请求，首先得了解HTTP。

超文本传输协议（HTTP）的设计目的是保证客户端与服务器之间的通信。HTTP的工作方式是客户端与服务器之间的请求-应答协议。

http协议客户端请求request消息包括以下格式：请求行、请求头部、空行、请求数据。

![在这里插入图片描述](GET%E5%92%8CPOST%E8%AF%B7%E6%B1%82.assets/1460000023940347.png)

服务器响应response也是有四个部分组成：相应行、相应头、空行、相应体。

<img src="https://segmentfault.com/img/remote/1460000023940348" alt="在这里插入图片描述" style="zoom:67%;" />

看一个实际的例子，不仅有请求消息，还有返回消息。

<img src="GET%E5%92%8CPOST%E8%AF%B7%E6%B1%82.assets/image-20210914125129019.png" alt="image-20210914125129019" style="zoom:50%;" />

```
POST /admin/login HTTP/1.1
Conte: application/json
Content-Type: application/json
User-Agent: PostmanRuntime/7.28.4
Accept: */*
Postman-Token: 5ba059d6-9b03-4afb-9bdb-e2fd5a189a0b
Host: localhost:8080
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
Content-Length: 49
 
{
"password": "12345",
"username": "test"
}
 
HTTP/1.1 200 OK
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Cache-Control: no-cache, no-store, max-age=0, must-revalidate
Pragma: no-cache
Expires: 0
X-Frame-Options: DENY
Content-Type: application/json
Transfer-Encoding: chunked
Date: Mon, 13 Sep 2021 07:35:28 GMT
Keep-Alive: timeout=60
Connection: keep-alive
 
{"code":404,"message":"用户名或密码错误","data":null}
```

**请求参数**：GET请求参数是通过URL传递的，多个参数以&连接，POST请求放在request body中。
**请求缓存**：GET请求会被缓存，而POST请求不会，除非手动设置。
**收藏为书签**：GET请求支持，POST请求不支持。
**安全性**：POST比GET安全，GET请求在浏览器回退时是无害的，而POST会再次请求。
**历史记录**：GET请求参数会被完整保留在浏览历史记录里，而POST中的参数不会被保留。
**编码方式**：GET请求只能进行url编码，而POST支持多种编码方式。
**对参数的数据类型**：GET只接受ASCII字符，而POST没有限制。

ref：

- https://www.runoob.com/tags/html-httpmethods.html
- https://segmentfault.com/a/1190000023940344