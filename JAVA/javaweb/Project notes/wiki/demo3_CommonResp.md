设置一个CommonResp的类。

```java
package com.roc.wiki.resp;

public class CommonResp<T> {
    private boolean success = true;
    private String message;
    private T content;
}
```

返回结果时将这个类返回。

```java
@GetMapping("/ebook")
public CommonResp ebookList() {
    CommonResp<List<Ebook>> resp = new CommonResp<>();
    List<Ebook> ebookList = ebookService.list();
    resp.setContent(ebookList);
    return resp;
}
```

返回的数据

```json
GET http://localhost:8080/ebook

HTTP/1.1 200 
Content-Type: application/json
Transfer-Encoding: chunked
Date: Sun, 21 Nov 2021 11:00:46 GMT
Keep-Alive: timeout=60
Connection: keep-alive

{
  "success": true,
  "message": null,
  "content": [
    {
      "id": 1,
      "name": "SpringBoot 入门教程",
      "category1Id": null,
      "category2Id": null,
      "description": "零基础入门 Java 开发，企业级应用开发最佳首选框架",
      "cover": null,
      "docCount": null,
      "viewCount": null,
      "voteCount": null
    },
    {
      "id": 2,
      "name": "Vue 入门教程",
      "category1Id": null,
      "category2Id": null,
      "description": "零基础入门 Vue 开发，企业级应用开发最佳首选框架",
      "cover": null,
      "docCount": null,
      "viewCount": null,
      "voteCount": null
    },
    {
      "id": 3,
      "name": "Python 入门教程",
      "category1Id": null,
      "category2Id": null,
      "description": "零基础入门 Python 开发，企业级应用开发最佳首选框架",
      "cover": null,
      "docCount": null,
      "viewCount": null,
      "voteCount": null
    },
    {
      "id": 4,
      "name": "MySQL 入门教程",
      "category1Id": null,
      "category2Id": null,
      "description": "零基础入门 MySQL 开发，企业级应用开发最佳首选框架",
      "cover": null,
      "docCount": null,
      "viewCount": null,
      "voteCount": null
    },
    {
      "id": 5,
      "name": "Oracle 入门教程",
      "category1Id": null,
      "category2Id": null,
      "description": "零基础入门 Oracle 开发，企业级应用开发最佳首选框架",
      "cover": null,
      "docCount": null,
      "viewCount": null,
      "voteCount": null
    }
  ]
}

Response code: 200; Time: 79ms; Content length: 922 bytes

```

