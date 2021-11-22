请求和返回值的封装。

一般情况下请求类不用封装。这个项目用到了，就按照它的写。

返回值封装时可以有的。一般我们查询用户，返回值的字段和数据库的字段并不是一一对应的，比如说用户的密码我们就不能返回给用户

封装请求实体类：`EbookReq.java`

```java
package com.roc.wiki.req;

public class EbookReq {
    private Long id;
    private String name;
    getter, setter...
}
```

使用方法：

```java
@GetMapping("/ebookLikeReq")
public CommonResp ebookLikeList(EbookReq req) {
    CommonResp<List<Ebook>> resp = new CommonResp<>();
    List<Ebook> ebookList = ebookService.listLike(req.getName());
    resp.setContent(ebookList);
    return resp;
}
```

封装返回实体类：`EbookResp.java` 它其实和`Ebook`实体类一摸一样。这里暂时没有改变。

```java
package com.roc.wiki.resp;

public class EbookResp {
    private Long id;

    private String name;

    private Long category1Id;

    private Long category2Id;

    private String description;

    private String cover;

    private Integer docCount;

    private Integer viewCount;

    private Integer voteCount;

}

```

使用：

```java
// 返回类。
public List<EbookResp> listLikeResp(String name) {
    EbookExample ebookExample = new EbookExample();
    EbookExample.Criteria criteria = ebookExample.createCriteria();
    criteria.andNameLike("%" + name + "%");
    List<Ebook> ebooks = ebookMapper.selectByExample(ebookExample);
    ArrayList<EbookResp> ebookResps = new ArrayList<>();
    for (Ebook ebook : ebooks) {
    EbookResp ebookResp = new EbookResp();
    BeanUtils.copyProperties(ebook, ebookResp);
    ebookResps.add(ebookResp);
    }
    return ebookResps;
}
```

```java
@GetMapping("/ebookLikeReqResp")
public CommonResp ebookLikeListResp(EbookReq req) {
    CommonResp<List<EbookResp>> resp = new CommonResp<>();
    List<EbookResp> ebookList = ebookService.listLikeResp(req.getName());
    resp.setContent(ebookList);
    return resp;
}
```

返回结果

```
GET http://localhost:8080/ebookLikeReqResp?name=vue

HTTP/1.1 200 
Content-Type: application/json
Transfer-Encoding: chunked
Date: Sun, 21 Nov 2021 11:50:04 GMT
Keep-Alive: timeout=60
Connection: keep-alive

{
  "success": true,
  "message": null,
  "content": [
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
    }
  ]
}

Response code: 200; Time: 339ms; Content length: 214 bytes

```

