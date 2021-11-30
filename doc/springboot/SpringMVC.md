### @RequestParam

```java
// 请求	localhost:8080/users/get?id=1
@GetMapping("/get")
public UserVO get(@RequestParam("id") Integer id) {
    return new UserVO(id, UUID.randomUUID().toString());
}
```

#### @PathVariable

```java
浏览器访问：http://localhost:8080/user/3
@GetMapping("/{id}")
public UserVO get(@PathVariable("id") Integer id) {
    return new UserVO(4, "username" + id);
}
```

### @RequestBody 前端提交的数据使用json格式，必须使用这个注解

```
@PostMapping("/ebook/save")
public CommonResp save(@RequestBody EbookSaveReq req) {
    CommonResp resp = new CommonResp<>();
    ebookService.save(req);
    return resp;
}
```
