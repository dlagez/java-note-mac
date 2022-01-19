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



### Model

```java
@GetMapping("/")
    public String index(@PageableDefault(size = 8, sort = {"updateTime"}, direction = Sort.Direction.DESC)
                                Pageable pageable, Model model) {
        model.addAttribute("page", blogService.listBlog(pageable));
        model.addAttribute("types", typeService.listTypeTop(6));
        model.addAttribute("tags", tagService.listTagTop(10));
        model.addAttribute("recommendBlogs", blogService.listRecommendBlogTop(8));
        return "index";
    }
```

从广义上来说，Model指的是MVC中的M，即Model(模型)。从狭义上讲，Model就是个key-value集合。实际上，上图home方法得到的model对象就是一个 `java.util.Map` ，你可以将Model类型替换为`Map<String, Object>` ，或者ModelMap——一个实现了Model接口的`java.util.HashMap`。

### @ModelAttribute的用法

在所有的请求处理方法执行之前，你有机会往Model里面加数据：新建一个方法，加上@ModelAttribute注解和Model参数，就可以像下图这种样加数据了。

```java
// 这个方法会在请求处理之前添加值到model里面
@ModelAttribute
public void fillData(Model model) {
    model.addAttribute("test", "test_value");
}
```

