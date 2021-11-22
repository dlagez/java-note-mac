前端将参数传到后台。

这里接收参数可以不加修饰的`@`，参数会自动绑定到同名的属性上。这里也可以使用对象。前端传来的参数可以自定绑定到对象的属性里面。要保证名字一样。

```java
@GetMapping("/ebookLike")
public CommonResp ebookLikeList(String name) {
    CommonResp<List<Ebook>> resp = new CommonResp<>();
    List<Ebook> ebookList = ebookService.listLike(name);
    resp.setContent(ebookList);
    return resp;
}
```

创建`EbookExample`类来使用条件

```java
public List<Ebook> listLike(String name) {
    // 下面两行时条件查询的固定表达式。
    EbookExample ebookExample = new EbookExample();
    EbookExample.Criteria criteria = ebookExample.createCriteria();
    criteria.andNameLike("%" + name + "%");
    return ebookMapper.selectByExample(ebookExample);
}
```