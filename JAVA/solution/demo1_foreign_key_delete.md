情景描述：

博客系统（照着某课程编写的），遇到这样一个问题：

下图是博客网站后台管理，可以对类型进行添加和删除操作。但是这个类型和博客是有一个外键连接的。如果在博客中已经使用了该`type`。比如下面的`java`类型就已经被使用了。有一篇`blog`的类型就是`java`。在`sql`中有外键约束，如果现在删除类型java的话会直接报错。

![image-20220119230420746](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220119230420746.png)

他们之间的关系如下图所示。

![image-20220119231748115](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220119231748115.png)

在`service`中，提供的删除操作代码如下。

原始代码：

```java
// 删除功能
    @GetMapping("/types/{id}/delete")
    public String delete(@PathVariable Long id, RedirectAttributes attributes) {
        typeService.deleteType(id);
        attributes.addFlashAttribute("message", "删除成功");
        return "redirect:/admin/types";
    }
```

这个问题该怎么解决呢：

解决方案一：

​	我想找一下有没有这样的`sql`语句，如果某一列已经被外键引用了就返回`true`，如果没有被引用就返回`false`。

但是找了很久都没有找到这样的语句。

解决方案二：

在`blog`中查询出所有的`type`。然后在原始代码中加入判断，

- 如果传进入的参数（要删除的`type`的`id`）在查询出来的结果里面，就表示该`type`已经被使用了。表示该`type`删除不了。返回相应的`message`。
- 如果传进入的参数（要删除的type的id）不在查询出来的结果里面。表示可以删除，直接调用`typeService.deleteType(id);`方法将`type`删除即可。

```
select distinct type_id from t_blog;
```

问题：如果使用这个解决方案，就会出现一个问题，增加了一次`sql`查询操作。这就会导致删除操作会消耗更多的资源。如果数据库中的博客数量很多。上述的语句不仅查询速度慢，还会消耗大量数据库资源。

解决方案三：

直接在`java`代码中做出判断。如果不能删除，说明`type`的外键已经被使用了。

```java
// 删除功能
    @GetMapping("/types/{id}/delete")
    public String delete(@PathVariable Long id, RedirectAttributes attributes) {
        try {
            typeService.deleteType(id);
            // 删除后返回删除信息，并重定向到分类页面
            attributes.addFlashAttribute("message", "删除成功");
            return "redirect:/admin/types";
        } catch (Exception e) {
            System.out.println("types外键已经被引用，不能删除");
        }
        attributes.addFlashAttribute("message", "删除失败, types已经被使用！");
        return "redirect:/admin/types";
    }
```

