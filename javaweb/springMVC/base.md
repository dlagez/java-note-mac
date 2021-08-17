### 注解：

#### @Controller

添加在类上，表示这是控制器 Controller 对象。属性如下：

- `name` 属性：该 Controller 对象的 Bean 名字。允许空。

[`@RestController`](https://github.com/spring-projects/spring-framework/blob/master/spring-web/src/main/java/org/springframework/web/bind/annotation/RestController.java) 注解，添加在类上，是 `@Controller` 和 [`@ResponseBody`](https://github.com/ndimiduk/spring-framework/blob/master/org.springframework.web/src/main/java/org/springframework/web/bind/annotation/ResponseBody.java) 的组合注解，直接使用接口方法的返回结果，经过 JSON/XML 等序列化方式，最终返回。也就是说，无需使用 InternalResourceViewResolver 解析视图，返回 HTML 结果。

目前主流的架构，都是 [前后端分离](https://blog.csdn.net/fuzhongmin05/article/details/81591072) 的架构，后端只需要提供 API 接口，仅仅返回数据。而视图部分的工作，全部交给前端来做。也因此，我们项目中 99.99% 使用 `@RestController` 注解。

#### @RequestMapping

添加在类或方法上，标记该类/方法对应接口的配置信息。

`@RequestMapping` 注解的**常用属性**，如下：

- `path` 属性：接口路径。`[]` 数组，可以填写多个接口路径。
- `values` 属性：和 `path` 属性相同，是它的别名。
- `method` 属性：请求方法 [RequestMethod](https://github.com/spring-projects/spring-framework/blob/master/spring-web/src/main/java/org/springframework/web/bind/annotation/RequestMethod.java) ，可以填写 `GET`、`POST`、`POST`、`DELETE` 等等。`[]` 数组，可以填写多个请求方法。如果为空，表示匹配所有请求方法。

`@RequestMapping` 注解的**不常用属性**，如下：

- `name` 属性：接口名。一般情况下，我们不填写。
- `params` 属性：请求参数需要包含值的**参数名**。可以填写多个参数名。如果为空，表示匹配所有请你求方法。
- `headers` 属性：和 `params` 类似，只是从参数名变成**请求头**。
- `consumes` 属性：和 `params` 类似，只是从参数名变成请求头的**提交内容类型**( [Content-Type](https://juejin.im/post/5cb34fc06fb9a068a75d3555) )
- `produces` 属性：和 `params` 类似，只是从参数名变成请求头的( [Accept](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Headers/Accept) )**可接受类型**。



- [`@GetMapping`](https://github.com/spring-projects/spring-framework/blob/master/spring-web/src/main/java/org/springframework/web/bind/annotation/GetMapping.java) 注解：对应 `@GET` 请求方法的 `@RequestMapping` 注解。
- [`@PostMapping`](https://github.com/spring-projects/spring-framework/blob/master/spring-web/src/main/java/org/springframework/web/bind/annotation/PostMapping.java) 注解：对应 `@POST` 请求方法的 `@RequestMapping` 注解。
- [`@PutMapping`](https://github.com/spring-projects/spring-framework/blob/master/spring-web/src/main/java/org/springframework/web/bind/annotation/PutMapping.java) 注解：对应 `@PUT` 请求方法的 `@RequestMapping` 注解。
- [`@DeleteMapping`](https://github.com/spring-projects/spring-framework/blob/master/spring-web/src/main/java/org/springframework/web/bind/annotation/DeleteMapping.java) 注解：对应 `@DELETE` 请求方法的 `@RequestMapping` 注解。

#### @RequestParam

[`@RequestParam`](https://github.com/spring-projects/spring-framework/blob/master/spring-web/src/main/java/org/springframework/web/bind/annotation/RequestParam.java) 注解，添加在方法参数上，标记该方法参数对应的请求参数的信息。属性如下：

- `name` 属性：对应的请求参数名。如果为空，则直接使用方法上的参数变量名。
- `value` 属性：和 `name` 属性相同，是它的别名。
- `required` 属性：参数是否必须传。默认为 `true` ，表示必传。
- `defaultValue` 属性：参数默认值。

例子：

```java
// 请求	localhost:8080/users/get?id=1
@GetMapping("/get")
public UserVO get(@RequestParam("id") Integer id) {
    return new UserVO(id, UUID.randomUUID().toString());
}
```



#### @PathVariable

[`@PathVariable`](https://github.com/spring-projects/spring-framework/blob/master/spring-web/src/main/java/org/springframework/web/bind/annotation/PathVariable.java) 注解，添加在方法参数上，标记接口路径和方法参数的映射关系。具体的，我们在示例中来看。相比 `@RequestParam` 注解，少一个 `defaultValue` 属性。

使用：

```
浏览器访问：http://localhost:8080/users/3
@GetMapping("/{id}")
public UserVO get(@PathVariable("id") Integer id) {
    return new UserVO(4, "username" + id);
}
```

### 示例

下面，让我们快速编写一个 SpringMVC 的示例。

实例：

```java
package com.example.springmvc.controller;

import com.example.springmvc.vo.UserVO;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/users")
public class UserController {

    @GetMapping("")
   public List<UserVO> list() {
        List<UserVO> result = new ArrayList<>();
        result.add(new UserVO(1, "roczhang"));
        result.add(new UserVO(2, "dlage"));
        result.add(new UserVO(3, "cccc"));
        return result;
    }
}
```



## 全局统一返回

实际项目在实践时，我们会将状态码放在 Response Body **响应内容**中返回

在全局统一返回里，我们至少需要定义三个字段：

- `code`：状态码。无论是否成功，必须返回。

  - 成功时，状态码为 0 。
  - 失败时，对应业务的错误码。

  > 关于这一块，也有团队实践时，增加了 `success` 字段，通过 `true` 和 `false` 表示成功还是失败。这个看每个团队的习惯吧。艿艿的话，还是偏好基于约定，返回 0 时表示成功。

- `data`：数据。成功时，返回该字段。

- `message`：错误提示。失败时，返回该字段。