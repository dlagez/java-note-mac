ref: [link](https://docs.spring.io/spring-security/reference/servlet/architecture.html)

## 结构

### DelegatingFilterProxy

它就是一个中介，允许用户自己注册`filter`。然后把用户注册的`filter`代理到`FilterChain`中，下面的`SecurityFilterChain`就是通过它加入到`FilterChain`中。怎么把它加进来呢。使用的是下面的`proxy`方法`FilterChainProxy`。

### FilterChainProxy

当请求到达的时候，`FilterChainProxy`的`dofilter()` 方法内部，会遍历所有的`SecurityFilterChain`，对匹配到的`url`，则一一调用`SecurityFilterChain`中的`filter`做认证和授权。



### SecurityFilterChain

spring security通过一系列filter来对请求进行拦截，他就是安全处理器。通过一系列的filter来匹配规则并作出相应的处理

### Security Filters

使用SecurityFilterChain API将安全过滤器插入到FilterChainProxy中。

上面的几个结构可以用下面的图表示。

![安全过滤器链](https://cdn.jsdelivr.net/gh/dlagez/img@master/securityfilterchain.png)

###  Handling Security Exceptions

`ExceptionTranslationFilter`作为安全过滤器之一插入到FilterChainProxy中

这个大哥就是用来处理验证用户的。

- 首先，`ExceptionTranslationFilter`调用`FilterChain.doFilter(request, response)`来调用应用程序的其余部分。
- 如果用户未通过身份验证或它是一个`AuthenticationException`，则*开始身份验证*。
  - [SecurityContextHolder](https://docs.spring.io/spring-security/reference/servlet/authentication/architecture.html#servlet-authentication-securitycontextholder)被清除
  - `HttpServletRequest`保存在[`RequestCache`](https://docs.spring.io/spring-security/site/docs/current/api/org/springframework/security/web/savedrequest/RequestCache.html). 当用户成功认证后，`RequestCache`用于重放原始请求。
  - `AuthenticationEntryPoint`用于从客户端请求凭据。例如，它可能会重定向到登录页面或发送`WWW-Authenticate`标头。
- 否则，如果是`AccessDeniedException`，则*拒绝访问*。被`AccessDeniedHandler`调用来处理拒绝访问

处理流程图如下：

![异常翻译过滤器](https://cdn.jsdelivr.net/gh/dlagez/img@master/exceptiontranslationfilter.png)



## 身份认证

- [SecurityContextHolder](https://docs.spring.io/spring-security/reference/servlet/authentication/architecture.html#servlet-authentication-securitycontextholder) - `SecurityContextHolder`Spring Security 存储[身份验证](https://docs.spring.io/spring-security/reference/features/authentication/index.html#authentication)者详细信息的位置。
- [SecurityContext](https://docs.spring.io/spring-security/reference/servlet/authentication/architecture.html#servlet-authentication-securitycontext) - 从 获取`SecurityContextHolder`并包含`Authentication`当前经过身份验证的用户的。
- [Authentication](https://docs.spring.io/spring-security/reference/servlet/authentication/architecture.html#servlet-authentication-authentication)- 可以是输入以`AuthenticationManager`提供用户提供的用于身份验证的凭据或来自`SecurityContext`.
- [GrantedAuthority](https://docs.spring.io/spring-security/reference/servlet/authentication/architecture.html#servlet-authentication-granted-authority) - 授予主体的权限`Authentication`（即角色、范围等）
- [AuthenticationManager](https://docs.spring.io/spring-security/reference/servlet/authentication/architecture.html#servlet-authentication-authenticationmanager) - 定义 Spring Security 的过滤器如何执行身份验证的[API](https://docs.spring.io/spring-security/reference/features/authentication/index.html#authentication)。
- [ProviderManager](https://docs.spring.io/spring-security/reference/servlet/authentication/architecture.html#servlet-authentication-providermanager) - 最常见的实现`AuthenticationManager`。
- [AuthenticationProvider](https://docs.spring.io/spring-security/reference/servlet/authentication/architecture.html#servlet-authentication-authenticationprovider) - 用于`ProviderManager`执行特定类型的身份验证。
- [Request Credentials with`AuthenticationEntryPoint`](https://docs.spring.io/spring-security/reference/servlet/authentication/architecture.html#servlet-authentication-authenticationentrypoint) - 用于从客户端请求凭据（即重定向到登录页面、发送`WWW-Authenticate`响应等）
- [AbstractAuthenticationProcessingFilter](https://docs.spring.io/spring-security/reference/servlet/authentication/architecture.html#servlet-authentication-abstractprocessingfilter) -`Filter`用于身份验证的基础。这也很好地了解了身份验证的高级流程以及各个部分如何协同工作。