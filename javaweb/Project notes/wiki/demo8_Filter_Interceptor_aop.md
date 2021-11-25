Filter & Interceptor & aop

这三个选一个用即可.

## Filter 

使用一个链来调用方法把请求传回过滤链. 

```java
package com.roc.wiki.filter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.servlet.*;
import javax.servlet.http.HttpServletRequest;
import java.io.IOException;

@Component
public class LogFilter implements Filter {
    private static final Logger LOG = LoggerFactory.getLogger(LogFilter.class);


    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
    // 这里只会执行一次
    }

    @Override
    public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse, FilterChain filterChain) throws IOException, ServletException {
        // 这里请求一次打印一次
        HttpServletRequest request = (HttpServletRequest) servletRequest;
        LOG.info("------------- LogFilter 开始 -------------");
        LOG.info("请求地址: {} {}", request.getRequestURL().toString(), request.getMethod());
        LOG.info("远程地址: {}", request.getRemoteAddr());

        long startTime = System.currentTimeMillis();
        filterChain.doFilter(servletRequest, servletResponse);
        LOG.info("------------- LogFilter 结束 耗时：{} ms -------------", System.currentTimeMillis() - startTime);
    }
}
```



###  拦截器：

Spring框架特有的，，请求日志打印 /login

过滤器的preHandler返回false, 整个业务也就结束了. 常用于登录校验，权限校验, 当你没有登陆时后面的业务逻辑也就不会执行了. 

```java
package com.roc.wiki.interceptor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;
import org.springframework.web.servlet.ModelAndView;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@Component
public class LogInterceptor implements HandlerInterceptor {
    private static final Logger LOG = LoggerFactory.getLogger(LogInterceptor.class);

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        // 打印请求信息
        LOG.info("------------- LogInterceptor 开始 -------------");
        LOG.info("请求地址: {} {}", request.getRequestURL().toString(), request.getMethod());
        LOG.info("远程地址: {}", request.getRemoteAddr());

        long startTime = System.currentTimeMillis();
        request.setAttribute("requestStartTime", startTime);
        return true;
    }

    @Override
    public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, ModelAndView modelAndView) throws Exception {
        long startTime = (Long) request.getAttribute("requestStartTime");
        LOG.info("------------- LogInterceptor 结束 耗时：{} ms -------------", System.currentTimeMillis() - startTime);
    }
}

```

与过滤器不同的是, 拦截器需要配置拦截路径等相关的配置.

```java
package com.roc.wiki.config;

import com.roc.wiki.interceptor.LogInterceptor;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import javax.annotation.Resource;

@Configuration
public class SpringMvcConfig implements WebMvcConfigurer {

    @Resource
    LogInterceptor logInterceptor;

    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(logInterceptor)
                .addPathPatterns("/**")
                .excludePathPatterns(
                        "/test/**",
                        "/redis/**",
                        "/login"
                );
    }
}
```



配置完上面两之后我们来看看他两执行的顺序

1. 先进入`Filter`方法, 将请求传回过滤器之前的代码都执行完了.
2. 进入`Interceptor`方法. 执行`preHandle`方法里面的代码.
3. 如果`preHandle`方法返回true
4. 执行`posthandle`方法
5. 执行Filter方法将请求传回过滤器之后的代码. 

总结: Filter 的作用范围更大, 因为他是在tomcat容器里面, 所以接口会先进到容器, 然后容器会将请求发送到一个应用, 我们的springboot就是一个web应用, 进入到web应用之后我们的Interceptor就拿到请求了.

```
2021-11-23 21:00:40.916  INFO 26232 --- [nio-8088-exec-1] com.roc.wiki.filter.LogFilter            : ------------- LogFilter 开始 -------------
2021-11-23 21:00:40.916  INFO 26232 --- [nio-8088-exec-1] com.roc.wiki.filter.LogFilter            : 请求地址: http://localhost:8088/ebookList GET
2021-11-23 21:00:40.917  INFO 26232 --- [nio-8088-exec-1] com.roc.wiki.filter.LogFilter            : 远程地址: 0:0:0:0:0:0:0:1
2021-11-23 21:00:40.918  INFO 26232 --- [nio-8088-exec-1] com.roc.wiki.interceptor.LogInterceptor  : ------------- LogInterceptor 开始 -------------
2021-11-23 21:00:40.918  INFO 26232 --- [nio-8088-exec-1] com.roc.wiki.interceptor.LogInterceptor  : 请求地址: http://localhost:8088/ebookList GET
2021-11-23 21:00:40.918  INFO 26232 --- [nio-8088-exec-1] com.roc.wiki.interceptor.LogInterceptor  : 远程地址: 0:0:0:0:0:0:0:1
2021-11-23 21:00:40.927  INFO 26232 --- [nio-8088-exec-1] com.zaxxer.hikari.HikariDataSource       : HikariPool-3 - Starting...
2021-11-23 21:00:40.970  INFO 26232 --- [nio-8088-exec-1] com.zaxxer.hikari.HikariDataSource       : HikariPool-3 - Start completed.
2021-11-23 21:00:40.994  INFO 26232 --- [nio-8088-exec-1] com.roc.wiki.interceptor.LogInterceptor  : ------------- LogInterceptor 结束 耗时：76 ms -------------
2021-11-23 21:00:40.995  INFO 26232 --- [nio-8088-exec-1] com.roc.wiki.filter.LogFilter            : ------------- LogFilter 结束 耗时：78 ms -------------

```



### aop

```java
package com.roc.wiki.aspect;

import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.support.spring.PropertyPreFilters;
import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.Signature;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;
import org.aspectj.lang.annotation.Pointcut;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import javax.servlet.http.HttpServletRequest;

@Aspect
@Component
public class LogAspect {
    private final static Logger log = LoggerFactory.getLogger(LogAspect.class);

    @Pointcut("execution(public * com.roc.*.controller..*Controller.*(..))")
    public void controllerPointcut() {}

    @Before("controllerPointcut()")
    public void doBefore(JoinPoint joinPoint) {

        // 打印请求日志
        ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        HttpServletRequest request = attributes.getRequest();
        Signature signature = joinPoint.getSignature();
        String name = signature.getName();

        log.info("----------aspect begin-------------");
        log.info("请求地址: {} {}", request.getRequestURL().toString(), request.getMethod());
        log.info("类名方法: {}.{}", signature.getDeclaringTypeName(), name);
        log.info("远程地址: {}", request.getRemoteAddr());

        // 下面就是拿到请求参数
        //  比如说拿到这个的EbookEeq参数   public CommonResp ebookLikeListResp(EbookReq req) {
        Object[] args = joinPoint.getArgs();
        Object[] arguments  = new Object[args.length];
        for (int i = 0; i < args.length; i++) {
            if (args[i] instanceof ServletRequest
                    || args[i] instanceof ServletResponse
                    || args[i] instanceof MultipartFile) {
                continue;
            }
            arguments[i] = args[i];
        }

        // 排除字段，敏感字段或太长的字段不显示
        String[] excludeProperties = {"password", "file"};
        PropertyPreFilters filters = new PropertyPreFilters();
        PropertyPreFilters.MySimplePropertyPreFilter excludefilter = filters.addFilter();
        excludefilter.addExcludes(excludeProperties);
        log.info("请求参数: {}", JSONObject.toJSONString(arguments, excludefilter));
    }
    // 环绕通知
    @Around("controllerPointcut()")
    public Object doAround(ProceedingJoinPoint proceedingJoinPoint) throws Throwable {
        long startTime = System.currentTimeMillis();
        // 在此之前的代码相当于放在了doBefore里面
        // 下面语句是执行方法的内容
        Object result = proceedingJoinPoint.proceed();
        // 排除字段，敏感字段或太长的字段不显示
        String[] excludeProperties = {"password", "file"};
        PropertyPreFilters filters = new PropertyPreFilters();
        PropertyPreFilters.MySimplePropertyPreFilter excludefilter = filters.addFilter();
        excludefilter.addExcludes(excludeProperties);
        log.info("返回结果: {}", JSONObject.toJSONString(result, excludefilter));
        log.info("------------- 结束 耗时：{} ms -------------", System.currentTimeMillis() - startTime);
        return result;
    }
}

```

