为了校验异常的时候能显示将异常打包成`CommonResp` 添加同一异常处理类

```java
package com.roc.wiki.controller;

import com.roc.wiki.resp.CommonResp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.validation.BindException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;

@ControllerAdvice
public class ControllerExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(ControllerExceptionHandler.class);

    /**
     * 校验异常同一处理
     * @param e
     * @return
     */
    @ExceptionHandler(value = BindException.class)
    @ResponseBody
    public CommonResp validExceptionHandler(BindException e) {
        CommonResp resp = new CommonResp();
        log.warn("参数校验失败： {}", e.getBindingResult().getAllErrors().get(0).getDefaultMessage());
        resp.setSuccess(false);
        resp.setMessage(e.getBindingResult().getAllErrors().get(0).getDefaultMessage());
        return resp;
    }

    @ExceptionHandler(value = Exception.class)
    @ResponseBody
    public CommonResp validExceptionHandler(Exception e) {
        CommonResp commonResp = new CommonResp();
        log.error("系统异常：", e);
        commonResp.setSuccess(false);
        commonResp.setMessage("系统出现异常，请联系管理员");
        return commonResp;
    }

}
```

EbookSaveReq.java

在实体类的属性上添加参数校验规则

```java
@NotNull(message = "名称不能为空")
private String name;
```

EbookController.java

在接受参数时添加`@Valid`参数，来添加参数校验方法

```java
public CommonResp ebookList(@Valid EbookQueryReq req) {
  
public CommonResp save(@Valid @RequestBody EbookSaveReq req) {
```



在页面上添加组件用来显示校验错误信息

admin-ebook.vue

```js
<script>
import { message } from 'ant-design-vue';
    const handleQuery = (params) => {
      loading.value = true;
      axios.get("/ebook/list", {
        params: {
          page: params.page,
          size: params.size
        }
      }).then((resp) => {
        loading.value = false;
        const data = resp.data;
        // data.content 会得到PageResp对象，resp对象的list才是数据
        if (data.success) {
          ebooks.value = data.content.list;
          // 重置分页按钮
          pagination.value.current = params.page;
          // 这里是后端分页查询时查询数据库的总数据量
          pagination.value.total = data.content.total;
        } else {
          message.error(data.message)  //  这里使用message组件
        }
      });
    };
</script>
```

