开发时遇到了这样一个问题：

后端数据库的id使用了bigint类型存储，使用雪花算法来生成id。

前端拿到这个id之后精度丢失了。

数据库的数据：

```
id:121234688009441285
```

前端拿到的数据：

```
id:121234688009441280
```

可以很明显的看到数据的最后一位丢失了。

**解决方法：**

在springboot添加一下的配置类即可。

```java
package com.roc.wiki.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.converter.json.Jackson2ObjectMapperBuilder;

@Configuration
public class JacksonConfig {

    @Bean
    public ObjectMapper jacksonObjectMapper(Jackson2ObjectMapperBuilder builder) {
        ObjectMapper objectMapper = builder.createXmlMapper(false).build();
        SimpleModule simpleModule = new SimpleModule();
        simpleModule.addSerializer(Long.class, ToStringSerializer.instance);
        objectMapper.registerModule(simpleModule);
        return objectMapper;
    }
}
```