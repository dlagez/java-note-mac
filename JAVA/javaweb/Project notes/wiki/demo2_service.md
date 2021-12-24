对应开发service层，这个项目并没有将service分层，所以我这里也就步分层了。因为他的代码两并不大。

```java
package com.roc.wiki.service;

import com.roc.wiki.domain.Ebook;
import com.roc.wiki.mapper.EbookMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.util.List;

@Service
public class EbookService {

    @Resource // 这里使用Resource注解不报红，使用@Aotowried报红，但是不影响使用。
    private EbookMapper ebookMapper;

    public List<Ebook> list() {
        return ebookMapper.selectByExample(null);
    }

}
```

其实这个service并不难。