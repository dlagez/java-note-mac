在sevice层使用pageHelper，我们从前端动态的获取分页的参数

```java
@Service
public class EbookService {

    @Resource
    private EbookMapper ebookMapper;

    private static final Logger log = LoggerFactory.getLogger(EbookService.class);

    public List<Ebook> list() {
        return ebookMapper.selectByExample(null);
    }

    // 返回类。
    public List<EbookResp> listLike(EbookReq req) {

        EbookExample ebookExample = new EbookExample();
        EbookExample.Criteria criteria = ebookExample.createCriteria();
        if (!ObjectUtils.isEmpty(req.getName())) {
            criteria.andNameLike("%" + req.getName() + "%");
        }
        // pageNum 是从1开始 只对第一个遇到的sql起作用，比如只对下面第一个ebookMapper.selectByExample起作用
        PageHelper.startPage(req.getPage(), req.getSize());
        List<Ebook> ebooks = ebookMapper.selectByExample(ebookExample);

        PageInfo<Ebook> info = new PageInfo<>(ebooks);
        log.info("总行数：{}", info.getTotal());
        log.info("总页数：{}", info.getPages());

        List<EbookResp> list = CopyUtil.copyList(ebooks, EbookResp.class);
        return list;
    }
}
```

使用：

```
GET http://localhost:8088/ebook/list?page=1&size=4
```