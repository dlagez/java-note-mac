点击ok提交数据， 并在后台修改

```js
const handleModalOk= () => {
  // 点击按钮之后呢显示一个loading的效果
  modalLoading.value = true;
  // 使用异步的方式保存修改的数据
  axios.post("/ebook/save", ebook.value).then((resp) => {
    const data = resp.data;
    // 这里的data就是commonResp
    if (data.success) {
      // 将对话框关闭
      modalVisible.value = false
      // 拿到值之后将loading效果去掉
      modalLoading.value = false
      // 重新加载列表
      handleQuery({
        // 查询当前页
        page: pagination.value.current,
        size: pagination.value.pageSize
      });
    }

  });
}
```

接收数据后调用service

```java
// json方式的提交 使用  @RequestBody 注解才能接收到
@PostMapping("/ebook/save")
public CommonResp save(@RequestBody EbookSaveReq req) {
    CommonResp resp = new CommonResp<>();
    ebookService.save(req);
    return resp;
}
```

service方法既保存有更新

```java
// 保存ebook
// 用于新增和更新
public void save(EbookSaveReq req) {
    Ebook ebook = CopyUtil.copy(req, Ebook.class);
    if (ObjectUtils.isEmpty(req.getId())) {
        // 新增
        ebookMapper.insert(ebook);
    } else {
        ebookMapper.updateByPrimaryKey(ebook);
    }
}
```

这个保存是有一定的问题的， 当把int类型的参数输入字符类型时会报错。