添加删除的controller

```java
@DeleteMapping("/ebook/delete/{id}")
public CommonResp delete(@PathVariable("id") Long id) {
    CommonResp resp = new CommonResp<>();
    int delete = ebookService.delete(id);
    if (delete == 0) {
        resp.setSuccess(false);
    }
    return resp;
}
```

删除的service

```java
public int delete(Long id) {
    return ebookMapper.deleteByPrimaryKey(id);
}
```

修改delete按钮的样式和方法

```java
<a-popconfirm
    title="Are you sure delete this ebook?"
    ok-text="Yes"
    cancel-text="No"
    @confirm="del(record.id)"
>
  <a-button type="primary" danger>
    delete
  </a-button>
</a-popconfirm>
```

```js
    const del = (id) => {
      axios.delete("/ebook/delete/" + id).then((resp) => {
        const data = resp.data;
        // 这里的data就是commonResp
        if (data.success) {
          // 重新加载列表
          handleQuery({
            // 查询当前页
            page: pagination.value.current,
            size: pagination.value.pageSize
          });
        }
      });
    };
```

