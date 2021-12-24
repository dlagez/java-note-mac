第一步在ebook里面请求所有分类数据。

```js
const categoryIds = ref()
const tree_data = ref()
let categorys

const handleQueryCategory = () => {
  loading.value = true;
  axios.get("/category/all").then((resp) => {
    loading.value = false;
    const data = resp.data;
    // data.content 会得到PageResp对象，resp对象的list才是数据
    if (data.success) {
      categorys = data.content;

      tree_data.value = []
      tree_data.value = Tool.array2Tree(categorys, 0)
      console.log("tree structe", tree_data)
    } else {
      message.error(data.message)
    }
  });
};
```

在点击编辑按钮时将category数据带到编辑框。

```js
const edit = (record) => {
  modalVisible.value = true;
  // 这里直接把record传递到ebook，编辑时会直接影响原值，即使没有提交。
  ebook.value = Tool.copy(record)
  categoryIds.value = [ebook.value.category1Id, ebook.value.category2Id]
}
```

在编辑框中显示目前的分类，并将所有分类渲染到候选框里面。

```html
<a-form :model="ebook" :label-col="{span : 6}">
  <a-form-item label="cover">
    <a-input v-model:value="ebook.cover"/>
  </a-form-item>
  <a-form-item label="name">
    <a-input v-model:value="ebook.name"/>
  </a-form-item>
  <a-form-item label="分类">
    <a-cascader v-model:value="categoryIds"
                :field-names="{label:'name', value:'id', children:'children'}"
                :options="tree_data"/>
  </a-form-item>
  <!--   name: 显示的值， value：实际的值 -->
  <a-form-item label="description">
    <a-input v-model:value="ebook.description" type="text"/>
  </a-form-item>
</a-form>
```