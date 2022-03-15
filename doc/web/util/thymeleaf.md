语法：

`th:text`：

### 模板：

第一步：定义模板文件

```html
<nav th:fragment="menu(n)" class="ui inverted attached segment m-padded-tb-mini m-shadow-small" >
  <div class="ui container">
    <div class="ui inverted secondary stackable menu">
      <h2 class="ui teal header item">Blog</h2>
      <a href="#" th:href="@{/}" class="m-item item m-mobile-hide " th:classappend="${n==1} ? 'active'"><i class="mini home icon"></i>首页</a>
      <a href="#" th:href="@{/types/-1}" class="m-item item m-mobile-hide" th:classappend="${n==2} ? 'active'"><i class="mini idea icon"></i>分类</a>
      <a href="#" th:href="@{/tags/-1}" class="m-item item m-mobile-hide" th:classappend="${n==3} ? 'active'"><i class="mini tags icon"></i>标签</a>
      <a href="#" th:href="@{/archives}" class="m-item item m-mobile-hide" th:classappend="${n==4} ? 'active'"><i class="mini clone icon"></i>归档</a>
      <a href="#" th:href="@{/about}" class="m-item item m-mobile-hide" th:classappend="${n==5} ? 'active'"><i class="mini info icon"></i>关于我</a>
      <div class="right m-item item m-mobile-hide">

        <!-- 这里的name 是为了绑定document事件 -->
        <form action="#" name="search" th:action="@{/search}" target="_blank" method="post">
          <div class="ui icon inverted transparent input m-margin-tb-tiny">
            <!-- 这里的name是为了获取输入的值，通过这个名字获取 -->
            <input type="text" name="query" th:value="${query}" placeholder="Search....">
            <i onclick="document.forms['search'].submit()" class="search link icon"></i>
          </div>
        </form>

      </div>
    </div>
  </div>
  <a href="#" class="ui menu toggle black icon button m-right-top m-mobile-show">
    <i class="sidebar icon"></i>
  </a>
</nav>
```

第二步：可以使用`th:include`或者`th:replace`属性来使用它：

```html

  <nav th:replace="_fragments :: menu(5)" class="ui inverted attached segment m-padded-tb-mini m-shadow-small" >
    <!--  这里的内容是引用fragments的模板  -->
  </nav>
```

