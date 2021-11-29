## git stash

`git stash`会把所有未提交的修改（包括暂存的和非暂存的）都保存起来(暂存)，用于后续恢复当前工作目录。

## 暂存:

用在以下的情形:

- 发现有一个类是多余的，想删掉它又担心以后需要查看它的代码，想保存它但又不想增加一个脏的提交。这时就可以考虑`git stash`
- 在你的分支改代码改到一半, 别人叫你去另外一个分支修改一个紧急的bug, 你不想提交一般的代码, 然后再切换分支去修改其他代码的时候就可以使用git stash

```cmd
PS C:\code\demo_wiki> git status
On branch laptop
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   web/src/views/admin/admin-ebook1.vue

PS C:\code\demo_wiki> git stash
Saved working directory and index state WIP on laptop: 5d9437f add page and data
PS C:\code\demo_wiki> git status
On branch laptop
nothing to commit, working tree clean
PS C:\code\demo_wiki> git stash apply
On branch laptop
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   web/src/views/admin/admin-ebook1.vue

PS C:\code\demo_wiki> git status
On branch laptop
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   web/src/views/admin/admin-ebook1.vue
```

需要说明一点，stash是本地的，不会通过`git push`命令上传到git server上。

可以直接使用`git stash`命令存储, 但是建议给每个暂存设置一个`message`, `git stash save 'temp' `这样方便管理.

```bash
PS C:\code\demo_wiki> git stash save 'temp'
Saved working directory and index state On laptop: temp
PS C:\code\demo_wiki> git stash list
stash@{0}: On laptop: temp
stash@{1}: WIP on laptop: 5d9437f add page and data
```

这里注意: 他是以栈的形式存储, 每次stash是一个压栈的行为, 新进入站的stash会出现再栈的顶端.

## 重新应用缓存

有两种:

- git stash pop  它会将栈顶的stash弹出, 删除缓存并将缓存重新读取到工作目录中.
- git stash apply 它会将缓存重新读取到工作目录中, 但不删除缓存 

```bash
PS C:\code\demo_wiki> git stash apply
On branch laptop
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   web/src/views/admin/admin-ebook1.vue

PS C:\code\demo_wiki> git status
On branch laptop
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   web/src/views/admin/admin-ebook1.vue
```



## 从stash创建新的分支

1. 将修改的文件暂存.
2. 此时代码变成原始代码, 继续修改.
3. 将暂存的代码拿出来. 拿出来的过程中可能会发生冲突.

git stash branch testchanges

```
PS C:\code\demo_wiki> git stash branch testStash
Switched to a new branch 'testStash'
A       web/src/views/admin/admin-ebook1.vue
error: web/src/views/admin/admin-ebook1.vue: already exists in index
error: conflicts in index. Try without --index.
```

可以看到他是有两个错误的, 这两个错误会以修改的形式出现再 modified 中

```
PS C:\code\demo_wiki> git status
On branch testStash
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   web/src/views/admin/admin-ebook1.vue

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   web/src/views/admin/admin-ebook1.vue
```



## 查看区别

如果你想查看以下stash与本地代码的区别, 使用以下命令.

git stash show    简单的实现区别

git stash show -p   详细显示区别

```bash
PS C:\code\demo_wiki> git stash show
 web/src/views/admin/admin-ebook1.vue | 135 +++++++++++++++++++++++++++++++++++
 1 file changed, 135 insertions(+)
PS C:\code\demo_wiki> git stash show -p
diff --git a/web/src/views/admin/admin-ebook1.vue b/web/src/views/admin/admin-ebook1.vue
new file mode 100644
index 0000000..f6fe79c
--- /dev/null
+++ b/web/src/views/admin/admin-ebook1.vue
@@ -0,0 +1,135 @@
+<template>
+  <a-layout>
+    <a-layout-content
+        :style="{ background: '#fff', padding: '24px', margin: 0, minHeight: '280px' }"
+    >
+<!--    :row-key="record => record.id" 每一行都要给一个key
+        :pagination="pagination" 定义了一个pagination变量
+        :loading="loading" 用到了loading变量
+        @change="handleTableChange" 点击分页会执行方法
+
+  -->
+      <a-table :columns="columns"
+               :row-key="record => record.id"
+               :data-source="ebooks"
+               :pagination="pagination"
+               :loading="loading"
+               @change="handleTableChange"
+      >
+        <!--   渲染封面, 对应setup里面的      -->
+        <template #cover="{text: cover}">
+          <img v-if="cover" :src="cover" alt="avator" style="width: 50px; height: 50px">
+        </template>
+        <!--   a-space 空格的组件     -->
+        <template v-slot:action="{ text, record }">
+          <a-space size="small">
+            <a-button type="primary">
+              编辑
+            </a-button>
+            <a-button type="primary" danger>
+              删除
```



## 删除缓存

- git stash drop stash@{0} 删除指定的stash
- git stash clear 删除所有缓存

```
C:\code\demo_wiki>git stash drop stash@{1}
Dropped stash@{1} (aaa6d77755860d067e47090e4af058a5ee97b64e)
```

这里要注意的是 powerShell 中 ,花括号在 PowerShell 中被认为是代码块执行标识符, 执行上面的代码中包含{}, 所以会报错,

解决方法是: 在 cmd 中执行即可

```
error: unknown switch `e'
usage: git stash drop [-q|--quiet] [<stash>]

    -q, --quiet           be quiet, only report errors
```



## 缓存一些忽略的文件

比如我们的日志文件一般会被忽略, 但是日志文件我也想缓存再stash里面.

`git stash`命令提供了参数用于缓存上面两种类型的文件。使用`-u`或者`--include-untracked`可以stash untracked文件。使用`-a`或者`--all`命令可以stash当前目录下的所有修改。

```bash
EAD+pzhang36@CN-PF2QT03L MINGW64 /c/code/demo_wiki (laptop)
$ git stash save 'temp' -u temp.txt
Saved working directory and index state On laptop: temp temp.txt

EAD+pzhang36@CN-PF2QT03L MINGW64 /c/code/demo_wiki (laptop)
$ git checkout master
Switched to branch 'master'
Your branch is up to date with 'origin/master'.

```

此时切换到master分支时, 还是会有temp.txt文件. 将temp.txt 删除, 再切换回laptop分支.

我这里实验的时候, apply时并没有把删除的temp.txt 重新从stash拿出来.



### 注:

撤销commit的文件

git reset HEAD^1 path/to/file1 代码会在cmd命令行报错.

```bash
假设-a误将 file1 file2 加入了commit，并且成功commit。

//将文件还原到上一个commit，也就是未修改的状态

git reset HEAD^1 path/to/file1
git reset HEAD^1 path/to/file2

//以amend方式对最新的commit进行修改，这个动作会将file1 file2从commit里“踢”出去

//但是file1 file2已经有的修改，会留在cached里。
git commit --amend
```





