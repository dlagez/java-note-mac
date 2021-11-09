ref：https://git-scm.com/book/zh/v2/Git-%E5%88%86%E6%94%AF-%E5%88%86%E6%94%AF%E7%9A%84%E6%96%B0%E5%BB%BA%E4%B8%8E%E5%90%88%E5%B9%B6

第一步：

新建一个git仓库：同时新建一个测试文件：test.txt

```
origin txt
```

提交到主分支：



第二步：

此时新建一个测试分支testing

```
git branch testing
```

此时分支内的内容test.txt的类容和master内容一样。



第三步：

将master分支的内容修改：添加一行　master modify　

```
origin txt
master modify　
```



第四步：

切换到testing分支：（此时的分支和未修改前是一样的）：添加一行：testing modify

```
origin txt
testing modify
```



### 此时可以将master分支合并到testing中。也可以将testing分支合并到master分支

#### 将testing分支合并到master分支：

此时：testing分支的test.txt文件内容如下

```
origin txt
testing modify
```

master分支的内容如下：

```
origin txt
master modify
```

直接再master分支合并testing分支

```
git checkout testing
```

结果如下，直接删除了master的内容将testing的内容替换到了master的内容中

```bash
EAD+pzhang36@CN-PF2QT03L MINGW64 /c/code/test (master)
$ git merge testing
Updating 13cdb60..fc815bc
Fast-forward
 test.txt | 3 ++-
 1 file changed, 2 insertions(+), 1 deletion(-)

EAD+pzhang36@CN-PF2QT03L MINGW64 /c/code/test (master)
$ cat test.txt
origin txt
testing modify
```

#### 将master分支合并到testing中

此时：testing分支的test.txt文件内容如下

```
origin txt
testing modify
```

master分支的内容如下：

```
origin txt
master modify
```

直接在testing分支合并master分支

结果如下，直接将将master分支的内容替换到了testing分支

```bash
EAD+pzhang36@CN-PF2QT03L MINGW64 /c/code/test (testing)
$ git merge master
Updating fc815bc..5a16cfe
Fast-forward
 test.txt | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

EAD+pzhang36@CN-PF2QT03L MINGW64 /c/code/test (testing)
$ cat test.txt
origin txt
master modify
```

