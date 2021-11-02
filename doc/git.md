在本地已经新建了项目，想关联到github。

```
1.在github先创建一个空项目，不要任何设置。

2. 将本地的项目git init 并将文件提交。
3. 在项目文件夹内使用 
git remote add origin git@github.com:dlagez/demo_JAVA.git

// 把本地库的内容推送到远程，用git push命令，实际上是把当前分支master推送到远程。
3.git push -u origin master
注：由于远程库是空的，我们第一次推送master分支时，加上了-u参数，Git不但会把本地的master分支内容推送的远程新的master分支，还会把本地的master分支和远程的master分支关联起来，在以后的推送或者拉取时就可以简化命令。

//从现在起，只要本地作了提交，就可以通过命令：
4.git push origin master

// 查看远程仓库信息
5. git remote -v

// 删除和远程仓库的连接
6. git remote rm origin
```

查看文件的差异：

```
git diff 文件名
```



### 版本回退：

git log 可以查看commit的版本。用`git reflog`查看命令历史

```bash
roczhang@roczhang-mac mall % git log
commit ec90e2258b414c188d6dbe42b9deb395cdcf89ad (HEAD -> master)
Author: roczhang <1587839905@qq.com>
Date:   Thu Sep 9 19:23:50 2021 +0800

    add readme

commit c01552ec72a354f7df3d45ad922d50cca8510294 (origin/master)
Author: roczhang <1587839905@qq.com>
Date:   Thu Sep 9 19:16:35 2021 +0800

    初始化项目基本骨架
```

add readme时当前版本。用`HEAD`表示当前版本，上一个版本就是`HEAD^`，上上一个版本就是`HEAD^^`，当然往上100个版本写100个`^`比较容易数不过来，所以写成`HEAD~100`。

比如我现在开发项目发现错了很多地方。想删除修改。把项目回退到上一次提交的版本，也就是当前版本。

```
git reset --hard HEAD
```

会删除所有的修改，变成提交时的状态。

### 撤销修改

1.你修改了一个文件，但是发现改错了，想丢弃修改类容

```
git checkout -- readme.txt
```

2.我修改了文件，并且已经add到暂存区了。

用命令`git reset HEAD <file>`可以把暂存区的修改撤销掉（unstage），重新放回工作区：

```
git reset HEAD readme.txt
```

然后丢弃修改和上面一样

```
git checkout -- readme.txt
```



### 删除文件

如果有大文件，或者无用的文件已经add了，但是想删除它怎么办（保留文件在电脑里面）

此时我们add了两个文件到暂存区。

```
roczhang@roczhang-mac git-demo % git status
On branch master

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
	new file:   readme.md
	new file:   src/test.txt
```

```
roczhang@roczhang-mac git-demo % git rm -r --cached src/ 

rm 'src/test.txt'
```

再次查看时已经显示src文件夹没有被跟踪了。

```
roczhang@roczhang-mac git-demo % git status
On branch master

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
	new file:   readme.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	src/
```

下次提交时，该文件将消失并且不再被跟踪



### 查看修改

```console
git diff
```

此命令比较的是工作目录中当前文件和暂存区域快照之间的差异

如果已经进行add操作了，使用git diff没有显示的。

查看已经add文件的差异使用：

```console
git diff --staged
```



两个电脑同时记笔记：

我有这样一个需求，就是我的mac和工作的笔记使用的是同一个git仓库。他两修改肯定会找成冲突。

解决方案：在mac上面可以随时修改，但是不要add和commit，在add和commit之前将仓库pull一下。

（add之前要保证一个mac或者公司电脑上的修改全部push到github上面）。之后再提交mac上的修改。