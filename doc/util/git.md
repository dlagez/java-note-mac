### 关联github

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

### add 加 commit

```
git commit -a -m 'made a change'
```

### 远程仓库的重命名与移除

```   console
git remote rename pb paul  # 重命名
git remote remove paul     # 移除远程分支
```

### 从远程仓库抓取数据

它只会抓取数据，并不会合并分支

```console
git fetch <remote>
```

自动抓取后合并该远程分支到当前分支

```
git pull 
```

### 已经提交到github的文件夹取消追踪,

 虽然会追踪, 但是还是需要在.gitignore里面设置一下,不然一直报红.

```
git rm -r --cached dir
```

### 合并提交

这次的改动比较小, 和上次的提交合并, 这个message会覆盖上次的信息.

```
git commit -a --amend -m "my message here"
```



### 版本回退：

```
git reset --hard HEAD
```

会删除所有的修改，变成提交时的状态。

### 撤销修改

你修改了一个文件，但是发现改错了，想丢弃修改类容（使用分支保存已经做的工作是更好的方法）

```
git checkout -- readme.txt
```

### 取消暂存commit的文件

```
git reset HEAD readme.txt
git commit --amend -m 'ad'
```

### 删除缓存区文件

已经add到暂存区的文件移除：该文件从暂存区移除，本地不会删除。

```
git rm -r --cached src/
```

使用了amend，推送不上去，这句话执行的后果就是在远程仓库中进行的相关修改会被删除，使远程仓库回到你本地仓库未修改之前的那个版本，   然后上传你基于本地仓库的修改。

```
git push -u origin master -f
```



### 查看修改

此命令比较的是

```console
git diff 工作目录中当前文件和暂存区域快照之间的差异
git diff --staged 查看已经add文件与暂存的文件差异：
git diff branchNmae 工作区与某分支的差异
git diff HEAD 工作区与HEAD指针指向的内容差异
git diff branch1 branch2 查看两个分支的差异 显示branch1的不同，比如branch1多出数据会显示绿色++
```

### 查看提交历史

```
git log
git log -p -2                      # 显示每次提交所引入的差异 -2 选项来只显示最近的两次提交
git log --oneline --decorate       # 简单的查看提交记录
git log --oneline --decorate --graph --all      # 查看提交记录图
git log --stat                     # 查看每次提交的变化
```



## 分支

```
git branch testing           # 新建一个分只
git checkout testing         #切换分支
git checkout -b hotfix       # 新建并转换到这个分支
git branch                   # 查看分支
git branch -a                # c
git merge testing            # 此时在分支master中，使用master分支合并其他分支
git branch -d hotfix         # 删除分支
```

提交分支到github

```
# 先切换到分支中
git branch testing
# 再提交
git push origin testing
```



图床使用token

```
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
```



加速：gh表示github，dlagez表示你的账户名，img表示仓库名。

```
https://cdn.jsdelivr.net/gh/dlagez/img@master
```

typora命令：

```
mac-- /usr/local/bin/node /usr/local/bin/picgo upload
win-- picgo upload
```

cdn 加速

```
https://cdn.jsdelivr.net/gh/dlagez/img@master
```





