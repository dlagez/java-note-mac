## docker

WIN

下载安装包：[Empowering App Development for Developers | Docker](https://www.docker.com/)

设置：win环境：[Manual installation steps for older versions of WSL | Microsoft Docs](https://docs.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package)



### install sqlserver: 

- articles:https://www.sqlservercentral.com/articles/docker-desktop-on-windows-10-for-sql-server-step-by-step
- docker link: https://hub.docker.com/_/microsoft-mssql-server

```
// pull
docker pull mcr.microsoft.com/mssql/server:2019-latest

// run
docker run --name sqlserver -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=Dlagez3133.." -e "MSSQL_PID=Enterprise" -p 1433:1433 -d mcr.microsoft.com/mssql/server

// test
docker exec -it sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa

// exec
exec -it sqlserver "bash"

// can't into 
docker exec -it sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P Dlagez3133..

登录名是：sa 密码Dlagez3133..
```



### install mysql

```
docker pull mysql:8.0.27
docker run -p 3306:3306 --name mysql -e MYSQL_ROOT_PASSWORD=password -d mysql:8.0.27 # run image
docker exec -it mysql bash # inside a Docker container
docker logs mysql # get log
mysql -u root -p # inside m
```



### install postgresql 

https://hub.docker.com/_/postgres

```cmd
docker pull postgres:9.4.26
#  -e TZ=PRC 设置时区为中国
docker run -p 15432:5432 --name postgres -e POSTGRES_PASSWORD=password -e TZ=PRC -d postgres:9.4.26

# 默认用户名是postgres 密码password
```



### install elasticsearch

```bash
docker pull elasticsearch:7.6.2
docker run -d --name elasticsearch  -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:7.6.2

// docker run -d --name elasticsearch --net elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:7.6.2
```



### install redis

```
docker pull redis
docker run -itd --name redis -p 6379:6379 redis
docker exec -it redis /bin/bash

```

### install kibana

```
docker pull kibana:7.6.2
docker run -d --name kibana -p 5601:5601 kibana:7.6.2
```



## git

ref：[link](https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247485544&idx=1&sn=afc9d9f72d811ec847fa64108d5c7412&scene=21#wechat_redirect)

**需求一，如何把`work dir`中的修改加入`stage`**。

这个是最简单，使用 **`git add`** 相关的命令就行了。顺便一提，`add`有个别名叫做`stage`，也就是说你可能见到`git stage`相关的命令，这个命令和`git add`命令是完全一样的。



**需求二，如何把`stage`中的修改还原到`work dir`中**。

这个需求很常见，也很重要，比如我先将当前`work dir`中的修改添加到`stage`中，然后又对`work dir`中的文件进行了修改，但是又后悔了，如何把`work dir`中的全部或部分文件还原成`stage`中的样子呢？

注意：`work dir`的修改会全部丢失，但是新加的文件不会被删除。

**风险等级：中风险。**

理由：在`work dir`做出的「修改」会被`stage`覆盖，无法恢复。所以使用该命令你应该确定`work dir`中的修改可以抛弃。

```bash
git checkout a.txt
git checkout .
```



**需求三，将`stage`区的文件添加到`history`区**。

```bash
git commit -m '一些描述'
```

再简单提一些常见场景， 比如说`commit`完之后，突然发现一些错别字需要修改，又不想为改几个错别字而新开一个`commit`到`history`区，那么就可以使用下面这个命令：

```bash
$ git commit --amend
```



**需求四，将`stage`区的文件还原到`work dir`区**。

这个需求很常见，比如说我用了一个`git add .`一股脑把所有修改加入`stage`，但是突然想起来文件`a.txt`中的代码我还没写完，不应该把它`commit`到`history`区，所以我得把它从`stage`中撤销，等后面我写完了再提交。

```bash
$ echo aaa >> a.txt; echo bbb >> b.txt;
$ git add .
$ git status
On branch master
Changes to be committed:
    modified:   a.txt
    modified:   b.txt
```

如何把`a.txt`从`stage`区还原出来呢？可以使用 **`git reset`** 命令：

```bash
$ git reset a.txt

$ git status
On branch master
Changes to be committed:
    modified:   b.txt

Changes not staged for commit:
    modified:   a.txt
```

你看，这样就可以把`a.txt`文件从`stage`区移出，这时候进行`git commit`相关的操作就不会把这个文件一起提交到`history`区了。



**需求六，将`history`区的历史提交还原到`work dir`中**。

这个场景，我说一个极端一点的例子：比如我从 GitHub 上`clone`了一个项目，然后乱改了一通代码，结果发现我写的代码根本跑不通，于是后悔了，干脆不改了，我想恢复成最初的模样，怎么办？

依然是使用`checkout`命令，但是和之前的使用方式有一些不同：

```bash
$ git checkout HEAD .
Updated 12 paths from d480c4f
```

这样，`work dir`和`stage`中所有的「修改」都会被撤销，恢复成`HEAD`指向的那个`history commit`。

注意，类似之前通过`stage`恢复`work dir`的`checkout`命令，这里撤销的也只是修改，新增的文件不会被撤销。

当然，只要找到任意一个`commit`的 HASH 值，`checkout`命令可就以将文件恢复成任一个`history commit`中的样子：

```bash
$ git checkout 2bdf04a some_test.go
Updated 1 path from 2bdf04a
# 前文的用法显示 update from index
```

比如，我改了某个测试文件，结果发现测试跑不过了，所以就把该文件恢复到了它能跑过的那个历史版本……

**风险等级：高风险。**

理由：这个操作会将指定文件在`work dir`的数据恢复成指定`commit`的样子，且会删除该文件在`stage`中的数据，都无法恢复，所以应该慎重使用。



### 关联github

```bash
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

### 版本回退：

```
git reset --hard HEAD
```

会删除所有的修改，变成提交时的状态。

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



### 分支

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

## linux

### linux查找命令

#### find

```
find /var -name test.file
	/roczhang/app/log.txt
```

#### locate

```
sudo updatedb
locate log.txt
	/roczhang/app/log.txt
    /usr/share/doc/cloud-init/examples/cloud-config-rsyslog.txt
    /usr/share/doc/util-linux/getopt_changelog.txt

```

#### grep

```
ps -aux|grep redis  # 查询进程号
```



### vim查找关键字

可在normal模式下直接按“/”进入查找模式，输入要查找的字符并按下回车，vim会跳到第一个匹配的位置。

按n查找下一个，按Shift+n查找上一个。

### 命令行下载文件

下面的下载方法会把整个html文件下载下来

```
wget https://github.com/macrozheng/mall-learning/blob/master/document/sql/mall.sql
```

在github里面点击raw即可跳转到只有文件内容的网页，这个时候再用wget命令

### 端口操作

查看某端口是否开放，没有返回就是没有开放

```
lsof -i:80
```

查看所有开放的端口

```
netstat -aptn
```

开启及关闭防火墙

```bash
sudo ufw enable # 开启
sudo ufw reload # 重启
sudo ufw disable # 关闭防火墙
```

查看防火墙状态

```bash
sudo ufw status
```

开放及关闭端口

```bash
sudo ufw allow 80 # 开放端口
sudo ufw delete allow 80 # 关闭防火墙
```

查看端口是否被占用

```
sudo lsof -i:端口号
```

### 设置文件夹权限

解释一下，其实整个命令的形式是
sudo chmod -（代表类型）×××（所有者）×××（组用户）×××（其他用户）

其中 -R 表示递归处理，*代表所有文件

```
sudo chmod 777 -R /usr/RocZhang/
```

sudo chmod 600 ××× （只有所有者有读和写的权限）
sudo chmod 644 ××× （所有者有读和写的权限，组用户只有读的权限）
sudo chmod 700 ××× （只有所有者有读和写以及执行的权限）
sudo chmod 666 ××× （每个人都有读和写的权限）
sudo chmod 777 ××× （每个人都有读和写以及执行的权限）

给文件赋予读写+执行权限

```bash
chmod 777 file
```

对目录和其子目录层次结构中的所有文件给用户增加读权限

```bash
chmod -R 777 的
```

### 压缩与解压

```
!tar -zcvf images.tar.gz /content/PyTorch-GAN/implementations/acgan/images
!unrar x /content/anime-WGAN-resnet-pytorch/data/faces.rar
jar -xvf game.war # 解压war包
```

### 查看文件夹下面的文件的数量

```
ls -l /content/gan_resnet/data/faces | grep "^-" | wc -l
```

### 用户操作

```
su root # 切换用户
sudo adduser roc
cat /etc/group  // 查看所有用户组
sudo cat /etc/shadow  // 查看所有用户
```

### apt使用

```
sudo apt-cache search java
sudo apt list | grep tomcat
sudo apt-get install java
sudo apt-get remove java
sudo apt-get purge XXX  # 卸载软件并删除配置文件
sudo apt-get autoremove  # 卸载软件并卸载不需要的包
```

### 软件下载

#### mysql 

腾讯云服务器远程可以登陆：用户：root 密码：password

```
sudo apt-get install mysql-server

sudo service mysql restart / sudo systemctl restart mysql
sudo netstat -tap|grep mysql # 查看m是否在运行
cd /etc/mysql
sudo cat debian.cnf # 即可查看初始用户名和密码
# debian-sys-maint JEuTuNNkEiZGGnMU

# 在mysql里面执行
alter user 'root'@'localhost' identified by 'Dlagez3133..'; #修改root用户的密码
select user,host,plugin from user; # 可以查看用户的权限情况
RENAME USER 'root'@'localhost' TO 'root'@'%';  # 改成「%」才能全局访问
sudo vim /etc/mysql/mysql.conf.d/mysqld.cnf  # 修改 my.cnf MySQL 配置文件 
#  改成这个样子即可 bind - address = 0.0.0.0
ALTER USER 'root'@'%' IDENTIFIED WITH mysql_native_password BY 'password';
sudo ufw allow 3306 # 开启防火墙
```

#### jdk

```
sudo apt install openjdk-11-jdk  # java --version 即可查看
```

#### tomcat

```
sudo apt install tomcat9
sudo service tomcat9 start
sudo service tomcat6 stop
Tomcat home directory : /usr/share/tomcat6
Tomcat base directory : /var/lib/tomcat6或/etc/tomcat6

/etc/tomcat6 - 全局配置?
/usr/share/tomcat6/ - 程序主目录?
/var/lib/ tomcat6/ - 工作主目录? http应在在这里配置

sudo vim /etc/tomcat9/server.xml  # x
```

#### redis

```
sudo apt install redis-server
ps -aux|grep redis
netstat -nlt|grep 6379 # 命令可以看到redis服务器状态
sudo /etc/init.d/redis-server status  # 命令可以看到Redis服务器状态
sudo service redis-server restart
配置文件为/etc/redis/redis.conf(在线安装推荐)

首先sudo vi /etc/redis/redis.conf
添加Redis的访问账号
Redis服务器默认是不需要密码的，假设设置密码为hzlarm。
去掉requirepass 前面的注释#，在后面添加密码
requirepass hzlarm

开启Redis的远程连接
注释掉绑定地址#bind 127.0.0.1

修改Redis的默认端口
port 6379

redis-cli k
```



#### nginx

```
sudo apt install nginx
nginx -v  // 查看版本
pm -ql nginx  // 查看安装目录
systemctl start nginx //启动
$sudo /etc/init.d/nginx stop //停止
$sudo /etc/init.d/nginx restart //重启

systemctl enable nginx  // 开机自启

nginx -s reload  # 向主进程发送信号，重新加载配置文件，热重启
nginx -s reopen	 # 重启 Nginx
nginx -s stop    # 快速关闭
nginx -s quit    # 等待工作进程处理完成后关闭
nginx -T         # 查看当前 Nginx 最终的配置
nginx -t -c <配置路径>    # 检查配置是否有问题，如果已经在配置目录，则不需要-c

systemctl start nginx    # 启动 Nginx
systemctl stop nginx     # 停止 Nginx
systemctl restart nginx  # 重启 Nginx
systemctl reload nginx   # 重新加载 Nginx，用于修改配置后
systemctl enable nginx   # 设置开机启动 Nginx
systemctl disable nginx  # 关闭开机启动 Nginx
systemctl status nginx   # 查看 Nginx 运行状态

如前面文件所示，Nginx的主配置文件为etc/nginx/nginx.conf,可以使用 cat命令命令进行查看cat -n nginx.conf
```

配置反向代理：ref:[link](https://dgideas.net/2020/configure-reverse-proxy-for-nginx-on-ubuntu-2004/)

Nginx 的默认配置文件位于目录 `/etc/nginx/sites-enabled/` 中。

Nginx 使用类似 JSON 的格式表示其配置文件。其中，对于网站的配置位于单独的 `server{}` 块中。一个典型示例如下：

```
server {
    listen 80 default_server;
    root /var/www/html;
    server_name dgideas.net;
    location / {
    }
}
```

![Untitled10](https://cdn.jsdelivr.net/gh/dlagez/img@master/Untitled10.png)

### 防火墙

tenxun云的防火墙和ubuntu里面的ufw是两个防火墙。

```
sudo ufw allow 3306
sudo ufw deny 25
sudo ufw status verbose
```



### 后台运行程序

nohup表示后台运行，>log.txt表示输出到这个文件，不在窗口输出。

```
sudo nohup java -jar blog-0.0.1-SNAPSHOT.jar >log.txt &
// s
snohup java -jar blog-0.0.1-SNAPSHOT.jar &
```

想把它停止的话，查找它的进程号即可

```
ps -ef|grep xxx.jar
或者 ps -aux | grep java
kill -9 1972459
```

查找的结果：

```
ubuntu@VM-16-10-ubuntu:/roczhang/app$ ps -ef | grep blog-0.0.1-SNAPSHOT.jar 
ubuntu    416681  415467  6 00:07 pts/0    00:00:20 java -jar blog-0.0.1-SNAPSHOT.jar
ubuntu    417798  415467  0 00:12 pts/0    00:00:00 grep --color=auto blog-0.0.1-SNAPSHOT.jar
```



github 图床key

```
ghp_nDA7Yb7n1oyfDihQ2dI---------------------dxpTwfcy1WH3JBePm
```

gitee图床key

```
ed413552b224df7781ad8af5417b6d7a
```



### conda

#### miniconda官网：[Miniconda — Conda documentation](https://docs.conda.io/en/latest/miniconda.html#)

#### 换源

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

#### 创建环境

```
conda create -n bigdata python=3.8

```

#### 临时使用pip源

```
-i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 安装jupyter

```
conda install jupyter notebook
```

#### 修改根目录

生成配置文件 

```
jupyter notebook --generate-config
```

修改配置文件jupyter_notebook_config.py中的 c.NotebookApp.notebook_dir = ‘’ 改为要修改的根目录把单引号换成双引号  c.NotebookApp.notebook_dir = "C:/roczhang"

#### 将环境添加到kernel，

```
pip install ipykernel
python -m ipykernel install --name bigdata --display-name "bigdata"
```

- `--display-name`指定jupyter notebook中显示的名字

升级包

```
conda update package
```



### vim



配置nginx的反向代理



beian

```HTML
<p>
				© 2022 ROC
				&nbsp;
				<a href="https://beian.miit.gov.cn/" target="_blank">鄂ICP备2022001317号-1</a>
				&nbsp;
				<a href="http://www.beian.gov.cn/portal/registerSystemInfo?recordcode=鄂公网安备 42011202002050号" target="_blank">
					<img src="//cdn.jsdelivr.net/gh/LIlGG/halo-theme-sakura@1.3.1/source/images/other/gongan.png">鄂公网安备 42011202002050号
				</a>	
			</p>
```



## mac

brew:

```
brew install go
```



下载的软件打不开：

```
sudo xattr -cr /Volumes/roczhang/download/V2RayX.app
sudo xattr -cr /Applications/V2RayX.app
```

安装pytorch：

```
conda install -c pytorch pytorch
```

### rar解压

```
opt + cmd + c 等于复制全路径
```

```
cmd+shift+. 显示隐藏文件
```

图床配置：

picgo ref： [link](https://picgo.github.io/PicGo-Doc/zh/guide/config.html#通过url上传)

picgo core ref: [link](https://picgo.github.io/PicGo-Core-Doc/)

cdn 加速：`https://cdn.jsdelivr.net/gh/dlagez/img@master`



picgo 图床插件

https://github.com/xlzy520/picgo-plugin-bilibili

https://github.com/PicGo/Awesome-PicGo



共享文件夹：

https://sspai.com/post/61388
