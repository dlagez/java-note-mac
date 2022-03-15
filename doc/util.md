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

### 介绍：

ref：[link](https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484490&idx=1&sn=f313a6bc4f577de63b7c7a81eda0f343&scene=21#wechat_redirect)

![Image](https://cdn.jsdelivr.net/gh/dlagez/img@master/640.png)

上图是 Linux 文件系统的一个整体结构，无论是什么 Linux 发行版，根目录`/`基本上就是这些文件。不要害怕这么多文件夹，大部分都不需要你操心，只要大概了解它们是干啥的就行了。

#### /bin 和 /sbin

`bin`是`Binary`的缩写，**存放着可执行文件或可执行文件的链接**（类似快捷方式）



与`/bin`类似的是`/sbin`目录，System Binary 的缩写，这里存放的命令可以对系统配置进行操作。普通用户可能可以使用这里的命令查看某些系统状态，但是如果想更改配置，就需要`sudo`授权或者切换成超级用户。

可以看到一些熟悉的命令，比如`ifconfig`,`iptables`。普通用户可以使用`ifconfig`查看网卡状态，但是想配置网卡信息，就需要授权了。

#### /boot

**这里存放系统启动需要的文件**，你可以看到`grub`文件夹，它是常见的开机引导程序。我们不应该乱动这里的文件。

#### /dev

`dev`是`device`的缩写，这里存放着所有的**设备文件**。在 Linux 中，所有东西都是以文件的形式存在的，包括硬件设备。

比如说，`sda`,`sdb`就是我电脑上的两块硬盘，后面的数字是硬盘分区：

鼠标、键盘等等设备也都可以在这里找到。

#### /etc

这个目录经常使用，存放很多程序的**配置信息**，比如包管理工具 apt：

在`/etc/apt`中就存放着对应的配置，比如说镜像列表（我配置的阿里云镜像）：

如果你要修改一些系统程序的配置，十有八九要到`etc`目录下寻找。

#### /lib

`lib`是 Library 的缩写，包含 bin 和 sbin 中可执行文件的依赖，类似于 Windows 系统中存放`dll`文件的库。

也可能出现`lib32`或`lib64`这样的目录，和`lib`差不多，只是操作系统位数不同而已。

#### /media

这里会有一个以你用户名命名的文件夹，**里面是自动挂载的设备**，比如 U 盘，移动硬盘，网络设备等。

比如说我在电脑上插入一个 U 盘，系统会把 U 盘自动给我挂载到`/media/fdl`这个文件夹里（我的用户名是 fdl），如果我要访问 U 盘的内容，就可以在那里找到。

#### /mnt

这也是和设备挂载相关的一个文件夹，一般是空文件夹。`media`文件夹是系统自动挂载设备的地方，这里是你**手动挂载设备**的地方。

比如说，刚才我们在`dev`中看到了一大堆设备，你想打开某些设备看看里面的内容，就可以通过命令把设备挂载到`mnt`目录进行操作。

不过一般来说，现在的操作系统已经很聪明了，像挂载设备的操作几乎都不用你手动做，系统应该帮你自动挂载到`media`目录了。

#### /opt

`opt`是 Option 的缩写，这个文件夹的使用比较随意，一般来说我们自己在浏览器上下载的软件，安装在这里比较好。当然，包管理工具下载的软件也可能被存放在这里。

比如我在这里存放了 Chrome 浏览器（google），网易云音乐（netease），CLion IDE 等等软件。

#### /proc

`proc`是`process`的缩写，这里存放的是全部**正在运行程序的状态信息**。

你会发现`/proc`里面有一大堆数字命名的文件夹，这个数字其实是 Process ID（PID），文件夹里又有很多文件。

前面说过，Linux 中一切都以文件形式储存，类似`/dev`，这里的文件也**不是真正的文件**，而是程序和内核交流的一些信息。比如说我们可以查看当前操作系统的版本，或者查看 CPU 的状态：

#### /root

这是超级用户的家目录，普通用户需要授权才能访问。

区别一下 root 用户和根目录的区别哈，root 用户就是 Linux 系统的超级用户（Super User），而根目录是指 / 目录，整个文件系统的「根部」。

#### /run 和 /sys

用来存储某些程序的运行时信息和系统需要的一些信息。比如说下面这个路径有一个名为 brightness 的文件：

```
sudo vim /sys/devices/pci0000:00/
    0000:00:02.0/drm/card0/card0-eDP-1/
    intel_backlight/brightness
```

里面存储着一个数字，是你的显卡亮度，你修改这个数字，屏幕亮度就会随之变化。

需要注意的是，这两个位置的数据都存储在内存中，所以一旦重启，`/run`和`/sys`目录的信息就会丢失，所以不要试图在这里存放任何文件。

#### /srv

`srv`是`service`的缩写，主要用来**存放服务数据**

对于桌面版 Linux 系统，这个文件夹一般是空的，但是对于 Linux 服务器，Web 服务或者 FTP 文件服务的资源可以存放在这里。

#### /tmp

`tmp`是`temporary`的缩写，存储一些程序的**临时文件**。

临时文件可能起到很重要的作用。比如经常听说某同学的 Word 文档崩溃了，好不容易写的东西全没了，Linux 的很多文本编辑器都会在`/tmp`放一份当前文本的 copy 作为临时文件，如果你的编辑器意外崩溃，还有机会在`/tmp`找一找临时文件抢救一下。

当然，`tmp`文件夹在系统重启之后会自动被清空，如果没有被清空，说明系统删除某些文件失败，也许需要你手动删除一下。

#### /usr

`usr`是 Universal System Resource 的缩写，这里存放的是一些**非系统必须的资源**，比如用户安装的应用程序。

`/usr`和`/usr/local`目录中又含有`bin`和`sbin`目录，也是存放可执行文件（命令），但和根目录的`bin`和`sbin`不同的是，这里大都是用户使用的工具，而非系统必须使用的。

比如说`/usr/bin`中含有我通过包管理工具安装的应用程序 Chrome 浏览器和 goldendict 字典的可执行文件：

值得一提的是，如果使用 Linux 桌面版，**有时候在桌面找不到应用程序的快捷方式**，就需要在`/usr/share/applications`中手动配置桌面图标文件：

#### /var

`var`是`variable`的缩写，这个名字是历史遗留的，现在该目录最主要的作用是**存储日志（log）信息**，比如说程序崩溃，防火墙检测到异常等等信息都会记录在这里。

这是我的`/var/log`目录，可以看到很多系统工具的 log 文件（夹）：

日志文件不会自动删除，也就是说随着系统使用时间的增长，你的`var`目录占用的磁盘空间会越来越大，也许需要适时清理一下。

#### /home

最后说`home`目录，这是普通用户的家目录。在桌面版的 Linux 系统中，用户的家目录会有下载、视频、音乐、桌面等文件夹，这些没啥可说的，我们说一些比较重要的隐藏文件夹（Linux 中名称以`.`开头就是隐藏文件）。



#### 最后总结

如果修改系统配置，就去`/etc`找，如果修改用户的应用程序配置，就在用户家目录的隐藏文件里找。

你在命令行里可以直接输入使用的命令，其可执行文件一般就在以下几个位置：

```
/bin    
/sbin
/usr/bin
/usr/sbin
/usr/local/bin
/usr/local/sbin
/home/USER/.local/bin
/home/USER/.local/sbin
```

如果你写了一个脚本/程序，想在任何时候都能直接调用，可以把这个脚本/程序添加到上述目录中的某一个。

如果某个程序崩溃了，可以到`/val/log`中尝试寻找出错信息，到`/tmp`中寻找残留的临时文件。

设备文件在`/dev`目录，但是一般来说系统会自动帮你挂载诸如 U 盘之类的设备，可以到`/media`文件夹访问设备内容。







### vim

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





## Linux shell

### 基础知识

#### 一、标准输入和参数的区别

比如说，我现在有个自动连接宽带的 shell 脚本`connect.sh`，存在我的家目录：

```
$ where connect.sh
/home/fdl/bin/connect.sh
```

错误做法：

```
$ where connect.sh | rm
```

正确做法：

```
$ rm $(where connect.sh)
```

前者试图将`where`的结果连接到`rm`的标准输入，后者试图将结果作为命令行参数传入。

**标准输入就是编程语言中诸如`scanf`或者`readline`这种命令；而参数是指程序的`main`函数传入的`args`字符数组**。

管道符和重定向符是将数据作为程序的标准输入，而`$(cmd)`是读取`cmd`命令输出的数据作为参数

用刚才的例子说，`rm`命令源代码中肯定不接受标准输入，而是接收命令行参数，删除相应的文件。

作为对比，`cat`命令是既接受标准输入，又接受命令行参数：

```
$ cat filename
...file text...

$ cat < filename
...file text...

$ echo 'hello world' | cat
hello world
```

**如果命令能够让终端阻塞，说明该命令接收标准输入，反之就是不接受**，比如你只运行`cat`命令不加任何参数，终端就会阻塞，等待你输入字符串并回显相同的字符串。





#### 二、后台运行程序

比如说你远程登录到服务器上，运行一个 Django web 程序：

```
$ python manager.py runserver 0.0.0.0
Listening on 0.0.0.0:8080...
```

现在你可以通过服务器的 IP 地址测试 Django 服务，但是终端此时就阻塞了，你输入什么都不响应，除非输入 Ctrl-C 或者 Ctrl-/ 终止 python 进程。

可以在命令之后加一个`&`符号，这样命令行不会阻塞，可以响应你后续输入的命令，但是如果你退出服务器的登录，就不能访问该网页了。

如果你想在退出服务器之后仍然能够访问 web 服务，应该这样把命令包裹成这样`(cmd &)`：

```
$ (python manager.py runserver 0.0.0.0 &)
Listening on 0.0.0.0:8080...

$ logout
```

**底层原理是这样的**：

每一个命令行终端都是一个 shell 进程，你在这个终端里执行的程序实际上都是这个 shell 进程分出来的子进程。正常情况下，shell 进程会阻塞，等待子进程退出才重新接收你输入的新的命令。加上`&`号，只是让 shell 进程不再阻塞，可以继续响应你的新命令。但是无论如何，你如果关掉了这个 shell 命令行端口，依附于它的所有子进程都会退出。

而`(cmd &)`这样运行命令，则是将`cmd`命令挂到一个`systemd`系统守护进程名下，认`systemd`做爸爸，这样当你退出当前终端时，对于刚才的`cmd`命令就完全没有影响了。

类似的，还有一种后台运行常用的做法是这样：

```
$ nohup some_cmd &
```

`nohup`命令也是类似的原理，不过通过我的测试，还是`(cmd &)`这种形式更加稳定。



#### 三、单引号和双引号的区别

不同的 shell 行为会有细微区别，但有一点是确定的，**对于`$`，`(`，`)`这几个符号，单引号包围的字符串不会做任何转义，双引号包围的字符串会转义**。

shell 的行为可以测试，使用`set -x`命令，会开启 shell 的命令回显，你可以通过回显观察 shell 到底在执行什么命令：



这里没看懂！

可见 `echo $(cmd)` 和 `echo "$(cmd)"`，结果差不多，但是仍然有区别。注意观察，双引号转义完成的结果会自动增加单引号，而前者不会。

**也就是说，如果 `$` 读取出的参数字符串包含空格，应该用双引号括起来，否则就会出错**。

#### 四、sudo 找不到命令

有时候我们普通用户可以用的命令，用`sudo`加权限之后却报错 command not found：

```
$ connect.sh
network-manager: Permission denied

$ sudo connect.sh
sudo: command not found
```

原因在于，`connect.sh`这个脚本仅存在于该用户的环境变量中：

```
$ where connect.sh 
/home/fdl/bin/connect.sh
```

**当使用`sudo`时，系统认为是 root 用户在执行命令，所以会去搜索 root 用户的环境变量**，而这个脚本在 root 的环境变量目录中当然是找不到的。

解决方法是使用脚本文件的路径，而不是仅仅通过脚本名称：

```
$ sudo /home/fdl/bin/connect.sh
```



### 小技巧

#### 输入相似文件名太麻烦

用花括号括起来的字符串用逗号连接，可以自动扩展，非常有用，直接看例子：

```bash
$ echo {one,two,three}file
onefile twofile threefile

$ echo {one,two,three}{1,2,3}
one1 one2 one3 two1 two2 two3 three1 three2 three3
```

你看，花括号中的每个字符都可以和之后（或之前）的字符串进行组合拼接，**注意花括号和其中的逗号不可以用空格分隔，否则会被认为是普通的字符串对待**。

这个技巧有什么实际用处呢？最简单实用的就是给`cp`,`mv`,`rm`等命令扩展参数：

```bash
$ cp /very/long/path/file{,.bak}
# 给 file 复制一个叫做 file.bak 的副本

$ rm file{1,3,5}.txt
# 删除 file1.txt file3.txt file5.txt

$ mv *.{c,cpp} src/
# 将所有 .c 和 .cpp 为后缀的文件移入 src 文件夹
```

#### 输入路径名称太麻烦

**用`cd -`返回刚才待的目录**，直接看例子吧：

```bash
$ pwd
/very/long/path
$ cd # 回到家目录瞅瞅
$ pwd
/home/labuladong
$ cd - # 再返回刚才那个目录
$ pwd
/very/long/path
```

**特殊命令`!$`会替换成上一次命令最后的路径**，直接看例子：

```
# 没有加可执行权限
$ /usr/bin/script.sh
zsh: permission denied: /usr/bin/script.sh

$ chmod +x !$
chmod +x /usr/bin/script.sh
```

**特殊命令`!*`会替换成上一次命令输入的所有文件路径**，直接看例子：

```
# 创建了三个脚本文件
$ file script1.sh script2.sh script3.sh

# 给它们全部加上可执行权限
$ chmod +x !*
chmod +x script1.sh script2.sh script3.sh
```

**可以在环境变量`CDPATH`中加入你常用的工作目录**，当`cd`命令在当前目录中找不到你指定的文件/目录时，会自动到`CDPATH`中的目录中寻找。

比如说我常去家目录，也常去`/var/log`目录找日志，可以执行如下命令：

```
$ export CDPATH='~:/var/log'
# cd 命令将会在 ~ 目录和 /var/log 目录扩展搜索

$ pwd
/home/labuladong/musics
$ cd mysql
cd /var/log/mysql
$ pwd
/var/log/mysql
$ cd my_pictures
cd /home/labuladong/my_pictures
```

这个技巧是十分好用的，这样就免了经常写完整的路径名称，节约不少时间。

需要注意的是，以上操作是 bash 支持的，其他主流 shell 解释器当然都支持扩展`cd`命令的搜索目录，但可能不是修改`CDPATH`这个变量，具体的设置方法可以自行搜索。



#### 输入重复命令太麻烦

**使用特殊命令`!!`，可以自动替换成上一次使用的命令**：

```
$ apt install net-tools
E: Could not open lock file - open (13: Permission denied)

$ sudo !!
sudo apt install net-tools
[sudo] password for labuladong:
```

有的命令很长，一时间想不起来具体参数了怎么办？



### 管道符

#### > 和 >> 重定向符的坑

先说第一个问题，执行如下命令会发生什么？

```
$ cat file.txt > file.txt
```

**实际上，上述命令运行的结果是清空`file.txt`文件中的内容**。



所以执行`cat file.txt > file.txt`这个命令时，shell 会先打开`file.txt`，由于重定向符号是`>`，所以文件中的内容会被清空，然后 shell 将`cat`命令的标准输出设置为`file.txt`，这时候`cat`命令才开始执行。

也就是如下过程：

1、shell 打开`file.txt`并清空其内容。

2、shell 将`cat`命令的标准输出指向`file.txt`文件。

3、shell 执行`cat`命令，读了一个空文件。

4、`cat`命令将空字符串写入标准输出（`file.txt`文件）。

所以，最后的结果就是`file.txt`变成了空文件。



我们知道，`>`会清空目标文件，`>>`会在目标文件尾部追加内容，**那么如果将重定向符`>`改成`>>`会怎样呢**？

```
$ echo hello world > file.txt # 文件中只有一行内容
$ cat file.txt >> file.txt # 这个命令会死循环
```

`file.txt`中首先被写入一行内容，执行`cat file.txt >> file.txt`后预期的结果应该是两行内容。

但是很遗憾，运行结果并不符合预期，而是会死循环不断向`file.txt`中写入 hello world，文件很快就会变得很大，只能用 Control+C 停止命令。

这就有意思了，为什么会死循环呢？其实稍加分析就可以想到原因：

首先要回忆`cat`命令的行为，如果只执行`cat`命令，就会从命令行读取键盘输入的内容，每次按下回车，`cat`命令就会回显输入，也就是说，`cat`命令是逐行读取数据然后输出数据的。

那么，`cat file.txt >> file.txt`命令的执行过程如下：

1、打开`file.txt`，准备在文件尾部追加内容。

2、将`cat`命令的标准输出指向`file.txt`文件。

3、`cat`命令读取`file.txt`中的一行内容并写入标准输出（追加到`file.txt`文件中）。

4、由于刚写入了一行数据，`cat`命令发现`file.txt`中还有可以读取的内容，就会重复步骤 3。

**以上过程，就好比一边遍历列表，一遍往列表里追加元素一样，永远遍历不完，所以导致我们的命令死循环**。



#### > 重定向符和 | 管道符配合

我们经常会遇到这样的需求：截取文件的前 XX 行，其余的都删除。

在 Linux 中，`head`命令可以完成截取文件前几行的功能：

```
$ cat file.txt # file.txt 中有五行内容
1
2
3
4
5
$ head -n 2 file.txt # head 命令读取前两行
1
2
$ cat file.txt | head -n 2 # head 也可以读取标准输入
1
2
```

如果我们想保留文件的前 2 行，其他的都删除，可能会用如下命令：

```
$ head -n 2 file.txt > file.txt
```

前文 [Linux 进程和文件描述符](http://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484887&idx=1&sn=08860c04f6e79f363f4414a2c605b296&chksm=9bd7fbdfaca072c961e057f204b82eaa168d32b7b26578bacd902bfccc12a3004e594fdd4b3d&scene=21#wechat_redirect) 也说过管道符的实现原理，本质上就是将两个命令的标准输入和输出连接起来，让前一个命令的标准输出作为下一个命令的标准输入。

但是，如果你认为这样写命令可以得到预期的结果，那可能是因为你认为管道符连接的命令是串行执行的，这是一个常见的错误，**实际上****管道符连接的多个命令是并行执行的**。



你可能以为，shell 会先执行`cat file.txt`命令，正常读取`file.txt`中的所有内容，然后把这些内容通过管道传递给`head -n 2 > file.txt`命令。

虽然这时候`file.txt`中的内容会被清空，但是`head`并没有从文件中读取数据，而是从管道读取数据，所以应该可以向`file.txt`正确写入两行数据。

但实际上，上述理解是错误的，shell 会并行执行管道符连接的命令，比如说执行如下命令：

```
$ sleep 5 | sleep 5
```

shell 会同时启动两个`sleep`进程，所以执行结果是睡眠 5 秒，而不是 10 秒。

这是有点违背直觉的，比如这种常见的命令：

```
$ cat filename | grep 'pattern'
```

直觉好像是先执行`cat`命令一次性读取了`filename`中所有的内容，然后传递给`grep`命令进行搜索。

但实际上是`cat`和`grep`命令是同时执行的，之所以能得到预期的结果，是因为`grep 'pattern'`会阻塞等待标准输入，而`cat`通过 Linux 管道向`grep`的标准输入写入数据。



比如说只保留`file.txt`文件中的头两行，可以这样写代码：

```
# 先把数据写入临时文件，然后覆盖原始文件
$ cat file.txt | head -n 2 > temp.txt && mv temp.txt file.txt
```

**这是最简单，最可靠，万无一失的方法**。

你如果嫌这段命令太长，也可以通过`apt/brew/yum`等包管理工具安装`moreutils`包，就会多出一个`sponge`命令，像这样使用：

```
# 先把数据传给 sponge，然后由 sponge 写入原始文件
$ cat file.txt | head -n 2 | sponge file.txt
```

`sponge`这个单词的意思是海绵，挺形象的，它会先把输入的数据「吸收」起来，最后再写入`file.txt`，核心思路和我们使用临时文件时类似的，这个「海绵」就好比一个临时文件，就可以避免同时打开同一个文件进行读写的问题。



总结：

- 管道符连接的多个命令是并行执行的
- 管道符本质上就是将两个命令的标准输入和输出连接起来，让前一个命令的标准输出作为下一个命令的标准输入。
- `>`会清空目标文件，`>>`会在目标文件尾部追加内容
- **标准输入就是编程语言中诸如`scanf`或者`readline`这种命令；而参数是指程序的`main`函数传入的`args`字符数组**
- 如果命令能够让终端阻塞，说明该命令接收标准输入，反之就是不接受。
- 每一个命令行终端都是一个 shell 进程，你在这个终端里执行的程序实际上都是这个 shell 进程分出来的子进程。终端退出，子进程结束。











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
