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



### conda

```
cat  /usr/local/cuda/version.txt
nvcc --version
```



### vim

