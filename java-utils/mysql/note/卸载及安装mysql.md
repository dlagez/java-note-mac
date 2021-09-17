查看依赖项

```
dpkg --list|grep mysql
```

卸载

```
sudo apt-get remove mysql-common

sudo apt-get autoremove --purge mysql-server-8.0
```

再次查看依赖项，然后继续使用卸载即可。

安装mysql（在服务器）

```
wget https://repo.mysql.com/mysql-apt-config_0.8.13-1_all.deb
sudo dpkg -i mysql-apt-config_0.8.13-1_all.deb
sudo apt-get update
sudo apt install mysql-server
```

## 2.1 初始化配置

```
sudo mysql_secure_installation
```

配置项较多，如下所示：

```
#1
VALIDATE PASSWORD PLUGIN can be used to test passwords...
Press y|Y for Yes, any other key for No: N (选择N ,不会进行密码的强校验)

#2
Please set the password for root here...
New password: (输入密码)
Re-enter new password: (重复输入)

#3
By default, a MySQL installation has an anonymous user,
allowing anyone to log into MySQL without having to have
a user account created for them...
Remove anonymous users? (Press y|Y for Yes, any other key for No) : N (选择N，不删除匿名用户)

#4
Normally, root should only be allowed to connect from
'localhost'. This ensures that someone cannot guess at
the root password from the network...
Disallow root login remotely? (Press y|Y for Yes, any other key for No) : N (选择N，允许root远程连接)

#5
By default, MySQL comes with a database named 'test' that
anyone can access...
Remove test database and access to it? (Press y|Y for Yes, any other key for No) : N (选择N，不删除test数据库)

#6
Reloading the privilege tables will ensure that all changes
made so far will take effect immediately.
Reload privilege tables now? (Press y|Y for Yes, any other key for No) : Y (选择Y，修改权限立即生效)
```

2.2检查mysql服务状态

```
systemctl status mysql.service
```

## 3.1配置远程访问

在Ubuntu下MySQL缺省是只允许本地访问的，使用workbench连接工具是连不上的； 如果你要其他机器也能够访问的话，需要进行配置；

找到 bind-address 修改值为 0.0.0.0(如果需要远程访问)

注意这里的登陆数据库只能用sudo

```
sudo vi /etc/mysql/mysql.conf.d/mysqld.cnf #找到 bind-address 修改值为 0.0.0.0(如果需要远程访问)
sudo /etc/init.d/mysql restart #重启mysql
sudo mysql -uroot -p
```

要想使用普通用户登陆：需要创建数据库管理员

```
$ sudo mysql -u root # I had to use "sudo" since is new installation

mysql> USE mysql;
mysql> CREATE USER 'YOUR_SYSTEM_USER'@'localhost' IDENTIFIED BY 'YOUR_PASSWD';
mysql> GRANT ALL PRIVILEGES ON *.* TO 'YOUR_SYSTEM_USER'@'localhost';
mysql> UPDATE user SET plugin='auth_socket' WHERE User='YOUR_SYSTEM_USER';
mysql> FLUSH PRIVILEGES;
mysql> exit;

$ sudo service mysql restart
```

Remember that if you use option #2 you'll have to connect to mysql as your system username (`mysql -u YOUR_SYSTEM_USER`)

现在ubuntu有两个用户：

root ：登陆方式：sudo mysql -uroot -p             password

roczhang :登陆方式： mysql -u roczhang -p          password

1 输入用户密码

\#切换数据库

```
mysql>use mysql;
```

\#查询用户表命令：

```
mysql>select User,authentication_string,Host from user;
```

\#查看状态

```
select host,user,plugin from user;
```

\#设置权限与密码

```
#使用mysql_native_password修改加密规则
mysql> ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';

#更新一下用户的密码
mysql> ALTER USER 'root'@'localhost' IDENTIFIED BY 'password' PASSWORD EXPIRE NEVER; 
#  执行这个语句必须mysql>use mysql;
mysql> UPDATE user SET host = '%' WHERE user = 'root'; #允许远程访问

#刷新cache中配置 刷新权限
mysql>flush privileges; 
mysql>quit;
```

3.2新建数据库和用户

```
##1 创建数据库studentService
CREATE DATABASE studentService;
##2 创建用户teacher(密码admin) 并赋予其studentService数据库的远程连接权限
GRANT ALL PRIVILEGES ON teacher.* TO studentService@% IDENTIFIED BY "admin";
```

3.3mysql服务命令

```
#检查服务状态
systemctl status mysql.service
或
sudo service mysql status
```

mysql服务启动停止

```
#停止
sudo service mysql stop
#启动
sudo service mysql start
```