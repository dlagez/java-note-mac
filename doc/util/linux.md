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
```

