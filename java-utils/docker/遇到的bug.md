### mac的shell不能使用ubuntu的docker命令。

```
roczhang@roczhang:~$ docker images
Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/images/json": dial unix /var/run/docker.sock: connect: permission denied
```

解决方法：

```
roczhang@roczhang:~$ sudo docker images
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
mysql        latest    c60d96bd2b77   8 weeks ago   514MB
```



### 配置文件挂载出错

```
sudo docker run -p 3306:3306 --name mysql -v /mydata/mysql/data:/var/lib/mysql -v /mydata/mysql/conf:/etc/mysql -e MYSQL_ROOT_PASSWORD=password -dit mysql
```

下面这条语句可以运行

```
sudo docker run -p 3306:3306 --name mysql -v /mydata/mysql/data:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=qq1597357 -d mysql
```

