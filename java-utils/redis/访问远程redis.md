修改Redis配置文件/etc/redis/redis.conf，找到bind那行配置：

```
bind 127.0.0.1
```


去掉#注释并改为：

```
bind 0.0.0.0
```

指定配置文件然后重启Redis服务即可：

```
$ sudo redis-server /etc/redis/redis.conf
```

开放端口：ubuntu

```
sudo ufw allow 6379 # 开放端口
```

