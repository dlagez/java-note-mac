互换友链

网站名称：小飞博客 网站地址：http://dlage.cn/

网站描述：记录个人成长。 

网站logo：https://static.xffjs.com/xffjs/static/front/images/logo.png 

站长邮箱：[mr.roczhang@outlook.com](mr.roczhang@outlook.com)



```
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
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
```

gitee图床key

```
ed413552b224df7781ad8af5417b6d7a
```





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





图床配置：

picgo ref： [link](https://picgo.github.io/PicGo-Doc/zh/guide/config.html#通过url上传)

picgo core ref: [link](https://picgo.github.io/PicGo-Core-Doc/)

cdn 加速：`https://cdn.jsdelivr.net/gh/dlagez/img@master`



picgo 图床插件

https://github.com/xlzy520/picgo-plugin-bilibili

https://github.com/PicGo/Awesome-PicGo



