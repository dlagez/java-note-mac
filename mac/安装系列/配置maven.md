maven下载地址：http://maven.apache.org/download.cgi

下载第一个即可

安装与配置：

首先解压，然后将解压的tomcat文件夹拷贝到 /opt/文件夹下

打开终端， vim ~/.zshrc 添加maven环境变量

```bash
# 由于我把tomcat放在了/opt/目录下，所以这样配置
export MAVEN_HOME=/opt/apache-maven-3.8.1
export PATH=$PATH:$MAVEN_HOME/bin
```

应用配置之后重启

```bash
source ~/.zshrc
```

测试配置成功没有

```bash
source ~/.zshrc
```

输出

```bash
roczhang@roczhangdeMac-mini ~ % mvn -v
Apache Maven 3.8.1 (05c21c65bdfed0f71a2f2ada8b84da59348c4c5d)
Maven home: /opt/apache-maven-3.8.1
Java version: 11.0.12, vendor: Azul Systems, Inc., runtime: /Library/Java/JavaVirtualMachines/zulu-11.jdk/Contents/Home
Default locale: zh_CN_#Hans, platform encoding: UTF-8
OS name: "mac os x", version: "11.4", arch: "aarch64", family: "mac"
```

表示maven配置成功

配置阿里云的源

```bash
vim /opt/apache-maven-3.8.1/conf/settings.xml 
```

将下面的源加入到配置文件即可

```java
<mirror>
    <id>alimaven</id>
    <mirrorOf>central</mirrorOf>
    <name>aliyun maven</name>
    <url>http://maven.aliyun.com/nexus/content/repositories/central/</url>
</mirror>
```



打开终端，在终端输入mvn help:system，下载一些默认的jar包到本地仓库，然后出现build success