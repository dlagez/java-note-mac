起因，Homebrew服务器在国外，国内下载报错443，哪怕更改hosts，下载速度也是龟速中的龟速，在度娘和谷歌找了很多资料，发现都没有一个比较完善的解决办法，索性把所有查到的资料整理到一起，达到100%能正常安装并且使用brew。

方法：

1.打开Homebrew官网获取安装链接：https://brew.sh/index_zh-cn
2.用浏览器打开安装链接里面的地址：

https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh
3.把.sh里面的整个脚本代码复制到本地.txt文件

4.找到里面的2个重点git地址替换成国内阿里云镜像地址：

![img](../%E5%AE%89%E8%A3%85homebrew.assets/20210822202440109.png)

![img](../%E5%AE%89%E8%A3%85homebrew.assets/20210822202539241.png)


        把对应位置的url替换为：注意，如下要加上后缀.git

```
https://mirrors.aliyun.com/homebrew/homebrew-core.git
https://mirrors.aliyun.com/homebrew/brew.git
```

5.在本地txt文件里面把url替换后，把文件改为install.sh文件

6.运行本地install.sh：

/bin/bash /这里写你的文件位置/install.sh
        然后等待5-10分钟，Homebrew即可安装完毕！注意：中途会要求你输入系统密码！

7.配置环境变量：打开命令行=>如果没有zshrc就创建一个：touch ~/.zshrc 其他系统根据自己的方式设置

```
    vim ~/.zshrc
```

```
export PATH=/opt/homebrew/bin:$PATH
```

```
source ~/.zshrc
```

8.至此Homebrew已经安装完毕，并且可以使用了，但是我们可以把这种骚操作变为正规化：正常配置好国内阿里云镜像：

    # 替换brew.git:
    cd "$(brew --repo)"
    git remote set-url origin https://mirrors.aliyun.com/homebrew/brew.git
    # 替换homebrew-core.git:
    cd "$(brew --repo)/Library/Taps/homebrew/homebrew-core"
    git remote set-url origin https://mirrors.aliyun.com/homebrew/homebrew-core.git
    # 应用生效
    brew update
    # 替换homebrew-bottles:
    echo 'export HOMEBREW_BOTTLE_DOMAIN=https://mirrors.aliyun.com/homebrew/homebrew-bottles' >> ~/.zshrc
    source ~/.zshrc

9.操作完上面的命令过后，全新的国内爽用的Homebrew就出炉了，开启你的工具安装之旅吧！！！
10.换源

```
echo 'export HOMEBREW_BOTTLE_DOMAIN=https://mirrors.ustc.edu.cn/homebrew-bottles' >> ~/.zshrc
 
source ~/.zshrc
```

