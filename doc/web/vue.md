### 安装（mac）：

这里首先需要安装npm，直接安装node即可同时安装npm。

https://nodejs.org/en/download/ 下载macos installer

安装完后执行下面的命令即可查看版本

```
npm -v
```

安装yarn

```
sudo npm install -g yarn
```

安装官方的cli

```
sudo npm install -g @vue/cli
```



创建项目：3步

```
vue init webpack demo
```

- Vue build ==> 打包方式，回车即可；
- Install vue-router ==> 是否要安装 vue-router，项目中肯定要使用到 所以Y 回车；
- Use ESLint to lint your code ==> 是否需要 js 语法检测 目前我们不需要 所以 n 回车；
- Set up unit tests ==> 是否安装 单元测试工具 目前我们不需要 所以 n 回车；
- Setup e2e tests with Nightwatch ==> 是否需要 端到端测试工具 目前我们不需要 所以 n 回车；

```
npm i # 安装依赖
npm run dev # 启动项目
```

