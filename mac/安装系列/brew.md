查看软件安装位置

```
brew list oclint
```



安装elasticsearch

```
brew tap elastic/tap

brew install elastic/tap/elasticsearch-full
```

1. 命令行运行：elasticsearch
2. 浏览器访问：http://localhost:9200/

```json
{
  "name" : "roczhang-mac.local",
  "cluster_name" : "elasticsearch_roczhang",
  "cluster_uuid" : "vDRrPhkxRQSG6HebN3fyKQ",
  "version" : {
    "number" : "7.14.1",
    "build_flavor" : "default",
    "build_type" : "tar",
    "build_hash" : "66b55ebfa59c92c15db3f69a335d500018b3331e",
    "build_date" : "2021-08-26T09:01:05.390870785Z",
    "build_snapshot" : false,
    "lucene_version" : "8.9.0",
    "minimum_wire_compatibility_version" : "6.8.0",
    "minimum_index_compatibility_version" : "6.0.0-beta1"
  },
  "tagline" : "You Know, for Search"
}
```

安装中文分词插件：

```
// 查看软件位置
brew list elasticsearch-full
// 进入软件bin目录
cd /opt/homebrew/Cellar/elasticsearch-full/7.14.1/bin

elasticsearch-plugin install https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v7.14.1/elasticsearch-analysis-ik-7.14.1.zip
```

安装它的软件

```undefined
brew install elastic/tap/kibana-full
```

1. 命令行运行：kibana
2. 浏览器访问：http://localhost:5601/

