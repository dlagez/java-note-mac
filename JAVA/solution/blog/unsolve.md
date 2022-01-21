1. index页面里面，recommendBlogs逻辑有错误
2. index页面里面，tag，type显示总数的时候将blog也返回回去了。如果博客比较多，会导致网络压力激增。所以需要从数据库层面解决。