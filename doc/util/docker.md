WIN

下载安装包：[Empowering App Development for Developers | Docker](https://www.docker.com/)

设置：win环境：[Manual installation steps for older versions of WSL | Microsoft Docs](https://docs.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package)



## install sqlserver: 

- articles:https://www.sqlservercentral.com/articles/docker-desktop-on-windows-10-for-sql-server-step-by-step
- docker link: https://hub.docker.com/_/microsoft-mssql-server

```
// pull
docker pull mcr.microsoft.com/mssql/server:2019-latest

// run
docker run --name sqlserver -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=Dlagez3133.." -e "MSSQL_PID=Enterprise" -p 1433:1433 -d mcr.microsoft.com/mssql/server

// test
docker exec -it sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa

// exec
exec -it sqlserver "bash"

// can't into 
docker exec -it sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P Dlagez3133..

登录名是：sa 密码password
```



## install mysql

```
docker pull mysql:8.0.27
docker run -p 3306:3306 --name mysql -e MYSQL_ROOT_PASSWORD=password -d mysql:8.0.27 # run image
docker exec -it mysql bash # inside a Docker container
docker logs mysql # get log
mysql -u root -p # inside m
```



## install postgresql 

https://hub.docker.com/_/postgres

```cmd
docker pull postgres:9.4.26
#  -e TZ=PRC 设置时区为中国
docker run -p 15432:5432 --name postgres -e POSTGRES_PASSWORD=password -e TZ=PRC -d postgres:9.4.26

# 默认用户名是postgres 密码password
```



## install elasticsearch

