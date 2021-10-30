WIN

下载安装包：[Empowering App Development for Developers | Docker](https://www.docker.com/)

设置：win环境：[Manual installation steps for older versions of WSL | Microsoft Docs](https://docs.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package)



# install 

## sqlserver: 

- articles:https://www.sqlservercentral.com/articles/docker-desktop-on-windows-10-for-sql-server-step-by-step
- docker link: https://hub.docker.com/_/microsoft-mssql-server

```
// pull
docker pull mcr.microsoft.com/mssql/server:2019-latest

// run
docker run --name sqlserver -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=Dlagez3133.." -e "MSSQL_PID=Enterprise" -p 1433:1433 -d mcr.microsoft.com/mssql/server

// test
docker exec -it sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa
```

