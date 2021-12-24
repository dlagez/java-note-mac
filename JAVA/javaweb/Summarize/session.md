.SESSION
session_id() //返回当前会话ID。 如果当前没有会话，则返回空字符串（”“）。
:当程序需要为某个客户端的请求创建一个session时，服务器首先检查这个客户端的请求里是否已包含了一个session标识
（称为session id），如果已包含则说明以前已经为此客户端创建过session，服务器就按照session id把这个session检索出来
使用（检索不到，会新建一个），如果客户端请求不包含session id，则为此客户端创建一个session并且生成一个与此session相
关联的session id，session id的值应该是一个既不会重复，又不容易被找到规律以仿造的字符串，这个session id将被在本次响应
中返回给客户端保存。保存这个session id的方式可以采用cookie，这样在交互过程中浏览器可以自动的按照规则把这个标识发送给
服务器。一般这个cookie的名字都是类似于SEEESIONID。但cookie可以被人为的禁止，则必须有其他机制以便在cookie被禁止时
仍然能够把session id传递回服务器。
session_start();
session_unset() ; //删除SESSION，但不删除SESSION的存储文件,就是会所还可以调用
session_destroy(); //清除SESSION，删除文件

总结：用户请求中带有session标识，说明服务器已经为该用户创建了session了。