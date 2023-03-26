---
title: Mongodb 基本
date: 2021-12-08
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- 数据库
mathjax: true
toc: true
comments: 笔记
---

> 安装、基本语法、资源整合

<!--more-->

## 安装

[官网](https://www.mongodb.com/try/download/community) 下载 tgz 文件

```
# 创建 mongodb 目录
mkdir -p /usr/local/mongodb
# 解压 mongodb 至指定目录
tar -zxvf /usr/local/src/mongodb-linux-x86_64-rhel70-4.4.1.tgz -C /usr/local/
# 重命名解压目录为 mongodb
mv /usr/local/mongodb-linux-x86_64-rhel70-4.4.1/ /usr/local/mongodb
```

```
# 创建存放数据的目录
mkdir -p /usr/local/mongodb/data/db
# 创建存放日志的目录
mkdir -p /usr/local/mongodb/logs
# 创建日志记录文件
touch /usr/local/mongodb/logs/mongodb.log
```

### docker 安装

```shell
docker run -p 27017:27017 --restart=unless-stopped  --name mongo -v $PWD:/data/db -d mongo:4.4.14-rc0-focal --auth
```

更多参考 mongo docker 官方，或者 [菜鸟教程](https://www.runoob.com/docker/docker-install-mongodb.html)

## 启动

```shell
# 切换至指定目录
cd /usr/local/mongodb/
# 前台启动
bin/mongod --dbpath /usr/local/mongodb/data/db/ --logpath /usr/local/mongodb/logs/mongodb.log --logappend --port 27017 --bind_ip 0.0.0.0
```

- `--dbpath`：指定数据文件存放目录
- `--logpath`：指定日志文件，注意是指定文件不是目录
- `--logappend`：使用追加的方式记录日志
- `--port`：指定端口，默认为 27017
- `--bind_ip`：绑定服务 IP，若绑定 127.0.0.1，则只能本机访问，默认为本机地址

后台启动添加 `--fork` 即可

在 `bin` 目录下增加一个 `mongodb.conf` 配置文件。

```
# 数据文件存放目录
dbpath = /usr/local/mongodb/data/db
# 日志文件存放目录
logpath = /usr/local/mongodb/logs/mongodb.log
# 以追加的方式记录日志
logappend = true
# 端口默认为 27017
port = 27017
# 对访问 IP 地址不做限制，默认为本机地址
bind_ip = 0.0.0.0
# 以守护进程的方式启用，即在后台运行
fork = true
auth = true
```

`vim /etc/profile`

```
# 添加环境变量
export MONGODB_HOME=/usr/local/mongodb
export PATH=$PATH:$MONGODB_HOME/bin
```

通过配置文件启动

```
mongod -f /usr/local/mongodb/bin/mongodb.conf
```

退出后台进程：

```
mongod -f /usr/local/mongodb/bin/mongodb.conf --shutdown
```

## 用户与权限管理

| 权限                 | 说明                                                         |
| -------------------- | ------------------------------------------------------------ |
| read                 | 允许用户读取指定数据库。                                     |
| readWrite            | 允许用户读写指定数据库。                                     |
| dbAdmin              | 允许用户在指定数据库中执行管理函数，如索引创建、删除，查看统计或访问 system.profile。 |
| userAdmin            | 允许用户向 system.users 集合写入，可以在指定数据库里创建、删除和管理用户。 |
| clusterAdmin         | 必须在 admin 数据库中定义，赋予用户所有分片和复制集相关函数的管理权限。 |
| readAnyDatabase      | 必须在 admin 数据库中定义，赋予用户所有数据库的读权限。      |
| readWriteAnyDatabase | 必须在 admin 数据库中定义，赋予用户所有数据库的读写权限。    |
| userAdminAnyDatabase | 必须在 admin 数据库中定义，赋予用户所有数据库的 userAdmin 权限。 |
| dbAdminAnyDatabase   | 必须在 admin 数据库中定义，赋予用户所有数据库的 dbAdmin 权限。 |
| root                 | 必须在 admin 数据库中定义，超级账号，超级权限。              |

### 创建管理权限

`use admin` 切换至 admin 数据库进行登；`show user` 查看用户。创建用户：

```shell
db.createUser({ 
    user: "kevin",
    pwd: "777777",
    roles: [
        { role: "userAdminAnyDatabase", db: "admin" } 
    ]
});
```

链接 `mongo` 服务后，登录`use admin`，`db.auth("kevin","777777")` 进行验证，返回 1 为验证成功。

### 创建普通用户

普通用户针对非 admin 数据库。首先需要使用 `use test` 创建数据库。

```shell
db.createUser({ 
    user: "user1",
    pwd: "111111",
    roles: [
        { role: "readWrite", db: "test" } 
    ]
});
```

同样登录使用 `use test` ，`db.auth("user1","111111")`

### 更新用户

需要当前用户具有 `userAdmin` 或 `userAdminAnyDatabse` 或 `root` 角色。

 **更新权限** 

```
db.updateUser("用户名", {"roles":[{"role":"角色名称",db:"数据库"},{"更新项 2":"更新内容"}]})
```

 **更新密码**  - 需要切换到该用户所在的数据库。

```
db.updateUser("用户名", {"pwd":"新密码"})
db.changeUserPassword("用户名", "新密码")
```

 **删除用户**  - 需要切换到该用户所在的数据库

```
db.dropUser("user name")
```

## 备份

[参考](https://baijiahao.baidu.com/s?id=1715684157607166534&wfr=spider&for=pc)

以下进讨论由用户密码情况，在服务器 A 上备份 mongodb

```shell
./mongodump --db=testdb --authenticationDatabase=admin --username=admin --password=123456 --host=127.0.0.1:27017 --out=./backups/mongofile
```

其中： db 为需要备份的数据库名， authenticationDatabase 为认证用的数据库名，username 为认证的用户名， password 为认证用的数据库用户密码， host 为连接的 IP 及端口， out 为备份后的存储目录

在 B 机上导入，首先创建对应的 mongo 用户

```shell
mongo admin
db.createUser({ user:'admin',pwd:'123456',roles:[ { role:'userAdminAnyDatabase', db: 'admin'},"readWriteAnyDatabase"]});
```

而后将需要导入的文件 `./backups/mongofile` 发送到 B 服务器上，而后进行回复：

```shell
mongorestore --db=testdb --authenticationDatabase=admin --username=admin --password=123456 --host=127.0.0.1:27017 ./backups/mongo/testdb
```

## GUI

[Robo 3t]([https://robomongo.org/download](https://links.jianshu.com/go?to=https%3A%2F%2Frobomongo.org%2Fdownload)) 下载解压后可通过 `bin/robo3t` 文件直接运行。

连接到 mongo 服务器时候需配置`Authentication` 4.0 以上版本选择 SHA-S56

## 语法

#### 数据库

切换/创建 数据库 `use test` ，显示数据库 `show dbs`，不同用户显示不一样；删除数据库需要切换到当前库，然后 `db.dropDatabase()`

```
# 执行以下函数(2 选 1)即可关闭服务
db.shutdownServer()
db.runCommand(“shutdown”)
```

#### 集合 collection

使用 `db.createCollection(name, option)` 直接创建； 使用 `show tables` 查看集合；使用 `db.collection_name.drop()` 删除。

#### 文档

 **插入** 

使用 `db.new_c.insert()` 可直接创建 `new_c` 集合并且插入一个文档。如：

```
user1 = {
"_id":102,
"students":[{
"id":1,"name":"zhangsan"
},{"id":2,"name":3}]
}

db.new_c.insert(user1)
```

`_id` 为文档唯一值，会更新原本已有的 `_id` 数据。批量插入的时候采用 `db.new_c.insert([doc1,doc2...])`

  **更新** 

```
db.collection_name.update(query, update, option)
```

-  **query**  : update 的查询条件，类似 sql update 查询内 where 后面的。
-  **update**  : update 的对象和一些更新的操作符（如`$`,`$inc`...）等，也可以理解为 sql update 查询内 set 后面的
-  **upsert**  : 可选，这个参数的意思是，如果不存在 update 的记录，是否插入 objNew,true 为插入，默认是 false，不插入。
-  **multi**  : 可选，mongodb 默认是 false,只更新找到的第一条记录，如果这个参数为 true,就把按条件查出来多条记录全部更新。
-  **writeConcern**  :可选，抛出异常的级别。

```
db.new_c.update({'name':'zhangsan'},{$set:{'name':'MongoDB'}},false,true)
```

如上例，`$set` 更新会覆盖原有数据，而非只更新 `name`

 **删除** 

`df.new_c.remove({"name":"zhangsang"}, {justOne: true})`

 **查询** 

```
db.user.find("_id":{"$lte":4,"gle":3})
db.user.distinct("name") # 根据 name 去重
```

正则定义在 `/ /` 内。

[更多查询案例与语法](https://www.runoob.com/mongodb/mongodb-query.html)

 **排序** 

```
db.COLLECTION_NAME.find().sort({KEY:1})
```

 **聚合** 

```
db.mycol.aggregate([{$group : {_id : "$by_user", num_tutorial : {$sum : 1}}}])
{
   "result" : [
      {
         "_id" : "runoob.com",
         "num_tutorial" : 2
      },
      {
         "_id" : "Neo4j",
         "num_tutorial" : 1
      }
   ],
   "ok" : 1
}
```

可以接 `$sum`,`$max`,`$first`等。

 **索引** 

索引通常能够极大的提高查询的效率。`db.collection.createIndex(keys, options)`

可以使用部分字段创建索引。

```
db.col.createIndex({"title":1,"description":-1})
```

1、查看集合索引

```
db.col.getIndexes()
```

2、查看集合索引大小

```
db.col.totalIndexSize()
```

3、删除集合所有索引

```
db.col.dropIndexes()
```

4、删除集合指定索引

```
db.col.dropIndex("索引名称")
```

## PyMongo

[官方文档](https://pymongo.readthedocs.io/en/stable/)

```python
myclient = pymongo.MongoClient(username=user,                password=password,                 authMechanism='SCRAM-SHA-256')
dblist = myclient.list_database_names() 
mydb = myclient["db1"]  # 创建/链接 db1 的数据库
collist = mydb. list_collection_names()
mycol = mydb["sites"]  # 创建集合 collection
```

 **文档处理** 

插入

```python
x = mycol.insert_one(mydict)  # x.inserted_id
mycol.insert_many(mylist)  # 插入多个文档
```

查找

```python
myquery = { "name": { "$regex": "^R" } }
for x in mycol.find(myquery):  
  print(x)
```

