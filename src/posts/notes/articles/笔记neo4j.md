---
title: 图数据库 Neo4j 基础与案例
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

> Neo4j 安装搭建、基础 Cypher 语法。大部分内容参考与 Neo4j [官网](https://neo4j.com/ )。 

<!--more-->

Neo4j 采用属性图模型（每个节点都有唯一标识符，节点与关系下储存有对应的属性与属性值键值对。节点之间使用关系链接，类似于 ER 图。）

## 安装搭建

[neo4j 下载中心](https://neo4j.com/download-center/#community) 选择需要的版本，Neo4j 4.4，需要 jdk11。

解压后安装服务 `<NEO4J_HOME>/bin/neo4j install-service`，在后台启动服务：

`<NEO4J_HOME>/bin/neo4j start`

浏览器访问 `localhost:7474`，首次访问会弹出登录界面，默认用户名与密码为 `neo4j`。

网页由多个命令框组成，顶部命令框中 `neo4j$` 表示目前正在使用 `neo4j` 数据库，命令都以 `:` 开头，一些常用与命令框的快捷键有： `ctrl+ Enter` 执行代码行，`ctrl + up-arrow/down-arrow`  查看上/下一条执行的代码。`F1` 查看更多快捷键。

可以点击命令框最右侧的星来收藏框中的命令。

可以将查询结果与代码等保存成  Cypher 文件，并储存与项目对应的仓库中，其他项目成员也可以访问到他们。

## Cypher

常用命令有 `CREATE`, `MATCH`, `RETURN`, `WHERE`, `DELETE`, `REMOVE`, `ORDER BY`, `SET` 等 [查看官网详细文档](https://neo4j.com/docs/cypher-manual/current/clauses/)。

#### CREATE 与 MATCH

通常使用 `()` 表示节点，`[]` 表示关系。

创建一个节点 `ee`，节点 `name` 为 "张三" ，标签为 `Person`。

```
CREATE (ee:Person {name:'张三',属性 2:'属性 2 值'})
```

使用 `MATCH` 查找对应的节点，

```
MATCH (ee:Person) WHERE ee.name = '张三' RETURN ee;
```

其中 `ee` 为临时变量，用来代表节点搜查结果。随后建立张三住在北京的关系：

```
match (c:Person) where c.name="张三"
CREATE(a:location {name:"北京"}),
(b:location {name:"上海"}),
(n:Person {name:"李四"}) ,
(n)-[:住在 {since:2001}]->(a),
(n)-[:住在 {since:2011}]->(b),
(c)-[:住在 {since:1999}]->(a)
```

`[]`表示关系，此处关系也可以写成`(b)<-[:住在 {since:2011}]-(n)`；查找和张三住一个地方的人：

```
MATCH (a:Person)-[]->(location)<-[]-(c:Person) return a,c,location
```

同时 match 支持正则表达式（语法如：`where n.name=~".*三"`） 。也可以根据关系搜索：

```
match p=()-[c:relation]-() return p
```

查询所有有对外关系的节点：`MATCH (a)-->() RETURN a`；查询所有有关系的节点`MATCH (a)--() RETURN a`

### DELETE

删除节点前需要修改节点属性：

```
MATCH (a:Person {name:'张三'}) SET a.test='test'
MATCH (a:Person {name:'张三'}) REMOVE a.test
```

删除所有节点与关系

```
MATCH (n) DETACH DELETE n
```

#### LOAD CSV

加载 csv 文件中的数据，csv 文件需要是 UTF-8 编码。导入后使用 `line[i]` 来索引需要的列。

```
LOAD CSV FROM 'file:///sample.csv' as line
create (:person {name:line[0]})
或
LOAD CSV WITH HEADERS FROM 'file:///sample.csv' as line
create (:person {name:line.name})
```

如上例，CSV 文件应放置于 `<NEO4J_HOME>/data/csv/sample.csv`。

通过导入存有 RDF 三元组的文件来导入节点关系：

```
USING PERIODIC COMMIT LOAD CSV FROM 'file:///rdf.csv' as line
create (:myrelation {subject:line[0],relation:line[1],object:line[2]})
```

`USING PERIODIC COMMIT ` 使 neo4j 在加载一定数据后 commit 一次，常用于载入大文件。在导入数据后，对两个数据集的内容进行匹配，建立节点关系：

```
match (n:person),(m:relation),(s:person) where n.name=m.subject and m.object=s.name
create (n)-[r:关系 {relation:m.relation}]->(s)
```

#### INDEX

创建索引可加快检索速度

```
CREATE INDEX ON :label(property)
CREATE INDEX ON :person(name)
DROP INDEX ON :person(name)
```

#### UNIQUE

确保数据库中 `person.name`是唯一的：

```
CREATE CONSTRAINT ON (n:Person) ASSERT (n.name) IS UNIQUE
DROP CONSTRAINT ON (n:Person) ASSERT (n.name) IS UNIQUE
```

## 数据回复与备份

备份需要关闭 neo4j 服务。

```
neo4j stop
neo4j-admin dump --database=neo4j --to=backup/1.dump
```

还原时仍需要关闭 neo4j 服务：

```
bin/neo4j-admin load --from=backup/1.dump --database=new.db --force
```

## 插件安装

 APOC（Awesome Procedures On Cypher）  APOC 可以提供 文本索引、图算法、空间函数 、数据集成、图形重构、触发器等功能 [APOC 详细](https://neo4j.com/labs/apoc/)。

 **APOC 安装：** 从 [apoc release](https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases) 下载对应版本 xx-all.jar 包，放置于 neo4j plugins 文件夹下。

重启 neo4j 后，输入 `return apoc.version()` 验证是否安装成功。

要实现与 MySQL 数据库链接的话，需要安装 MySQL connector 插件。从 [页面](https://downloads.mysql.com/archives/c-j/) 下载 Platform Independent 对应文件，解压出其中的 `mysql-connector-java-8.0.26.jar` 放置于 neo4j plugins 文件夹。

`apoch.load.jdbc` 加载 MySQL 数据库时，使用`row.col_name` 选择。

```
call apoc.load.jdbc('jdbc:mysql://{IP}:{PORT}/{DBNAME}? user={USERNAME}&password={PASSWORD} ","{TABLENAME}") yield row
create (b:Black{number:row.black_id, type:row.type})
```

## py2neo

```python
from py2neo import Graph
graph = Graph("http://localhost:7687", auth=("neo4j", "password"))
```

创建带有属性的节点

```python
from py2neo import Graph,Node
def create_diseases_nodes(self, disease_infos):
    for disease_dict in disease_infos:
        node = Node(label="Disease", name=disease_dict['name'], desc=disease_dict['desc'],
                    prevent=disease_dict['prevent'] ,cause=disease_dict['cause'],
 easy_get=disease_dict['easy_get'],cure_lasttime=disease_dict['cure_lasttime'],
                    cure_department=disease_dict['cure_department'], cure_way=disease_dict['cure_way'] , cured_prob=disease_dict['cured_prob'])
        self.g.create(node)
```

 **创建关系**  - 使用 `Graph.run(query)`

```python
def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
    # 去重处理
    set_edges = []
    for edge in edges:
        set_edges.append('###'.join(edge))
    all = len(set(set_edges))
    for edge in set(set_edges):
        edge = edge.split('###')
        p = edge[0]
        q = edge[1]
        query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
            start_node, end_node, p, q, rel_type, rel_name)
        try:
            self.g.run(query)
        except Exception as e:
            print(e)
    return

# self.create_relationship('Disease', 'Food', rels_doeat, 'do_eat', '宜吃')
```

查询的方式类似：

```python
query = "MATCH (m:Disease) where m.name = '苯中毒' return m.name, m.cause"
ress = self.g.run(query).data()
```

## 其他

[APOC 函数](https://neo4j.com/labs/apoc/4.1/overview/)

[Debian/Ubuntu 下安装 Neo4j server](https://neo4j.com/docs/operations-manual/current/installation/linux/debian/#debian-ubuntu-prerequisites)

#### ubuntu18.04+上安装 jdk11

`apt-get install default-jdk` 安装好后，`jave -version` 检查。

处理多个 java 版本问题：`update-java-alternatives --list` 显示已安装的版本，

`sudo update-java-alternatives --jre --set <java11name>` 指定 java 版本

#### docker 运行：

```
docker pull neo4j:community
docker run \
    -p 7474:7474 \
    -p 7687:7687 \
    -p 7473:7473 \
    -v $HOME/neo4j/data:/data \
    neo4j:community
```

最短路径查询：(`*..10` 表示限定 10 跳内)

```
match (p1:Person{name:"Joel Silver"}),(p2:Person{name:"Emil Eifrem"}),p=shortestpath((p1)-[*..10]-(p2)) return p1, p2, p
```

