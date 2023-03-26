---
title: 嵌入式学习之（六）| QT 使用说明
date: 2020-09-24
author: Kevin 吴嘉文
keywords: 
language: cn
category:
- 知识笔记
tag:
- 计算机软件
- 嵌入式学习
mathjax: true
toc: true
comments: 

---

# QT

> 嵌入式自学开始啦~ 为了自己的贾维斯
>
> 多看帮助文档！！帮助》索引查看函数与类详情，也可以通过 qt 助手查询，即 qt assistant
>
> application》qt widgets application 
>
> 笔记总结 课程链接：[千峰嵌入式教程](https://www.bilibili.com/video/BV1FA411v7YW?p=530&spm_id_from=pageDriver)

<!-- more-->

# QWidget

base class

+ qwidget 窗口应用
+ qmainwindow 带工具栏的窗口
+ qdialog 对画框窗口

## qt 框架

```cpp

#include <QApplication> //qt 框架头文件

int main(int argc, char *argv[])
{
    QApplication a(argc, argv); //框架初始化
    //你的设计
    Qwidget w;
    w.show()
    return a.exec(); 
}
```

+ a.exec() 让程序不终结，类似于 while 循环 程序进入消息循环，等待对用户输入进行响应。这里 main()把控制权转交给 Qt，Qt 完成事件处理工作，当应用程序退出的时候 exec()的值就会返回。在 exec()中 Qt 接受并处理用户和系统的事件并且把它们传递给适当的窗口部件。
+ 编译后生成对应文件夹

## 手动创建

其他项目，空项目

+ 添加 c/c++ source file

+ debug 输出 

```cpp
#include <QDebug>

qDebug()<<"hello";
```

### 。pro 文件

```
30 l greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
31 这条语句的含义是，如果 QT_MAJOR_VERSION 大于 4（也就是当前使用的 Qt5 及更高版本）需
要增加 widgets 模块。如果项目仅需支持 Qt5，
32 也可以直接添加“QT += widgets”一句。
33 不过为了保持代码兼容，最好还是按照 QtCreator 生成的语句编写。
34 l #配置信息
35 CONFIG 用来告诉 qmake 关于应用程序的配置信息。
36 CONFIG += c++11 //使用 c++11 的特性
37 在这里使用“+=”，是因为我们添加我们的配置选项到任何一个已经存在中。这样做比使
用“=”那样替换已经指定的所有选项更安全。
```

## 设置窗口属性

所以一般窗口的属性和添加控件以及对控件的操作都会在类的构造函数中（widget.cpp）书写

```cpp
#include "widget.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
{
    //代码
    this->setWindowTitle("hello");
    this->setFixedSize(600,400);

}

Widget::~Widget()
{

}
```

其他窗口属性在 Qwidget 类中查看

+ 按钮

```cpp
#include "widget.h"
#include <QPushButton>
Widget::Widget(QWidget *parent)
    : QWidget(parent)
{
    //代码
    this->setWindowTitle("hello");
    this->setFixedSize(600,400);

    button = new QPushButton("登入",this);//构造函数时指定父对象和设置文本
    //如果不给按钮指定父对象 那么按钮和窗口是单独显示 如果给按钮指定了父对象,只要父对象显示了,按钮也会显示
    button->show();
    //指定按钮的父类是窗口
     // button‐>setParent(this);//指定按钮的父亲是窗口
    button->resize(300,200);//设置按钮的大小
    button->move(100,100);//设置按钮在窗口中的位置
    // button‐>setText("登入");//设置按钮的文本
}

Widget::~Widget()
{
}
```

widget.h 中声明

```cpp
#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QPushButton>

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parenc++t = nullptr);
    ~Widget();
    QPushButton *button;
};
#endif // WIDGET_H
```



### 对象树

QObject 中会自动添加创建的子对象到 children 列表中，当父对象析构时，这个列表里面的所有对象都会被析构。

```
{QPushButton quit("Quit");
QWidget window;
quit.setParent(&window);}
```

但析构顺序是未定义的，上例中 quit 会被析构两次

## 窗口坐标系

左上角为 0，0，X,Y 分别向右，向下增大

## 信号与槽

当事件发生后，会有一个信号被广播出来。如果右对象对这个信号感兴趣，他就会将想处理的信号和自己的一个函数（称为槽 slot）绑定来处理这个信号。也就是说，当信号发出时，被连接的槽函数会自动被回调。

```cpp
connect(sender, signal, receiver, slot);

class Widget : public QWidget
{
    Q_OBJECT
public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
    QPushButton *button;
public slots: //下下方声明自定义的槽函数
    void print();
};

Widget::Widget(QWidget *parent)
    : QWidget(parent)
{   this->setWindowTitle("hello");
    this->setFixedSize(600,400);
    button = new QPushButton("登入",this);	connect(button,&QPushButton::pressed,this,&Widget::print);
}
void Widget::print(){
    qDebug()<<"点击了";
}
```

### 查找系统自带的信号和槽

在帮助文档中输入 QPushButton，首先我们可以在 Contents 中寻找关键字 signals，如果没找到，应该想到也许这个信号是从父类继承下来的，因此去他的父类查找就行。

### 自定义信号

例中先通过 clicked 接受点击信号，然后启动设计好的 emit_mysignal 槽函数，在槽函数种设置发射信号。 

```cpp
void Sonwidget::emit_mysignal()
{emit show_hide_signal(10);  }
```

信号只需要声明，不需要实现。

槽函数需要声明与实现。会受到 public，private 等影响。

信号和槽的参数类型应该一致。

信号槽可以使用 disconnect 断开连接

任何成员函数、static 函数、全局函数和 Lambda 表达式都可以作为槽函数

# Qmainwindow

```cpp
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
QMenuBar *menubar1 = this->menuBar();//取出菜单栏
QMenu *filemenu = menubar1->addMenu("文件"); //向菜单栏上去添加 菜单项
//QMenu *fileedit = menubar1->addMenu("编辑");

//向菜单添加菜单项
QAction *openaction = filemenu->addAction("文件");  //可以通过槽函数来实现打开文件等操作
filemenu->addSeparator();//添加分割线
QAction *saveaction = filemenu->addAction("保存");

QPixmap pix;
//选择图片
pix.load(":/image/test.png");
//给菜单项设置图片
openaction->setIcon(QIcon(pix));

//获取工具栏 工具栏中的工具其实是菜单栏中的快捷方式
QToolBar *toolbar = this->addToolBar("");
//向工具栏中添加工具(添加菜单项)
toolbar->addAction(openaction);
toolbar->addAction(saveaction);

//取出状态栏
QStatusBar *status = this->statusBar();
status->addWidget(new QLabel("状态"));//向状态添加控件

////创建铆接部件
//QDockWidget *dockwidget = new QDockWidget("这是一个铆接部件", this);
//this->addDockWidget(Qt::TopDockWidgetArea, dockwidget);//将浮动窗口添加到 mainwindow
QTextEdit *edit = new QTextEdit("文本编辑器", this);
this->setCentralWidget(edit);
}
```



## 对话框

```cpp

Widget::Widget(QWidget *parent)
    : QWidget(parent)
{
    this->resize(600,400);
    button = new QPushButton("选择文件",this);
    button->resize(100,100);
    button->move(250,100);
    connect(button,&QPushButton::clicked,this,[](){
    // QString str = QFileDialog::getOpenFileName();
    // qDebug()<<str;
    QStringList str = QFileDialog::getOpenFileNames();
    qDebug()<<str;
});
    button1 = new QPushButton("模态对话框",this); //模态对话框，必须关了才能操作主窗口
    button1->resize(100,100);
    button1->move(150,100);
    connect(button1,&QPushButton::clicked,this,[](){
        QDialog dialog;
        dialog.setWindowTitle(tr("Hello, dialog!"));  //设置标题
        dialog.exec();//阻塞对话框

});
    button2 = new QPushButton("非模态对话框",this); //非模态对话框，不关闭也能操作主窗口
    button2->resize(100,100);
    button2->move(250,100);
    connect(button2,&QPushButton::clicked,this,[](){
        QDialog *dialog = new QDialog;
        dialog->setAttribute(Qt::WA_DeleteOnClose); //释放内存
        dialog->setWindowTitle(tr("Hello, dialog!"));  //设置标题
        dialog->show();});
}
```

QColorDialog： 选择颜色；
QFileDialog： 选择文件或者目录；
QFontDialog： 选择字体；
QInputDialog： 允许用户输入一个值，并将其值返回；
QMessageBox： 模态对话框，用于显示信息、询问问题等；
QPageSetupDialog： 为打印机提供纸张相关的选项；
QPrintDialog： 打印机配置；
QPrintPreviewDialog：打印预览；
QProgressDialog： 显示操作过程。

消息对话框

如果我们希望制作一个询问是否保存的对话框，我们可以使用如下的代码：

```cpp
QMessageBox msgBox;
msgBox.setText(tr("The document has been modified."));
msgBox.setInformativeText(tr("Do you want to save your changes?"));
msgBox.setDetailedText(tr("Differences here..."));
msgBox.setStandardButtons(QMessageBox::Save
| QMessageBox::Discard
| QMessageBox::Cancel);
msgBox.setDefaultButton(QMessageBox::Save);
int ret = msgBox.exec();
switch (ret)
{
case QMessageBox::Save:
qDebug() << "Save document!";
break;
case QMessageBox::Discard:
qDebug() << "Discard changes!";
break;
case QMessageBox::Cancel:
qDebug() << "Close document!";
break;
}
```

### 布局管理器

在创建项目的时候勾选使用 widget.ui，然后打开项目在 widget.ui 中编辑窗口

.ui 文件会被系统解析成.h 文件

.h 中会有 ui_widget 的类，namespace 是 ui

+ 工具中可以设置按钮布局，按钮大小等

spacer 弹簧可以用来控制界面的优化

图片的相对路径的上一级目录是 makefile（）如果在 qt 运行的话。建议将图片编辑成资源 sources 来添加到程序中，而不是使用相对路径。



+  **在按钮上右击，选择转到槽，那么就会跳转到相对应的槽函数界面。只需要编辑槽函数的反应内容就行，connect 部分 qt 会自动设置，但是槽函数的名字不能乱改** 

## 常用控件

### label

在界面中显示文本图片和图画或者超链接（使用 HTML 格式）

头文件：

```
#include <Qlabel>
//类中添加 Qlabel *label;
```

```cpp
#include "widget.h"

Widget::Widget(QWidget *parent)
: QWidget(parent)
{
this‐>resize(600,400);
label = new QLabel(this);
label‐>resize(200,200);
label‐>move(100,100);
// label‐>setText("我是一个标签");
label ‐>setText("<a href=\"https://www.baidu.com\">百度一下</a>");
label->setOpenExternalLinks(true);//设置点击链接自动打开
    
//设置标签  
label_pic = new QLabel(this);
label_pic‐>resize(100,100);
label_pic‐>move(200,200);
label_pic‐>setPixmap(QPixmap("../Image/face.png"));
label_pic‐>setScaledContents(true);//设置自适应大小
}

Widget::~Widget()
{

}
```

### LineEdit

行文本输入框

```c
#include "widget.h"
#include "ui_widget.h"
#include <QDebug>
Widget::Widget(QWidget *parent) :
QWidget(parent),
ui(new Ui::Widget)
{
    ui‐>setupUi(this);
    ui‐>lineEdit‐>setText("hello");//设置行编辑内容
}


//密码模式
{
    ui‐>setupUi(this);
    ui‐>lineEdit‐>setEchoMode(QLineEdit::Password);//设置密码模式
    10 ui‐>lineEdit‐>setTextMargins(100,0,0,0);//设置间距
    11 ui‐>lineEdit‐>setText("hello");//设置行编辑内容
}
//密码模式


Widget::~Widget()
{
    delete ui;

    void Widget::on_lineEdit_returnPressed()
    {
        qDebug()<<ui‐>lineEdit‐>text();//获取行编辑内容
    }
```

### 自定义控件

### 定时器

```cpp
Widget::Widget(QWidget *parent) :
QWidget(parent),
ui(new Ui::Widget)
{ui‐>setupUi(this);
t = new QTimer;
//设置定时器的超时事件 如果超时 会发出一个超时信号
connect(t,&QTimer::timeout,this,[](){
qDebug()<<"timeout";
});
 //定时器必须启动
}

Widget::~Widget()
{delete ui;}

void Widget::on_pushButton_clicked()
{
t‐>start(1000);//启动定时器 1ms 超时一次}

void Widget::on_pushButton_2_clicked()
{
t‐>stop();//停止定时器
}
```

### 鼠标

.cpp

```c
#include "widget.h"
#include "ui_widget.h"
#include <QMouseEvent>
#include <QDebug>
Widget::Widget(QWidget *parent) :
QWidget(parent),
ui(new Ui::Widget)
{ ui‐>setupUi(this);
}

Widget::~Widget()
{
    delete ui;
}

void Widget::mousePressEvent(QMouseEvent *event)
{
    qDebug()<<"鼠标点击"<<event‐>x()<<event‐>y();
    if(event‐>button() ==Qt::LeftButton )
    {
        qDebug()<<"点击了左键";

    }
    else if(event‐>button() ==Qt::RightButton )
    {
        qDebug()<<"点击了右键";

    }

}

void Widget:: mouseMoveEvent(QMouseEvent *event)
{

    qDebug()<<"鼠标点击"<<event‐>x()<<event‐>y();

}
```

.h

```c
#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
namespace Ui {
class Widget;
}
class Widget : public QWidget
{
    Q_OBJECT

        public:
    explicit Widget(QWidget *parent = 0);
    ~Widget();
    protected:
    void mousePressEvent(QMouseEvent *event);c
        void mouseMoveEvent(QMouseEvent *event);

    private:
Ui::Widget *ui;
};

#endif // WIDGET_H
```



未完待续...