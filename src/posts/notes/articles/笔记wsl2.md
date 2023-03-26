---
title: Linux|Windows WSL2 下的配置
date: 2022-03-22
author: Kevin 吴嘉文
keywords: 
language: cn
category:
- Hobbies 业余爱好
tag:
- Linux
mathjax: true
toc: true
comments: 博客总结
---

由于 Linux 桌面系统下有太多的不方便，如会议 app 功能不全、部分软件不支持（特别是 word、PPT 一系列的）。因此对于需要兼顾开发和办公的人员来说，使用 WSL2 或者虚拟机可能是一个比较权衡的选择。

目前 WSL2 可以和 VM 虚拟机共存了，个人推荐使用 WSL2，其支持 docker，个人在 Ubuntu 系统的 docker 与 WSL2 docker 中进行深度学习训练，WSL2 的性能丢失仅为 8%。此外他支持 U 盘接口等，文件储存方便，并且可以通过 WSL2 内的 Linux 子系统直接调用 windows 下的 `.exe` 文件。缺点就是 WSL2 默认是服务器终端，要设置桌面版教麻烦。

这边建议统一用   **Windows Terminal**   进行终端管理，进入微软商店直接搜索下载即可。

##  WSL 下安装 ubuntu

### WSL

安装 WSL 参考 [官方文档](https://docs.microsoft.com/zh-cn/windows/wsl/install)。必须运行 Windows 10 版本 2004 及更高版本（内部版本 19041 及更高版本）或 Windows 11。

 **可以通过微软商店安装，方便快捷** 

也可以通过终端安装指定 WSL 版本，powershell 下执行 `wsl --install` 安装，之后可以安装对应的 Linux 版本：通过 `wsl --install -d <Distribution Name>` 安装。`wsl --list --online` 查看发行版本。

部分命令：

切换到子 Linux 系统中：`wsl -d Ubuntu-20.04 -u root`
查看全部安装的子系统：`wsl -l -v`
设置默认使用的系统：`wsl --setdefault <DistributionName>`
设置使用 WSL2：`wsl --set-version <distro name> 2`

### 更换 WSL 储存地址

子系统的备份与恢复：

WSL2 默认存放路径可以通过下面的，备份+恢复方式来修改。

```shell
wsl --shutdown
wsl --export Ubuntu-20.04 D:\Ubuntu-20.04.tar
# 卸载前必须备份，否则数据会全部丢失
wsl --unregister Ubuntu-20.04
# 下面导入 WSL 系统， D:\new_path\就是储存文件的地方。
wsl --import Ubuntu-20.04 D:\new_path\ D:\Ubuntu-20.04.tar
```

此外 WSL 还支持预览功能（Linux GUI），但是需要加入 Windows 预览体验计划。

### 其他 WSL 下的 ubuntu

关于终端的配置可以参考我的另一篇博客 [Ubuntu 配置与软件推荐](http://wujiawen.xyz/archives/linuxubuntu%E9%85%8D%E7%BD%AE%E4%B8%8E%E8%BD%AF%E4%BB%B6%E6%8E%A8%E8%8D%90)，此处列几个要点。

#### docker

+  **WSL2 下默认是不支持 systemctl 的** ，但可以手动使用 service 启动 docker 服务。
+ 对于深度学习 GPU 加速，可以参考[官方配置文档](https://docs.microsoft.com/zh-cn/windows/wsl/tutorials/gpu-compute) ，个人推荐使用 nvidia-docker2。WSL2 中使用 CUDA 启动较慢，但是经过个人测试，深度学习训练效率相对于标准的 Ubuntu 下 nvidia-docker，仅降低了 7%左右。
+ 你可能要重新配置以下 docker 以及 WSL2 的储存路径，否则 C 盘爆炸是一段时间内的事。

#### zsh

+ 个人很喜欢使用 oh-my-zsh，包括其中的 autojump, autosuggestion 等插件等，能够极大的提高生产效率。 **WSL2 设置 ZSH 为默认终端后，Windows Terminal 下的默认 Ubuntu 子系统快捷方式会消失，因此最保险的方式是安装非默认的 ubuntu 子系统，如 `Ubuntu-20.04` 等。** 

### WSL 下使用 windows 软件

 wsl 中可以使用 windows 上的软件，如 vscode, pycharm， typora 等。windows 下安装 vscode 后，在 wsl2 中可以使用 `code .` 自动打开工作目录，无需配置。

首先需要在 windows 中配置系统环境变量。而后在 wsl 的终端中设置环境变量，指向 windows 下软件的位置即可，如采用 zsh，则在 `.zshcr` 中配置:

```
alias ty="/home/D/apps/Typora/Typora.exe"  # 用 windows 上的 typora 打开 wsl2 中的文件。
alias nhere="explorer.exe"                 # 在 windows 上打开 wsl 目录文件夹
alias pycharm="pycharm64.exe"              # 用 windows 上的 pycharm 打开 wsl2 中的工作目录。
alias git="git.exe"                        # 用 windows 上的 git，这是比较不推荐的。
```

此外 pycharm 等也可以配置 wsl 中的 python 解释器，参考：[官方](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter-1.html) 在添加解析器时候选择 wsl 就行。

### 修改 wsl 的默认用户

通过 `wsl -l -v` 查看系统名称，如 `Ubuntu-18.04` 的话，命令行则使用 `ubuntu1804`。修改默认用户为 `kevin`：

```
ubuntu2004 config --default-user kevin
```

修改后，通过 vscode, pycharm, 文件管理器等 windows 应用访问时，默认采用 `kevin` 进行操作。若非必须的话，建议不用 `root` 作为默认用户，避免 linux 和 windows 系统文件权限不匹配的问题。

#### 目前发现问题

+ 由于 wsl 和 windows 共享磁盘，如 D，C 盘采用的磁盘格式特殊，因此无法修改共享磁盘下的文件权限。
+ 如果使用 `apt-get install git` 再 linux 系统中安装 git 的话，默认情况下是不能够在挂载路径进行 git 操作的，如 `/mnt/d/xxx`。可以使用 `windows`下安装的 `git.exe`，设置 `alias git=git.exe`；但更推荐的是在非共享磁盘下进行进行 git 操作，然后采用上节介绍的方法来修改代码和文件，这样能省去很多 windows git 的麻烦。 
+ `No route to host` 错误：wlan 网络下应该禁用 internet 共享。确保 `控制面板\网络和 Internet\网络连接` 下的 WLAN-属性-共享下的与其他人共享 internet 功能没有开启

## powershell

推荐使用最新版的 powershell，从微软商店下载安装即可，这样可以省去很多 oh-my-posh 安装时候的坑。

### oh-my-posh

安装参考 [oh-my-posh 官网](https://ohmyposh.dev/docs/installation/windows)，网上也有很多博客如 [Oh My Posh：全平台终端提示符个性化工具](https://sspai.com/post/69911)

#### 字体设计

从 [Nerd Font](https://www.nerdfonts.com/font-downloads) 下载字体并安装（解压到 C:\WINDOWS\FONTS)，而后进行配置[参考官方](https://ohmyposh.dev/docs/configuration/fonts)。

若使用 windows Terminal，则直接在设置里面配置使用字体即可。（windows terminal 》设置》配置文件》Powershell》外观》字体）个人采用了 `JetBrainsMono NF`

#### 下载安装

```powershell
winget install oh-my-posh  
# 很慢的话尝试下科学上网
```

完成后进行初始化，首先创建配置文件，`code $PROFILE`。往里面添加：`oh-my-posh init pwsh | Invoke-Expression`后保存。最后应用生效 `. $PROFILE`

#### 主题设置

皮肤个人推荐 wopian，简约好看！

```powershell
# code $PROFILE 而后在其中修改
oh-my-posh init pwsh --config ~\AppData\Local\Programs\oh-my-posh\themes\wopian.omp.json | Invoke-Expression
```

其中，使用不同安装方式得到的主题路径不同，如下：

|  **安装方式**          |  **主题路径**                                                  |
| -------------------- | ------------------------------------------------------------ |
| Windows Scoop        | `~\scoop\apps\oh-my-posh\current\themes\wopian.omp.json`     |
| Windows Choco/Winget | `~\AppData\Local\Programs\oh-my-posh\themes\wopian.omp.json` |
| macOS Brew           | `~/.poshthemes/wopian.omp.json`                              |
| GNU/Linux 命令行     | `~/.poshthemes/wopian.omp.json`                              |
| 自行下载             | Oh my Posh 和 Themes 需要填完整的自定义路径                  |

#### alias

`$PROFILE` 文件中添加：

```shell
Set-Alias -Name ty -Value D:\apps\Typora\Typora.exe
```

对于有变量的命令，可以用函数进行间接设置：

```shell
function todo_func {D:\apps\Typora\Typora.exe D:\我的坚果云\todo\todo.md}
Set-Alias -Name todo -Value todo_func
```

#### 自动跳转

linux 下的 autojump 是真的好用，powershell 中的替代品为 z.lua。首先在 windows 中安装 [lua](https://www.runoob.com/lua/lua-environment.html) 环境，安装后检查环境变量等是否正常。之后下载 [z.lua](https://github.com/skywind3000/z.lua.git) `git clone https://github.com/skywind3000/z.lua.git`

最后在 powershell 的配置文件 `$PROFILE` 中引入插件即可：

```shell
Invoke-Expression (& { (lua D:\apps\z.lua\z.lua --init powershell) -join "`n" })
```

可能需要重启终端才能生效。另外 z.lua 采用了 **子串匹配** 的规则，而非最小编辑距离。因此如果要跳转到 `post` 文件夹的话，使用 `z pst` 是不行的。

#### 自动填充

`https://github.com/PowerShell/PSReadLine` 中安装该插件

同样在 powershell 的 `$PROFILE` 中配置以下插件

```shell
Set-PSReadLineKeyHandler -Key "Ctrl+z" -Function Undo # 设置 Ctrl+z 为撤销
Set-PSReadLineKeyHandler -Key UpArrow -Function HistorySearchBackward # 设置向上键为后向搜索历史记录
Set-PSReadLineKeyHandler -Key DownArrow -Function HistorySearchForward # 设置向下键为前向搜索历史纪录
Set-PSReadLineOption -PredictionSource History # 设置预测文本来源为历史记录
Set-PSReadlineKeyHandler -Key Tab -Function Complete # 设置 Tab 键补全
```

## CMD

CMD 个人使用的不多，放点链接直接跳过吧！

插件：[CLINK](https://chrisant996.github.io/clink/)

