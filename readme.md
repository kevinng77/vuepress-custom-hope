# 快速开始

## 准备
安装  npm 9.6.7 node v18.17.1
```bash
nvm alias default 18
nvm install 18
nvm use 18
npm install .
```

### dist 文件

1. 将博客静态文件仓库 clone 到本地的 `./dist` 文件夹中

### 草稿博客

2. 将草稿博客文件夹连接到 `./post`

```bash
ln -s ~/baidusyncdisk/post ./post
```

3. 开始处理：
```bash
zsh ./run_build.sh
```