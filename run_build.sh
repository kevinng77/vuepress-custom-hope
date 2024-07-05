#!/bin/zsh
#  the script will now exit immediately if any of the commands fail.
set -e

export NODE_OPTIONS="--max-old-space-size=8192"

cd "$(dirname "$0")"
echo -e "\033[35m >>> 脚本执行路径： `pwd` \033[0m"

git_branch="master"
temp_blog_path="./cleaned_post"
source_blog_path="../post/"
blog_public_folder_path="./src/.vuepress/public"
blog_output_folder_path="./src/posts"
dist_git_folder_path="./dist/"  # dist git 文件夹博客文件所在路径
dist_source_folder_path="./src/.vuepress/dist"
target_img_folder="/home/kevin/blog/src/.vuepress/public/assets"

echo -e "\033[36m >>> Switching to master branch..... \033[0m"

current_branch=$(git rev-parse --abbrev-ref HEAD)
# Check if the current branch is "github"
if [ "$current_branch" != "$git_branch" ]; then
    git checkout $git_branch
else
    echo -e "Already on $git_branch branch"
fi

# nvm_version=$(npm -v)
# if [ "$nvm_version" != "9.6.7" ]; then
#     nvm use 18
# else
#     echo -e "Already on nvm 18"
# fi
echo -e "\033[36m >>> 开始格式化博客文件 \033[0m"

python codecorrect/vuepress_formating.py \
    --source_blog_path=$source_blog_path \
    --output_blog_path=$temp_blog_path \
    --days=7 \
    --blog_public_folder_path=$blog_public_folder_path \
    --target_img_folder=$target_img_folder

echo -e "\033[36m >>> 转移格式化后的博客文件 \033[0m"
echo -e "执行 cp -r $temp_blog_path/notes/articles/* $blog_output_folder_path/notes/articles/"

cp -r $temp_blog_path/notes/articles/* $blog_output_folder_path/notes/articles/


echo -e "\033[36m >>> 开始构建 dist 博客文件 \033[0m"

npm run docs:build

echo -e "\033[36m >>> 转移构建好的博客 dist 网页 \033[0m"
echo -e "执行 cp -r $dist_source_folder_path/* $dist_git_folder_path"

cp -r $dist_source_folder_path/* $dist_git_folder_path

cd $dist_git_folder_path

echo -e "\033[36m >>> Push 到服务器 \033[0m"
git add .
git commit -m "add blog"
echo -e "\033[36m >>> Push 到 github 服务器 \033[0m"
git push -f 
echo -e "\033[36m >>> Push 到腾云服务器 \033[0m"
git push -f kmtencent