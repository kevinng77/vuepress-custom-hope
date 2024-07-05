
git_branch="master"
temp_blog_path="./cleaned_post"
source_blog_path="/home/kevin/post/"
blog_public_folder_path="/home/kevin/blog/src/.vuepress/public"
blog_output_folder_path="/home/kevin/blog/src/posts"
dist_git_folder_path="/home/kevin/dist/"  # dist git 文件夹博客文件所在路径
dist_source_folder_path="/home/kevin/blog/src/.vuepress/dist"
target_img_folder="/home/kevin/blog/src/.vuepress/public/assets"
python codecorrect/vuepress_formating.py \
    --source_blog_path=$source_blog_path \
    --output_blog_path=$temp_blog_path \
    --days=7 \
    --blog_public_folder_path=$blog_public_folder_path \
    --target_img_folder=$target_img_folder

echo -e "执行 cp -r $temp_blog_path/notes/articles/* $blog_output_folder_path/notes/articles/"

cp -r $temp_blog_path/notes/articles/* $blog_output_folder_path/notes/articles/
