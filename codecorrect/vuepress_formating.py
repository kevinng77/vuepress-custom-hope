import os
import subprocess
import time
import os
import time
import datetime
import argparse
from article_utils import *
from line_utils import *
from math_utils import EditBlogMath, has_ch_char


def edit_article(source_file, output_file, pre_processer):
    global ALL_CHARS

    text = []
    with open(source_file, "r") as fp:
        for line in fp.readlines():
            text.append(line)
            
    if len(text) == 0:
        print(f"忽略文章 {source_file}: 空文件")
        return
    elif not text[0].startswith("---"):
        print(f"忽略文章 {source_file}: 没有yaml config")
        return False
    
    # 复制图片到目录文件夹
    file_name = os.path.basename(source_file)
    folder_name  = f'img/{re.sub(".md", "", file_name)}'
    source_img_folder = os.path.join(os.path.dirname(source_file),folder_name)
    target_img_folder = os.path.join("/home/kevin/test/blog/src/.vuepress/public/assets", folder_name)
    # if os.path.isdir(target_img_folder):
    #     os.system(f"rm -r {target_img_folder}")
    if os.path.exists(source_img_folder):
        print(f">>> 复制图片文件夹 {source_img_folder} 到 {target_img_folder}")
        os.system(f"cp -r {source_img_folder} {target_img_folder}")
    
    # 对整篇博客统一处理
    text = UpdateYaml4Vuepress(text)
    # text = process_list_line(text)
    text = EditBlogMath(text, file_name=source_file)
    # 对博客逐行进行处理
    text = [pre_processer(line) for line in text]

    # halo 博客需要用到
    # has_math = article_has_math(text)
    # text.append('\n<link rel="stylesheet" href="https://unpkg.com/katex@0.12.0/dist/katex.min.css" />')

    
    with open(output_file,'w') as fp:
        for line in text:
            fp.write(line)
            
    return True

def get_article_list(source_blog_path, output_blog_path):
    """
    返回一个目录下所有需要编辑的博客文章
    """
    files = []
    os.makedirs(os.path.join(output_blog_path, "xijie/articles"))
    os.makedirs(os.path.join(output_blog_path, "notes/articles"))

    for file in os.listdir(source_blog_path):  # no math
        if file.endswith(".md"):
            source_file = source_blog_path + file
            output_file = os.path.join(output_blog_path,"notes/articles",inner_files) 
            files.append((source_file, output_file, file))
        elif file.startswith("bin"):
            continue
        elif os.path.isdir(source_blog_path + file):
            for inner_files in os.listdir(source_blog_path + file):
                if inner_files.endswith(".md"):

                    # 处理文件名称
                    inner_files_out = re.sub(r"\+", r"plus", inner_files)
                    if not has_ch_char(inner_files_out):
                        inner_files_out = "笔记" + inner_files_out

                    source_file = source_blog_path + file + '/' +  inner_files
                    if "xijie" in source_file:
                        output_file = os.path.join(output_blog_path,"xijie/articles",inner_files_out)
                        files.append((source_file, 
                            output_file, inner_files))
                    else:
                        output_file = os.path.join(output_blog_path,"notes/articles",inner_files_out) 

                        files.append((source_file, output_file, inner_files))
        
    return files

def main(args, process_pipelines=[]):
    """formating markdown file for halo blog.

    Args:
        output_blog_path (List[str]): list of article directory, 
        only markdown file will be process.
    """
    
    source_blog_path, output_blog_path = args.source_blog_path, args.output_blog_path
    
    print(f"处理近 {args.days} 天修改过的markdown文件")
    print(f"加载目标文件夹笔记：{source_blog_path}" )
    print(f"输出笔记到：{output_blog_path}" )
    print("开始处理")
    if os.path.exists(output_blog_path):
        p = subprocess.Popen(f"rm -r {output_blog_path}*", shell=True)
        p.wait()
    else:
        os.makedirs(output_blog_path)

    cur_time = TimeStampToTime(time.time())
    pre_processer = createEditLinePipeline(process_pipelines)
    
    # 获取所有需要编辑的文章
    files = get_article_list(source_blog_path, output_blog_path)
    print(f"一共有 {len(files)} 篇文章要处理\n")

    processed_files = []
    for source_file, output_file, file_name in files:
        file_time = get_FileModifyTime(source_file)
        
        if (cur_time - file_time).days < args.days:
            # print("processing\t", source_file)
            status = edit_article(source_file = source_file,
                                    output_file = output_file,
                                    pre_processer= pre_processer)
            if status:
                processed_files.append((file_time, file_name))

    
    print(">>>>>>>>>>>>>>>>>> 处理完毕 >>>>>>>>>>>>>>>>>>>\n")
    for file_time, file_name in sorted(processed_files, key=lambda x:x[0], reverse=True):
        print(f"处理文章： {file_name}, \t修改时间: {file_time}")
    
#     if not args.ignore_ttf_subset:
#         gen_ttf_subset(args.ttf_path, 
#         target_charset=ALL_CHARS,
#         include=set("""爱弹吉福希望一忙刺桐见年百年老故录民我者石狮居民帮古城漫千古街温陵掌贤逸事记忆笔书首页嘉
# """),
#         exclude=set([chr(ord('a')+i) for i in range(26)]) | set([chr(ord('A')+i) for i in range(26)])
#         )


def TimeStampToTime(timestamp):
    return datetime.date.fromtimestamp(timestamp)

def get_FileCreateTime(filePath):
    t = os.path.getctime(filePath)
    return TimeStampToTime(t)


def get_FileModifyTime(filePath):
    t = os.path.getmtime(filePath)
    return TimeStampToTime(t)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",default=7,type=int)
    parser.add_argument("--source_blog_path", required=True, type=str, 
                        help="存放未格式化 md 文件的文件夹")
    parser.add_argument("--output_blog_path", required=True, type=str,
                        help="输出的格式化后 md 文件的路径")
    parser.add_argument("--name",default="",type=str)
    parser.add_argument("--ignore_ttf_subset",action="store_true")
    parser.add_argument("--ttf_path",default=".")
    parser.add_argument("--image_bed",type=str,default=None,
                        help="当部署服务器博客时，采用的图床 root 地址")
    parser.add_argument("--blog_public_folder_path", type=str, 
                        default=r'/home/kevin/test/blog/src/.vuepress/public', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    from functools import partial
    args = get_args()
    if args.image_bed is None:
        print("不适用任何图床...")
    else:
        print(f"采用图床 {args.image_bed}")
    process_pipelines = [partial(img_path_edit, image_bed=args.image_bed),
                         bolder_edit,
                         sep_ch_and_en, 
                         code_block_edit,
                         img_md_format_extract,
                         partial(htmlImg2mdImg, blog_public_folder_path=args.blog_public_folder_path),
                         partial(log_missing_img, blog_public_folder_path=args.blog_public_folder_path)
                         ]
    main(args, process_pipelines=process_pipelines)
    # pre_processer = createEditLinePipeline(process_pipelines)

    # edit_article(source_file="/home/kevin/nut/post/nlp/ann.md", 
    # output_file="./test.md", pre_processer=pre_processer)