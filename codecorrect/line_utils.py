import re
from PIL import Image
import os

def centerIMG(s):
    """
    将图片居中
    """
    # <img src="image-20220408180900669.png" alt="image-20220408180900669" style="zoom:50%;" />
    # for halo2
    s = re.sub(r"(<img.+?>)",r'<div style="text-align:center">\1</div>',s)

    return s
    
def htmlImg2mdImg(s, blog_public_folder_path):

    file_path = re.findall(r"<img src=\"/?(.+?)\".+/?>",s)

    if len(file_path) > 0:
        assert len(file_path) == 1
        img_path = os.path.join(blog_public_folder_path, file_path[0])

        if os.path.isfile(img_path):
            w, h = Image.open(img_path).size
            if w/h > 2 :
                s = re.sub(r"<img src=\"(.+?)\".+/?>",r'![相关图片](\1 )',s)
            else:
                s = re.sub(r"<img src=\"(.+?)\".+/?>",r'![相关图片](\1 =x300)',s)
        elif file_path[0].startswith("htt"):
            
            # print(f">>> update http img path {file_path[0]}")
            s = re.sub(r"<img src=\"(.+?)\".+/?>",r'![相关图片](\1 )',s)
        else:
            print(f">> <img> img path missing : {img_path}")
    return s

def img_md_format_extract(s):
    """将 [] 中的 大小 format: `=100x200` 字样移到后面的 () 中，以适配 vuepress 格式
    """
    s = re.sub(r"\!\[(.+) (=\d*x\d*)\]\((.+?)\)", r'![\1](\3 \2)',s)

    return s
    

def log_missing_img(s, blog_public_folder_path):

    img_path = re.findall(r'\!\[.+\]\((.+?)( +=x\d*)?\)',s)
    if len(img_path) > 0:
        img_path = img_path[0][0]
        if img_path.startswith("/"):
            img_path = img_path[1:]
        file_path = os.path.join(blog_public_folder_path, img_path)
        file_path = re.sub(" ","",file_path)
        if not os.path.isfile(file_path) and not img_path.startswith("http"):
            print(f">>> []() format img file_missing {file_path}")
    return s

def bolder_edit(s):
    """
    markdown 格式中，**加粗内容** ，加粗符号内如果有空格链接，会导致识别错误。
    """
    s = re.sub(r"(\*\*) *([^\*]+?) *(\*\* *)",r" \1\2\3 ",s)
    return s


def img_path_edit(s, image_bed=None):
    """
    更换资源到博客相对路径地址
    """

    s = re.sub(r"(/)?img/",r"/img/",s)
    # s = re.sub(r"(/upload)?/img/",r"/upload/img/",s)
    # s = re.sub(r"(/upload)?/img/",r"../../../img/",s)
    if image_bed:
        if image_bed.endswith("/"):
            image_bed = image_bed[:-1]
        s = re.sub(r"(/upload)?/img/", image_bed + r"/assets/img/",s)
    else:
        s = re.sub(r"(/upload)?/img/",r"/assets/img/",s)


    return s

def vuepress_hope_format(s):
    s = re.sub(r"::: details?(.+)?", r"::: details\1",s )
    return s

def code_block_edit(s):
    s = re.sub(r"```[cC]\+\+",r"```cpp",s)
    s = re.sub(r"```[cC]\#",r"```c",s)
    s = s.lower()
    return s


def sep_ch_and_en(s):
    # 在中文和英文之间添加空格。
    s = re.sub(r"([\u4e00-\u9fa5]+)([a-zA-Z0-9])",r"\1 \2",s)
    s = re.sub(r"([a-zA-Z0-9])([\u4e00-\u9fa5]+)",r"\1 \2",s)

    return s

if __name__ == "__main__":
    
    # s = """<img src="https://pic2.zhimg.com/80/v2-a1677bec536477c83e93a14f2c452ee9_1440w.jpg" style="zoom: 67%;">
    # """
    # print(re.findall(r"<img src=\"/?(.+?)\".+/?>",s))
    # print(os.path.join('/home/kevin/test/my-docs/src/.vuepress/public', "123123.png"))
    # s = r"![img x300](/img/path/123/awer.png)"
    # s = re.sub(r"\!\[(.+) (\d*x\d*)\]\((.+?)\)", r'![\1](\3 \2)',s)

    # print(s)
    import os
    import re
    # path = "/home/kevin/nut/post/nlp/hf_co_model_note.md"
    # file_name = os.path.basename(path)
    # path = os.path.join(os.path.dirname(path), f'img/{re.sub(".md", "", file_name)}')
    
    # print(os.path.exists(path))
    # print(os.path.basename("/home/kevin/nut/post/nlp/distill_flan.md"))
    test = "::: details 123"
    print(vuepress_hope_format(test))