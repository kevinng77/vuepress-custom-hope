import re

def centerIMG(s):
    # <img src="image-20220408180900669.png" alt="image-20220408180900669" style="zoom:50%;" />

    s = re.sub(r"(<img.+?>)",r'<div style="text-align:center">\1</div>',s)
    return s

def math_part_formating(s):
    s = re.sub(r"(^ )(.+?)( $)",r"\2",s)
    # s = re.sub(r"\\",r"\\\\",s)
    # s = re.sub(r"\*",r"\*",s)
    # s = re.sub(r"_",r"\_",s)
    # s = re.sub(r"\(",r"\(",s)
    # s = re.sub(r"\)",r"\)",s)
    # s = re.sub(r"\[",r"\[",s)
    # s = re.sub(r"]",r"\]",s)
    # s = re.sub(r"\^",r"\^",s)
    # s = re.sub(r"{",r"\{",s)
    # s = re.sub(r"}",r"\}",s)
    # s = re.sub(r"<",r"\<",s)
    # s = re.sub(r">",r"\>",s)
    return s
    

def bolder_edit(s):
    s = re.sub(r"(\*\*) *([^\*]+?) *(\*\* *)",r" \1\2\3 ",s)
    return s


def img_path_edit(s):
    s = re.sub(r"(/)?img/",r"/img/",s)
    s = re.sub(r"(/upload)?/img/",r"/upload/img/",s)
    return s


def code_block_edit(s):
    s = re.sub(r"```[cC]\+\+",r"```cpp",s)
    s = re.sub(r"```[cC]\#",r"```c",s)

    return s


def sep_ch_and_en(s):
    # 在中文和英文之间添加空格。
    s = re.sub(r"([\u4e00-\u9fa5]+)([a-zA-Z0-9])",r"\1 \2",s)
    s = re.sub(r"([a-zA-Z0-9])([\u4e00-\u9fa5]+)",r"\1 \2",s)

    return s


if __name__ == '__main__':
    print('123'+math_part_formating(' 123 456 789 '))