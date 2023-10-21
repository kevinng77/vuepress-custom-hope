import os
import re


def is_inline_math(s):
    # 判断是否为行间公式
    return 

def has_math(s):
    if s.count("$")>=2:
        return True
    return False

# 删除数学公式间空格


def math_edit(s, is_math_block):
    if len(s) > 3:
        
        s = re.sub(r"\$\$", r"$",s)  # 替换所有行间公式为 $.
        
        if "$" in s or is_math_block:
            s = re.sub(r"\\",r"\\\\",s)
            s = re.sub(r"\*",r"\*",s)
            s = re.sub(r"_",r"\_",s)
            s = re.sub(r"\(",r"\(",s)
            s = re.sub(r"\)",r"\)",s)
            s = re.sub(r"\[",r"\[",s)
            s = re.sub(r"]",r"\]",s)
            s = re.sub(r"\^",r"\^",s)

            s = re.sub(r"{",r"\{",s)
            s = re.sub(r"}",r"\}",s)
    return s


def bloder_edit(s):
    s = re.sub(r"(\*\*) ?(\S+) ?(\*\* ?)",r" \1\2\3 ",s)
    s = re.sub(r" +",r" ",s)
    return s

def img_path_edit(s):
    s = re.sub(r"(/upload)+/img/",r"/upload/img/",s)
    return s


def sep_math_file():
    no_math_files = ["shell.md", "hadoop.md", "docker.md", "blog.md","mongodb.md","linux.md" ]
    for file in os.listdir("../_posts/"):
        if file.endswith("md"):
            path = "../_posts/" + file
            text = []
            m = False
            with open(path,"r") as fp:
                lines = fp.readlines()
                for line in lines:
                    m = m or has_math(line)
                    line = bloder_edit(line)
                    text.append(line)
            if m and file not in no_math_files:
                math_path = "../has_math/" + file
                with open(math_path,"w") as fp:
                    for line in text:
                        fp.write(line)
            else:
                no_math_path = "../no_math/" + file
                with open(no_math_path,"w") as fp:
                    for line in text:
                        fp.write(line)


def main_edit_math():
    for file in os.listdir("../has_math/"):
        if file.endswith("md"):
            math_path = "../has_math/" + file
            text = []
            is_math_block = False
            with open(math_path,"r") as fp:
                lines = fp.readlines()
                for line in lines:
                    line = line
                    if line == "$$\n":
                        is_math_block = ~is_math_block
                    line = math_edit(line, is_math_block)
                    text.append(line)
            math_path = "../has_math/" + file
            with open(math_path,"w") as fp:
                for line in text:
                    fp.write(line)


if __name__ == '__main__':
    sep_math_file()
    main_edit_math()
    s = r"  ** 年后** "
    t = bloder_edit(s)
    print(t)
    