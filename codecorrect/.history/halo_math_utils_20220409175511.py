import os
import re
import subprocess
import time
import os
import time
import datetime


def article_has_math(lines):
    labels, texts = label_math_line(lines)
    return sum(labels) > 0
    

def label_math_line(lines):
    """label math line with 1
    """
    labels = []
    texts = []
    is_code = 0
    is_math_block = 0
    for i in range(len(lines)):
        assert is_code + is_math_block != 2, \
        f"{lines[i]} conflict on math and code\n{lines[:5]}\t{i}"
        line = lines[i]
        if line.strip() == "$$":
            is_math_block ^= 1
            if is_math_block == 1:
                if lines[i-1].strip() != "":
                    texts.append("\n")
                    labels.append(0)
                texts.append(line)
                labels.append(0)
            else:
                texts.append(line)
                labels.append(0)
                if lines[i+1].strip() != "":
                    texts.append("\n")
                    labels.append(0) 

        else:
            texts.append(line)
            if is_math_block == 1:
                labels.append(1)
            elif line.startswith("```"):
                is_code ^= 1
                labels.append(0)
            elif line.count('$') > 1:
                if is_code:
                    labels.append(0)
                    continue
                # 判断行间代码块
                is_inline_code = 0
                for char in line:
                    if char == '`':
                        is_inline_code ^= 1
                    elif char == "$" and is_inline_code == 0:
                        labels.append(1)
                        break 
                else:
                    labels.append(0)
            else:
                labels.append(0)
    assert len(labels) == len(texts), f"{len(labels)} not match {len(texts)}"
    return labels, texts


def label_math_part(line):
    line = re.sub(r"\$\$", r"$",line)  # 替换所有行间公式为 $.
    if line.count('$') == 0:
        return [line],[1]
    labels = []
    texts = line.split('$')
    tag = 0
    for text in texts:
        labels.append(tag)
        tag ^= 1
    return texts, labels


def edit_math_line(line):
    sub_text,part_labels = label_math_part(line)
    for i in range(len(sub_text)):
        if part_labels[i] == 1:
            sub_text[i] = math_part_formating(sub_text[i])
            
    return "$".join(sub_text)


def math_part_formating(s):
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

def sep_math_file(source_path, output_path):
    no_math_files = []

    for path in output_path:
        p = subprocess.Popen(f"rm {path}*", shell=True)
        p.wait()
    
    for file in os.listdir(source_path):
        if file.endswith("md"):
            path = source_path + file
            with open(path,"r") as fp:
                lines = fp.readlines()
                m = article_has_math(lines)
            
            if m and file not in no_math_files:
                math_path = output_path[0] + file
                with open(math_path,"w") as fp:
                    for line in lines:
                        fp.write(line)
            else:
                no_math_path = output_path[1] + file
                with open(no_math_path,"w") as fp:
                    for line in lines:
                        fp.write(line)


def createEditPipeline(functions):
    def wrapper(s):
        # input a string of article line text.
        for func in functions:
            assert hasattr(func, '__call__'),f"{type(func)} is not function"
            s = func(s)
        return s
    return wrapper
    

def process_list_line(lines):
    """处理 markdown 中以 - 或者 + 开头的列举内容。删除列举内容之间的空白行。

    Args:
        lines (List(string)): 每一项为文章一行。

    Returns:
        List(string): 每一项为文章一行。
    """
    list_tag = ["- ","+ "] + [f"{i}." for i in range(1, 10)]
    new_lines = [lines[0]]
    for i in range(1, len(lines)-1):
        if lines[i].strip() == "" and len(lines[i-1]) > 2 and len(lines[i+1])>2:
            if lines[i-1][:2] in list_tag and lines[i+1][:2] in list_tag:
                continue
        new_lines.append(lines[i])

    return new_lines

def edit_article(source_file, output_file, pre_processer):
    text = []
    with open(source_file, "r") as fp:
        for line in fp.readlines():
            text.append(pre_processer(line))
            
    if len(text) == 0:
        print(f"{source_file} is empty")
        return
    text = process_list_line(text)
    has_math = article_has_math(text)
    text.append('\n<link rel="stylesheet" href="https://unpkg.com/katex@0.12.0/dist/katex.min.css" />')
    if has_math:
        with open(output_file,'w') as fp:
            line_labels, text = label_math_line(text)
            for i in range(len(text)):
                if line_labels[i] == 1:
                    if text[i].strip() == "" or text[i].count(" ") == len(text[i].strip()):
                        continue
                    text[i] = edit_math_line(text[i])
                    fp.write(text[i])
                else:
                    fp.write(text[i])
    else:
        with open(output_file,'w')as fp:
            for line in text:
                fp.write(line)


def main(source_path, output_path, process_pipelines=[]):
    """formating markdown file for halo blog.

    Args:
        output_path (List[str]): list of article directory, 
        only markdown file will be process.
    """
    if os.path.exists(output_path):
        p = subprocess.Popen(f"rm {output_path}*", shell=True)
        p.wait()
    else:
        os.makedirs(output_path)

    cur_time = TimeStampToTime(time.time())
    pre_processer = createEditPipeline(process_pipelines)

    for file in os.listdir(source_path):  # no math
        if file.endswith(".md"):
            source_file = source_path + file
            file_time = get_FileModifyTime(source_file)
            
            print(f"processing {source_file}, create time: {file_time} \n")
            
            output_file = output_path + file
            edit_article(source_file = source_file,
                            output_file = output_file,
                            pre_processer=pre_processer)


def TimeStampToTime(timestamp):
    return datetime.date.fromtimestamp(timestamp)


def get_FileCreateTime(filePath):
    t = os.path.getctime(filePath)
    return TimeStampToTime(t)


def get_FileModifyTime(filePath):
    t = os.path.getmtime(filePath)
    return TimeStampToTime(t)

if __name__ == '__main__':
    source_path = "/mnt/together/nut/_posts/"
    # output_path = ["../has_math/","../no_math/"]
    output_path = "../cleaned_posts/"
    print(f"加载目标文件夹笔记：{source_path}" )
    print(f"输出笔记到：{output_path}" )
    process_pipelines = [img_path_edit,bolder_edit,sep_ch_and_en, code_block_edit]

    
    main(source_path = source_path,
         output_path = output_path,
         process_pipelines=process_pipelines)

    # test = "/mnt/together/nut/_posts/shell.md"
    # print(os.stat(test))
    # a = os.path.getmtime(test)
    # print(datetime.date.fromtimestamp(a))
    # print(get_FileModifyTime(test))