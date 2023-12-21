import re
import subprocess
import os

def has_ch_char(string):
    return len(re.findall(r"([^\x00-\xff]+)", string))>0


def label_math_line(lines):
    """
    1. 标记所有存在数学公式的行
    2. 对于 $$\n(.+)\n$$ 的数学公式内容，在数学模块前后添加换行符。

    Args:
        lines(List[string]): 博客内容列表，列表中每一项为博客中的一行
    Return:
        labels: List[int]: 1 表示该行中存在数学公式内容
        text: 经过预处理的博客文章内容
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


def article_has_math(lines):
    labels, texts = label_math_line(lines)
    return sum(labels) > 0

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


def edit_math_line(line, file_name, line_id=None):
    sub_text,part_labels = label_math_part(line)
    for i in range(len(sub_text)):
        if part_labels[i] == 1:
            sub_text[i] = math_part_formating(sub_text[i])

            sub_text4check = re.sub(r"\\text ?\{.*?\}", r"textpart", sub_text[i])
            if has_ch_char(sub_text4check):
                print(f">>> {file_name} has chinese char . in 【{line_id}】")
                # print(sub_text[i], "\n---->", sub_text4check)

    return "$".join(sub_text)


def EditBlogMath(lines, file_name):
    """
    处理一篇文章中的所有数学公式内容
    """
    line_labels, text = label_math_line(lines)
    for i in range(len(text)):
        if line_labels[i] == 1:
            if text[i].strip() == "" or text[i].count(" ") == len(text[i].strip()):
                continue
            text[i] = edit_math_line(text[i], file_name=file_name, line_id= i)
    return text
    

def math_part_formating(s):
    """
    输入数学公式内容，针对数学公式内容进行 latex 格式修正
    """
    s = re.sub(r"(^ )?(.+?)( $)?",r"\2",s)

    # 去除数学公式中的中文符号
    s = re.sub(r"，",r",",s)
    s = re.sub(r"：",r":",s)
    s = re.sub(r"！",r"!",s)
    s = re.sub(r"（",r"(",s)
    s = re.sub(r"）",r")",s)
    s = re.sub(r"−",r"-",s)
    s = re.sub(r"’",r"'",s)
    s = re.sub(r"。",r".",s)
    s = re.sub(r"…",r"...",s)
    s = re.sub(r"μ",r"\\mu ",s)
    s = re.sub(r"θ",r"\\theta ",s)
    s = re.sub(r"𝚪",r"\\Gamma ",s)
    s = re.sub(r"…",r"...",s)
    s = re.sub(r"⟨", r"\\langle ", s)
    s = re.sub(r"⟩", r"\\rangle ", s)
    s = re.sub(r"𝜎", r"\\sigma ", s)
    for key, value  in sym2latex.items():
        s = re.sub(key, value, s)

    # s = re.sub(r"则",r"\ then\ ",s)
    if r"\text" not in s:
        s = re.sub(r"([\u4e00-\u9fa5]+)",r"\\text{ \1 }",s)
        result = s
    else:
        texts = re.findall(r"\\text ?{.*?}", s)
        s = re.split(r"\\text ?{.*?}",s)
        result = ""
        for i in range(len(texts)):
            result += s[i]
            c = texts[i]
            c = re.sub(r"_", r" ", c)
            c = re.sub(r"&", r"and", c)
            result += c
        result += s[-1]



    # /text 中不能用下划线

    # 数学公式中，不能包含对于的括号
    return result


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

sym2latex = {
    r"𝐖":r"\\mathbf{W}",
    r"𝜶":r"\\alpha",
    r"𝑜": r"o",
    r"𝐱": r"x",
    r"Σ": r"\\Sigma",
    r"𝐛": r"b",
    r"≈": r"\\approx",
    r"τ": r"\\tau",
    r"𝑡": r"t"


}
if __name__ == "__main__":
    s = r"""𝚪^{⟨𝑡⟩}_o=𝜎(𝐖_𝑜[𝐚^{⟨𝑡-1⟩},𝐱^{⟨𝑡⟩}]+𝐛_𝑜)"""
    # s = re.sub(r"\\text ?{.*?}", r"textpart", s)
    # print(has_ch_char(s))
    print(math_part_formating(s))