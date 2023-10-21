import re 

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


def UpdateYaml4Vuepress(lines):
    """
    根据 vuepress 修改 tags 和 categories。
    """
    update_dict = {
        r"Notes\|理论梳理": "知识笔记",
        "Notes": "知识笔记",
        "理论梳理": "知识笔记",

    }
    if not lines[0].startswith("---"):
        assert f"{lines[0:5]} has no yaml config"
    i = 1
    while i<len(lines):
        if lines[i].startswith("---"):
            break
        i += 1
    for j in range(1, i):
        if lines[j].startswith("tags"):
            lines[j] = re.sub("tags", "tag", lines[j] )
        if lines[j].startswith("categories"):
            lines[j] = re.sub("categories", "category", lines[j] )
        lines[j] = re.sub(r"Notes\|理论梳理", "知识笔记", lines[j] )

        for key, value in update_dict.items():
            lines[j] = re.sub(key, value, lines[j] )

    return lines

def createEditLinePipeline(functions):
    """
    创建文章处理 pipeline
    """
    def wrapper(s):
        # input a string of article line text.
        for func in functions:
            assert hasattr(func, '__call__'),f"{type(func)} is not function"
            s = func(s)
        return s
    return wrapper

if __name__ == "__main__":
    
    s = """
    2、A Deep Reinforced Model for Abstractive Summarization
3、Incorporating Copying Mechanism in Sequence-to-Sequence Learning
4、Get To The Point: Summarization with Pointer-Generator Networks"""
    lines = s.split("\n")
    print(process_list_line(lines))
