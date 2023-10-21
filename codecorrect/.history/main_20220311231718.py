import os


def codedigit(s):
    beg = ''
    if s.strip().isdigit():
        return ''
    i = 0
    if s.startswith("> "):
        i = 2
        beg = "> "
    while (s[i].isdigit()):
        i += 1
    
    if s[i] == ' ':
        return beg + s[i+1:]
    else:
        return s

def main():
    path = "/mnt/together/nut/source/_posts/原始套接字.md"
    text = []
    with open(path,"r") as fp:
        lines = fp.readlines()
        for line in lines:
            to_write = codedigit(line)
            text.append(to_write)
    
    with open(path,"w") as fp:
        for line in text:
            fp.write(line)
        
        
if __name__ == '__main__':
    main()
    