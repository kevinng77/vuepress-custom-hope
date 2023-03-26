import base64



def img2base64(img_path):
    """
    读取本地的图片文件，并返回对应的 base64 编码。
    """
    with open(img_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = str("data:;base64," + str(base64.b64encode(img_stream).decode('utf-8')))
    return img_stream