import os

data_dir = "./data"
blog_dir = "/home/kevin/test/my-docs" # the path you clone the server repo

result4sample_img = os.path.join(data_dir, "sampleResult") 

cache_dir = os.path.join(data_dir, "cache")
server_dir = os.path.join(blog_dir, "src/.vuepress/public")

# 预先储存 sample input 对应的预测结果。
# result path 是 图片在 vuepress 服务器上的相对位置
sample_img_result = {
        "image url":"result path"
        }