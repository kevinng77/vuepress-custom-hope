from flask import Flask, request, make_response
from flask_cors import CORS
import json
import os
from config import *
# from AI_app import PipeLine

app = Flask(__name__)
CORS(app)
{
    "task name": "target AI class",
    "stamp detection": "ObjectDetection" 
}
# pipeline = PipeLine(output_dir=cache_dir,
#                     server_dir=server_dir)


@app.route('/',methods=['POST'])
def aiapp():

    form = request.form
    print(form)
    # 用户选择了样本图片
    # img_file = request.files.get('file')
    # response = pipeline.getResponse(form=form, img_file=img_file)
    # return response

    # print(task)
    if form.get('img_url', False):
        input_url = form.get('img_url')
    #     if sample_img_result.get(input_url, False):
    #         response = {"url":input_url}
    #     else:
    #         response = objectdetection(input_img=input_url, 
    #                                    task=task)
    else:
        img = request.files.get('file')
        print(img)
    #     input_url = os.path.join(output_dir, img.filename)
    #     img.save(input_url)
    #     response = objectdetection(input_img=input_url, 
    #                                task=task)
    return "123"
    # return json.dumps(response)



if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)