from tkinter import N
from utils import img2base64
import os

class PipeLine():
    """
    Construct pipeline for organize different Ai Apps processing steps.
    """


    def __init__(self, output_dir, server_dir) -> None:
        self._model = self.get_model()
        self.output_dir = output_dir
        self.server_dir = server_dir

    def get_model():
        return

    def _get_img_url(self, img_file, server_img_path):
        """
        Args:
            img_file [FileStorage]: img file from `request.files.get('file')`
            server_img_path [String]: it should be a related img path corresponding to vuepress folder
                        for example: `/img/ann/123456.png`
        Return:
            An absolute path of image file. or None.
        """
        local_img_path = None
        if img_file is not None:
            local_img_path = os.path.join(self.output_dir, img_file.filename)
            img_file.save(local_img_path)
        if server_img_path is not None:
            if server_img_path.startswith("/") :
                server_img_path = server_img_path[1:]
            local_img_path = os.path.join(self.server_dir, server_img_path)
        return local_img_path

    def _preprocess(self, img_file=None, form=None):
        """
        """
        local_img_path = self._get_img_url(img_file=img_file, server_img_path=form['img_url'])
        

        return {"img_file": img_file, "form": form}

    def _postprocess(self, model_output):
        return model_output

    def _modelprediction(self):
        raise NotImplementedError

    def getResponse(self, img_file=None, form=None):
        """
        Args:
            img_file [FileStorage]: img file from `request.files.get('file')`
            form [Dict]: form data of format:
                {
                    "task": "ai task name",  # Required
                    "img_url":"",            
                    # Optional, it should be a related img path corresponding to vuepress folder
                    "text":"",               # Optional
                    "options":"",            # Optional
                }
        Return:

        """
        model_inputs = self._preprocess(img_file=img_file, form=form)

        model = self.get_model()
        model_outputs = model(**model_inputs)
        response = self._postprocess(model_output=model_outputs)
        return response
