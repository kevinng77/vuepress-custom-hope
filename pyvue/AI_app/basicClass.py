


class PipeLine():
    """
    Construct AiApp for organize different Ai Apps processing steps.
    """

    def __init__(self, task) -> None:
        self._model = self.get_model()

    def get_model():
        return

    def _preprocess(self, img_file=None, form=None):
        return {"img_file": img_file, "form": form}

    def _postprocess(self, model_output):
        return model_output

    def _modelprediction(self):
        raise NotImplementedError

    def getResponse(self, img_file=None, form=None):
        model_inputs = self._preprocess(img_file=img_file)
        model_outputs = self._model(**model_inputs)
        response = self._postprocess(model_output=model_outputs)
        return response
