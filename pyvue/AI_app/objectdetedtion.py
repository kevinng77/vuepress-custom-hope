from .utils import img2base64


def objectdetection(input_img,  model):
    """
    Args:
        input_img [String]: URL of input image
        model [Str]: model_id to used for prediction
    Return:
        Dict:
            {
            "img": img file in base64 format,  // optional
            "text" : text result,              // optional
            "url": image url,                  // optional
            "data": other type of data         // optional
            }
    """


    result_path = 1
    
    response = {
        "img": img2base64(result_path),
        "url": None,
        "text" : None,
        "data": None
    }
    return