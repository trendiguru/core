import traceback
# from importlib import import_module
import falcon
from jaweson import json, msgpack
from ..features.feature import Feature
from ..features.config import FEATURES
from ..Utils import get_cv2_img_array

class ClassifierResource(object):
    def __init__(self, feature_name, gpu_device=None):
        # self.feature = import_module(".{0}".format(feature_name), package)
        self.feature = Feature(feature_name)
        self.labels = FEATURES[feature_name].get("labels")
        self.feature.load(gpu_device=gpu_device)

    def on_get(self, req, resp):
        ret = {"success": False}
        try:
            image_url = req.get_param("imageUrl")
            ret = {"data": self.feature.execute(image_url),
                   "labels: self.labels,
                   "success": True}    
        except Exception as e:
            ret["error"] = str(e)
            ret["trace"] = traceback.format_exc()
        resp.body = json.dumps(ret)

    def on_post(self, req, resp):
        ret = {"success": False}
        try:
            data = msgpack.loads(req.stream.read())
            ret["data"] = self.feature.execute(**data)
            ret["success"] = True
            resp.status = falcon.HTTP_200
        except Exception as e:
            ret["error"] = str(e)
            ret["trace"] = traceback.format_exc()
        
        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        
