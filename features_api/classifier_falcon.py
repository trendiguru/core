import traceback
# from importlib import import_module
import falcon
from jaweson import json, msgpack
from ..features.feature import Feature

class ClassifierResource(object):
    def __init__(self, feature_name, gpu_device=None):
        # self.feature = import_module(".{0}".format(feature_name), package)
        self.feature = Feature(feature_name)
        self.feature.load(gpu_device=gpu_device)

    def on_get(self, req, resp):
        """Handles GET requests"""
        quote = {
            'quote': 'I\'ve always been more interested in {0} than in the past.'.format(self.feature.name),
            'author': 'Grace Hopper'
        }

        resp.body = json.dumps(quote)

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
        
