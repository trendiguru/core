from importlib import import_module
import falcon
from jaweson import json, msgpack


class Classifier(object):
    def __init__(self, feature_name, package="trendi.features"):
        self.feature = import_module(".{0}".format(feature_name), package)

    def on_get(self, req, resp):
        """Handles GET requests"""
        quote = {
            'quote': 'I\'ve always been more interested in {0} than in the past.'.format(self.feature),
            'author': 'Grace Hopper'
        }

        resp.body = json.dumps(quote)

    def on_post(self, req, resp):
        ret = {"success": False}
        try:
            data = msgpack.loads(req.stream.read())
            ret["data"] = self.feature.execute(**data)
            ret["success"] = True
        except Exception as e:
            ret["error"] = str(e)
        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200
