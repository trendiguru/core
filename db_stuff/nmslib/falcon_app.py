import falcon
import numpy as np

from jaweson import json, msgpack

import test_nmslib

class test:
    def on_get(self, req, resp):
        """Handles GET requests"""
        quote = {
            'quote': 'I\'ve always been more interested in the future than in the past.',
            'author': 'Grace Hopper'
        }

        resp.body = json.dumps(quote)

    def on_post(self, req, resp):
        ret = {"success": False}
        try:
            data = msgpack.loads(req.stream.read())
            fp = data.get("fp")
            ret["data"] = test_nmslib.find_to_k(fp)
            ret["success"] = True
        except Exception as e:
            ret["error"] = str(e)
        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200


api = falcon.API()
api.add_route('/test', test())
