__author__ = 'yonatan'

import falcon
import numpy as np
from jaweson import json, msgpack
from ..yonatan import new_genderDetector


class NeuralResource:
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
            image = data.get("image_or_url")
            face = data.get("face")
            print('gender_app nEURALrESOURCE got face {}'.format(face))
            ret["gender"] = new_genderDetector.theDetector(image, face)
            if ret["gender"] is not None:
                ret["success"] = True
            else:
                ret["error"] = "NN returned None, FACE="+str(face)
                ret["face"] = face
        except Exception as e:
            ret["error"] = str(e)

        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200


api = falcon.API()
api.add_route('/gender', NeuralResource())
