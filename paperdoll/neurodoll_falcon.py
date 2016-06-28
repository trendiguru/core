__author__ = 'liorsabag'

import falcon
from .. import neurodoll, neurodoll_single_category
from .. import constants

from jaweson import json, msgpack


class PaperResource:
    def on_get(self, req, resp):
        """Handles GET requests"""
        quote = {
            'quote': 'I\'ve always been more interested in the future than in the past.',
            'author': 'Grace Hopper'
        }

        resp.body = json.dumps(quote)

    def on_post(self, req, resp):
        category_index = int(req.get_param('categoryIndex'))

        ret = {"success": False}
        try:
            data = msgpack.loads(req.stream.read())
            img = data.get("image")

            if category_index:
                ret["mask"] = neurodoll_single_category.get_category_graylevel(img, category_index) 
            else:
                ret["mask"] = neurodoll.infer_one(img)
            
            ret["label_dict"] = constants.ultimate_21
            if ret["mask"] is not None:
                ret["success"] = True
            else:
                ret["error"] = "No mask from ND"

        except Exception as e:
            ret["error"] = str(e)

        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200


api = falcon.API()
api.add_route('/nd/', PaperResource())
