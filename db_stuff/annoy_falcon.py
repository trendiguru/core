import falcon
from .fanni import load_all_forests
from jaweson import json, msgpack
from time import time


class PaperResource:
    def __init__(self):
        self.forests = load_all_forests()

    def on_get(self, req, resp):
        """Handles GET requests"""
        quote = {
            'quote': 'I\'ve always been more interested in the future than in the past.',
            'author': 'Grace Hopper'
        }

        resp.body = json.dumps(quote)

    def on_post(self, req, resp):
        t1= time()
        category = req.get_param('category')
        col_name = req.get_param('col_name')
        ret = {"success": False}
        key = col_name+'.'+category
        forest_handle = self.forests[key]
        try:
            data = msgpack.loads(req.stream.read())
            fp = data.get("fingerprint")

            ret['id_list']  = forest_handle.get_nns_by_vector(fp, 1000)
            if ret["id_list"] is not None:
                ret["success"] = True
            else:
                ret["error"] = "No list"

        except Exception as e:
            ret["error"] = str(e)
        t2 = time()
        duration = t2-t1
        ret['duration'] = duration
        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200


api = falcon.API()
api.add_route('/nd/', PaperResource())