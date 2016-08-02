__author__ = 'liorsabag'

import falcon
import matlab

from jaweson import json, msgpack
from . import pd

eng = matlab.engine.start_matlab('-nodesktop -nojvm')
print "Starting MATLAB engine {0}".format(eng)

class PaperResource:
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

            img = data.get("image")

            # mask_np, label_dict, pose_np, filename
            ret["mask"], ret["label_dict"], ret["pose"], ret["filename"] = pd.get_parse_mask_parallel(eng, img)
            if ret["mask"] is not None:
                ret["success"] = True
            else:
                ret["error"] = "No mask from PD"

        except Exception as e:
            ret["error"] = str(e)

        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200


api = falcon.API()
api.add_route('/pd/', PaperResource())
