import falcon
import datetime

from jaweson import json, msgpack
from . import new_pd

import random
import string


def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))

rand_eng_name = randomword(4)
print "{0}: Starting MATLAB engine {1}".format(datetime.datetime.now(), rand_eng_name)
eng = new_pd.init_pd_eng()
print "{0}: Started MATLAB engine {1}".format(datetime.datetime.now(), rand_eng_name)


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
            ret["mask"], ret["label_dict"], ret["pose"], ret["filename"] = new_pd.parse(img, _eng=eng)
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
