import traceback
import falcon

from trendi import constants
from trendi import multilabel_from_hydra

from jaweson import json, msgpack

print "Done with imports"

class HydraResource:
    def __init__(self):
        print "Loaded Resource"


    def on_get(self, req, resp): #
        """Handles GET requests"""
        quote = {
            'quote': 'just work already ',
            'author': 'jeremy rutman'
        }
        resp.body = json.dumps(quote)

    def on_post(self, req, resp):
        print "Reached on_post"
        gpu = req.get_param('gpu')
        ret = {"success": False}

        try:
            data = msgpack.loads(req.stream.read())
            img = data.get("image")

            output = multilabel_from_hydra.get_hydra_output(img)
            ret["output"] = output
            if ret["output"] is not None:
                ret["success"] = True
            else:
                ret["error"] = "No output from mlb"

        except Exception as e:
            traceback.print_exc()
            ret["error"] = traceback.format_exc()

        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200


api = falcon.API()
#api.add_route('/mlb3/', HydraResource())
api.add_route('/hydra/', HydraResource())
#send post request to http://ip-of-server/hydra
#make sure to run docker container with -p