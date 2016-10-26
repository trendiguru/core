__author__ = 'liorsabag'
# labels for pixel level parsing (neurodoll) are in constants.ultimate21 (21 labels)
# labels for multilabel image-level categorization are in constants.web_tool_categories (also 21 labels)
import traceback
import falcon
from .. import multilabel_from_binaries2
from .. import constants
# from .darknet.pyDarknet import mydet

from jaweson import json, msgpack

print "Done with imports"

class PaperResource:
    def __init__(self):
        print "Loaded Resource"


    def on_get(self, req, resp):
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

            output = multilabel_from_binaries.get_multiple_single_label_outputs(img)
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
api.add_route('/mlb2/', PaperResource())
