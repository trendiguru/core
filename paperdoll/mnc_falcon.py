__author__ = 'liorsabag'
import falcon
from jaweson import json, msgpack
from .. import mnc_voc_pixlevel_segmenter

class MNCResource:
    def on_get(self, req, resp):
        """Handles GET requests"""
        quote = {
            'quote': 'Time is a very misleading thing. All there is ever, is the now. ',
            'author': 'George Harrison'
        }

        resp.body = json.dumps(quote)

    def on_post(self, req, resp):
        category_to_look_for  = req.get_param('catToLookFor')
        print('category to look for:'+str(category_to_look_for))
        ret = {"success": False}
        try:
            data = msgpack.loads(req.stream.read())
            img = data.get("image")
            mnc_mask,mnc_box = mnc_voc_pixlevel_segmenter.mnc_pixlevel_detect(img)
            if mnc_mask is not None:
                ret["success"] = True
                ret['mnc_output'] = [mnc_mask,mnc_box,im,im_name]

        except Exception as e:
            ret["error"] = str(e)

        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200


#help
api = falcon.API()
#i have no idea wtf is going on here
api.add_route('/mnc/', MNCResource())
