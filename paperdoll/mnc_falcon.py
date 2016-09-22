import traceback
import falcon
import json
from jaweson import msgpack
from .. import mnc_voc_pixlevel_segmenter as mnc


class MNCResource:
    def on_get(self, req, resp):

        url = req.get_param('url')

        show = req.get_param('show')

        if url:
            mnc_mask, mnc_box, im, im_name, orig_im, boxes, scalefactor, superimpose_name = mnc.mnc_pixlevel_detect(url)
            if show:
                with open(superimpose_name, 'r') as f:
                    resp.body = f.read()
                    resp.content_type = 'image/jpeg'

            if boxes:
                boxes = [a.tolist() for a in boxes]
                resp.body = json.dumps(boxes)
                resp.content_type = 'application/json'

        resp.status = falcon.HTTP_200

    def on_post(self, req, resp):
        category_to_look_for = req.get_param('catToLookFor')
        print('category to look for:' + str(category_to_look_for))
        ret = {"success": False}
        try:
            data = msgpack.loads(req.stream.read())
            img = data.get("image")
        except Exception:
            ret["error"] = traceback.format_exc()
        try:
            mnc_mask, mnc_box, im, im_name, orig_im, boxes, scalefactor, superimpose_name = mnc.mnc_pixlevel_detect(img)
#            mnc_mask, mnc_box = mnc.mnc_pixlevel_detect(img)
            if mnc_mask is not None:
                ret["success"] = True
                ret['mnc_output'] = {"mask": mnc_mask, "box": mnc_box, "superimposed_image": im,
                                     "image_name": im_name, "original_image": orig_im,
                                     "bounding_boxes": boxes, "scale_factor": scalefactor}

        except Exception:
            ret["error"] = traceback.format_exc()

        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200


api = falcon.API()
api.add_route('/', MNCResource())

