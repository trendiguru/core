from bson import json_util
from rq import Queue
import falcon
import time

from . import fast_results
from .. import constants
from .. import page_results

storm_q = Queue('star_pipeline', connection=constants.redis_conn)


class Images(object):

    def on_post(self, req, resp):
        ret = {"success": False}
        try:
            data = json_util.loads(req.stream.read())
            page_url = data.get("pageUrl")
            images = data.get("imageList")
            if type(images) is list and page_url is not None:
                relevancy_dict = {url: fast_results.fast_route(url, page_url) for url in images}
                ret["success"] = True
                ret["relevancy_dict"] = relevancy_dict
            else:
                ret["success"] = False
                ret["error"] = "Missing image list and/or page url"

        except Exception as e:
            ret["error"] = str(e)

        resp.data = json_util.dumps(ret)
        resp.content_type = 'application/json'
        resp.status = falcon.HTTP_200

    def on_get(self, req, resp):
        ret = {}
        image_url = req.get_param("imageUrl")

        if 'fashionseoul' in image_url:
            products = "GangnamStyle"
        else:
            products = "ShopStyle"

        if image_url:
            start = time.time()
            ret = page_results.get_data_for_specific_image(image_url=image_url, products_collection=products)
            while not ret:
                if time.time()-start > 10:
                    break
                time.sleep(0.3)
                ret = page_results.get_data_for_specific_image(image_url=image_url, products_collection=products)
            resp.status = falcon.HTTP_200*bool(ret) or falcon.HTTP_400
        else:
            resp.status = falcon.HTTP_400

        resp.data = json_util.dumps(ret) + "\n"
        resp.content_type = 'application/json'
