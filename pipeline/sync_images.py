from urlparse import urlparse
import traceback
import ctypes
import time
import gevent
from gevent import Greenlet, monkey
monkey.patch_all(thread=False)
from bson import json_util
from rq import Queue
import falcon
from redis import Redis
from . import fast_results
from .. import constants
from .. import page_results
from .. import simple_pool

# Patch db for multiprocessing http://api.mongodb.com/python/current/faq.html#using-pymongo-with-multiprocessing
# fast_results.db = pymongo.MongoClient(host=os.getenv("MONGO_HOST", "mongodb1-instance-1"),
#                                       port=int(os.getenv("MONGO_PORT", "27017")),
#                                       connect=False).mydb

storm_q = Queue('start_pipeline', connection=constants.redis_conn)
r = Redis()
relevancy_q = Queue(connection=r)


class Images(object):

    def __init__(self):
        print "created Images"


    def check_existences(self, images, products):
        # db CHECK PARALLEL WITH gevent
        exists = {url: Greenlet.spawn(fast_results.check_if_exists, url, products) for url in images}
        gevent.joinall(exists.values())
        relevancy_dict = {}
        images_to_rel_check = []

        # DIVIDE RESULTS TO "HAS AN ANSWER" AND "WE DON'T KNOW THIS IMAGE"
        for url, green in exists.iteritems():
            if green.value is not None:
                relevancy_dict[url] = green.value
            else:
                images_to_rel_check.append(url)
        return relevancy_dict, images_to_rel_check


    def on_post(self, req, resp, image_url=None, pid=None):
        start = time.time()
        ret = {}
        method = 'nd'
        pid = pid or req.get_param("pid") or 'default'
        products = 'shopstyle_US' # page_results.get_collection_from_ip_and_pid(req.env['REMOTE_ADDR'], pid)
        # print "using products collection {0}".format(products)
        data = {"imageList": [image_url]} if image_url else json_util.loads(req.stream.read())
        images = data.get("imageList")
        # attempt to filter bad urls
        images = filter(lambda url:all(list(urlparse(url))[:3]), images)


        try:
            if isinstance(images, list):

                relevancy_dict, images_to_rel_check = self.check_existences(images, products)


                # RELEVANCY CHECK LIOR'S POOLING
                inputs = [(image_url, "dummy_page", products) for image_url in images_to_rel_check]
                outs = []
                for image in images_to_rel_check:
                    image_obj_result = fast_results.process_image(image_url, "dummy_page", products)
                    res = constants.db.images.insert_one(image_obj_result)
                    slimage = constants.db.images.find_one({"_id": res.inserted_id}, {"people.items.similar_results": 1})
                    outs.append(slimage)
                # outs = simple_pool.map(fast_results.check_if_relevant_and_enqueue, inputs, ouput_ctype=ctypes.c_char_p)
                relevancy_dict.update({images_to_rel_check[i]: outs[i] for i in xrange(len(images_to_rel_check))})

                ret["success"] = True
                ret["relevancy_dict"] = relevancy_dict

            else:
                ret["success"] = False
                ret["error"] = "Missing image list"

        except Exception as e:
            ret["error"] = traceback.format_exc()

        resp.data = json_util.dumps(ret)
        resp.content_type = 'application/json'
        # resp.status = falcon.HTTP_200
        print "ON_POST v2 took {0} seconds".format(time.time() - start)

        return ret


    def on_get(self, req, resp):
        ret = {}
        image_url = req.get_param("imageUrl")
        pid = req.get_param("pid") or 'default'
        if not image_url:
            raise falcon.HTTPMissingParam('imageUrl')

        products = page_results.get_collection_from_ip_and_pid(req.env['REMOTE_ADDR'], pid)
        print "on_get: products_collection {0}".format(products)
        
        start = time.time()
        ret = page_results.get_data_for_specific_image(image_url=image_url, products_collection=products)
        while not ret:
            if time.time()-start > 10:
                break
            time.sleep(0.25)
            ret = page_results.get_data_for_specific_image(image_url=image_url, products_collection=products)
        
        resp.status = falcon.HTTP_200 if ret else falcon.HTTP_400
        resp.data = json_util.dumps(ret)  # + "\n"
        resp.content_type = 'application/json'


# if __name__ == '__main__':
def run():
    image_url = 'http://fazz.co/src/img/demo/gettyimages-490421970.jpg'

    class Dummy(object):
        pass

    resp = Dummy()

    ret = Images().on_post(Dummy(), resp, image_url=image_url, pid='default')
    print ret, resp.data
