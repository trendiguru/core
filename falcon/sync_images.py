import traceback
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
    
    # def __init__(self,  process_pool):
    #     self.process_pool = process_pool
    #     print "created Images"

    def __init__(self):
        print "created Images"

    def on_post(self, req, resp):
        start = time.time()
        ret = {"success": False}
        method = req.get_param("method") or 'nd'
        pid = req.get_param("pid") or 'default'
        products = page_results.get_collection_from_ip_and_pid(req.env['REMOTE_ADDR'], pid)
        print "using products collection {0}".format(products)
        data = json_util.loads(req.stream.read())
        page_url = data.get("pageUrl")
        images = data.get("imageList")
        try:
            if type(images) is list and page_url is not None:
                if method == 'pd':
                    relevancy_dict = {url: page_results.handle_post(url, page_url, products, 'pd') for url in images}
                    ret["success"] = True
                    ret["relevancy_dict"] = relevancy_dict
                else:
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

                    # RELEVANCY CHECK LIOR'S POOLING
                    # TODO - add products collection to inputs
                    inputs = [(image_url, page_url, products) for image_url in images_to_rel_check]
                    outs = simple_pool.map(fast_results.check_if_relevant_and_enqueue, inputs)
                    relevancy_dict.update({images_to_rel_check[i]: outs[i] for i in xrange(len(images_to_rel_check))})

                    ret["success"] = True
                    ret["relevancy_dict"] = relevancy_dict

            else:
                ret["success"] = False
                ret["error"] = "Missing image list and/or page url"

        except Exception as e:
            ret["error"] = traceback.format_exc()

        resp.data = json_util.dumps(ret)
        resp.content_type = 'application/json'
        resp.status = falcon.HTTP_200
        print "ON_POST took {0} seconds".format(time.time()-start)

    def on_get(self, req, resp):
        ret = {}
        image_url = req.get_param("imageUrl")
        pid = req.get_param("pid") or 'default'
        if not image_url:
            raise falcon.HTTPMissingParam('imageUrl')

        products = page_results.get_collection_from_ip_and_pid(req.env['REMOTE_ADDR'], pid)
        
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
