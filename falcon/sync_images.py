# import os
import traceback
import time
from functools import partial
import gevent
from gevent import Greenlet, monkey
monkey.patch_all()
import socket
from bson import json_util
from rq import Queue
import falcon

from . import fast_results
from .. import constants
from .. import page_results

# Patch db for multiprocessing http://api.mongodb.com/python/current/faq.html#using-pymongo-with-multiprocessing
# fast_results.db = pymongo.MongoClient(host=os.getenv("MONGO_HOST", "mongodb1-instance-1"),
#                                       port=int(os.getenv("MONGO_PORT", "27017")),
#                                       connect=False).mydb

storm_q = Queue('star_pipeline', connection=constants.redis_conn)


class Images(object):
    
    def __init__(self,  process_pool):
        self.process_pool = process_pool
        print "created Images"

    # def __init__(self):
    #     print "created Images"

    def on_post(self, req, resp):
        start = time.time()
        ret = {"success": False}
        try:
            data = json_util.loads(req.stream.read())
            page_url = data.get("pageUrl")
            images = data.get("imageList")
            print "POST: after data gets: {0}".format(time.time()-start)
            if type(images) is list and page_url is not None:
                # db CHECK PARALLEL WITH gevent
                exists = {url: Greenlet.spawn(fast_results.check_if_exists, url) for url in images}
                gevent.joinall(exists.values())
                relevancy_dict = {}
                images_to_rel_check = []
                print "POST: after db_checks: {0}".format(time.time()-start)

                # DIVIDE RESULTS TO "HAS AN ANSWER" AND "WE DON'T KNOW THIS IMAGE"
                for url, green in exists.iteritems():
                    if green.value is not None:
                        relevancy_dict[url] = green.value
                    else:
                        images_to_rel_check.append(url)
                print "POST: after deviation: {0}".format(time.time()-start)
                print "POST: rel_dict: {0}\n images to check: {1}".format(relevancy_dict, images_to_rel_check)

                # RELEVANCY CHECK POOLING
                reload(socket)
                check_relevancy_partial = partial(fast_results.check_if_relevant_and_enqueue, page_url=page_url)
                print "after partial: {0}".format(time.time()-start)
                fast_route_results = self.process_pool.map(check_relevancy_partial, images_to_rel_check)
                print "after process_pool_mapping: {0}".format(time.time()-start)
                relevancy_dict.update({images[i]: fast_route_results[i] for i in xrange(len(images_to_rel_check))})

                # RELEVANCY CHECK GEVENT
                # relevant = {url: Greenlet.spawn(fast_results.check_if_relevant_and_enqueue, url, page_url)
                #             for url in images_to_rel_check}
                # print "POST: before join_all: {0}".format(time.time()-start)
                # gevent.joinall(relevant.values())
                # print "POST: after join_all: {0}".format(time.time()-start)
                # relevancy_dict.update({url: green.value for url, green in relevant.iteritems()})
                print "POST: after dictionary update: {0}".format(time.time()-start)

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
        if not image_url:
          raise falcon.HTTPMissingParam('imageUrl')

        if 'fashionseoul' in image_url:
            products = "GangnamStyle"
        else:
            products = "ShopStyle"
        
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
