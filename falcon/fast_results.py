import gevent
from gevent import Greenlet, monkey

import os
import bson
import time
from rq import push_connection, Queue

# ours
from .. import Utils
from .. import constants
from .. import background_removal
from ..page_results import genderize

db = constants.db
push_connection(constants.redis_conn)
start_q = Queue('start_synced_pipeline', connection=constants.redis_conn)


def check_if_exists(image_url):

    # Temporarily remove whitelist for Recruit Test -- LS 22/06/2016.
    # domain = tldextract.extract(page_url).registered_domain
    # if not db.whitelist.find_one({'domain': domain}):
    #     return False
    start = time.time()
    if image_url[:4] == "data":
        return False

    def check_db(collection):
        if db[collection].find_one({'image_urls': image_url}, {'_id': 1}):
            return True
        else:
            return False

    greens = {collection: Greenlet.spawn(check_db, collection) for collection in ['images', 'irrelevant_images', 'iip']}
    gevent.joinall(greens.values())
    if greens['images'].value or greens['iip'].value:
        return True
    elif greens['irrelevant_images'].value:
        return False
    print "after db checks: {0}".format(time.time()-start)
    return None


def check_if_relevant_and_enqueue(image_url, page_url):
    pid = os.getpid()
    start = time.time()
    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        return False

    small_img, rr = background_removal.standard_resize(image, 600)
    print "Thread {1}: after image_DL: {0}".format(time.time()-start, pid)
    relevance = background_removal.image_is_relevant(small_img, use_caffe=False, image_url=image_url)
    print "Thread {1}: after image is relevant: {0}".format(time.time()-start, pid)
    if relevance.is_relevant:
        image_obj = {'people': [{'person_id': str(bson.ObjectId()), 'face': face.tolist()} for face in relevance.faces],
                     'image_urls': image_url, 'page_url': page_url}
        db.iip.insert_one(image_obj)
        print "Thread {1}: after db.iip insert checks: {0}".format(time.time()-start, pid)
        start_q.enqueue_call(func="", args=(page_url, image_url, 'nd'), ttl=2000, result_ttl=2000, timeout=2000)
        print "Thread {1}: total fast_results: {0}".format(time.time()-start, pid)
        return True
    else:
        print "Thread {1}: total fast_results: {0}".format(time.time()-start, pid)
        return False
