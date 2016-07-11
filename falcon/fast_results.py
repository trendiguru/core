import datetime
import tldextract
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


def fast_route(image_url, page_url):

    # Temporarily remove whitelist for Recruit Test -- LS 22/06/2016.
    # domain = tldextract.extract(page_url).registered_domain
    # if not db.whitelist.find_one({'domain': domain}):
    #     return False
    start = time.time()
    if image_url[:4] == "data":
        return False

    if db.irrelevant_images.find_one({'image_urls': image_url}, {'_id': 1}):
        return False

    if db.iip.find_one({'image_url': image_url}, {'_id': 1}):
        return True

    if db.images.find_one({'image_urls': image_url}, {'_id': 1}):
        return True
    print "after db checks: {0}".format(time.time()-start)
    # Check Relevancy
    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        return
    print "after image_DL: {0}".format(time.time()-start)
    relevance = background_removal.image_is_relevant(image, use_caffe=False, image_url=image_url)
    print "after image is relevant: {0}".format(time.time()-start)
    if relevance.is_relevant:
        image_obj = {'people': [{'person_id': str(bson.ObjectId()), 'face': face.tolist()} for face in relevance.faces],
                     # 'gender': genderize(image, face.tolist())['gender']} for face in relevance.faces],
                     'image_url': image_url, 'page_url': page_url}
        db.iip.insert_one({'image_url': image_url, 'insert_time': datetime.datetime.utcnow()})
        db.genderator.insert_one(image_obj)
        start_q.enqueue_call(func="", args=(page_url, image_url, 'nd'), ttl=2000, result_ttl=2000, timeout=2000)
        print "total fast_results: {0}".format(time.time()-start)
        return True
    else:
        print "total fast_results: {0}".format(time.time()-start)
        return False
