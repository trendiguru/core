from .. import constants
import pymongo
import subprocess
import time
import datetime
import yunomi
import requests
from rq import Queue
import gevent
from gevent import Greenlet, monkey
monkey.patch_all(thread=False)


db = constants.db
prefix = 'https://storage.googleapis.com/'
images_q = Queue('start_synced_pipeline', connection=constants.redis_conn)
post_q = Queue('post_it', connection=constants.redis_conn)
relevancy_relation = 0.1
API_URL = 'https://api.trendi.guru/images'


def overflow_test(batch_size):
    # create list of ir/relevant images urls from db.irrelevant_/images &
    relevant_urls = get_first_x_images_from_collection(2000, 'images') + \
                    get_urls_from_gs("gs://tg-training/doorman/relevant")
    print "done collecting relevant-images, total {0}".format(len(relevant_urls))
    irrelevant_urls = get_urls_from_gs("gs://tg-training/doorman/irrelevant/irrelevant_images_for_doorman")
    print "done collecting irrelevant-images, total {0}".format(len(irrelevant_urls))

    rel_num = int(relevancy_relation*batch_size)
    irrel_num = batch_size-rel_num
    queue_file = open('/home/nadav/test_queue_log.txt', 'w')
    requests_file = open('/home/nadav/test_req_log.txt', 'w')
    # MaiN LooP
    while len(relevant_urls) or len(irrelevant_urls):
        # create batch by the relations given
        urls_batch = []
        if len(relevant_urls):
            urls_batch = [relevant_urls.pop() for i in xrange(0, rel_num)]
        if len(irrelevant_urls):
            urls_batch += [irrelevant_urls.pop() for i in xrange(0, irrel_num)]

        # simulate reasonable POST requests tempo to https://api.trendi.guru/images
        data = {"pageUrl": "overflow_test", "imageList": urls_batch}
        post_q.enqueue_call(func='post_it', args=[data, requests_file])
        time.sleep(1)

        # get a few measurements and print to log file:
        queue_file.write("{0}: ".format(str(datetime.datetime.now()), ))
    queue_file.close()
    requests_file.close()


def get_first_x_images_from_collection(x, collection):
    curs = db[collection].find({}, {'image_urls': 1}).sort('_id', pymongo.ASCENDING).limit(x)
    return [doc['image_urls'][0] for doc in curs]


def get_urls_from_gs(storage_lib):
    p = subprocess.Popen(["gsutil", "ls", storage_lib], stdout=subprocess.PIPE)
    output, err = p.communicate()
    output = [prefix+url[17:] for url in output.split('\n')]
    return output


def post_it(data, log_file):
    start = time.time()
    requests.post(API_URL, data=data)
    log_file.write("{0}: POST duration was {1} seconds\n".format(str(datetime.datetime.now()), time.time()-start))