from .. import constants
import pymongo
import subprocess
import time
import random
import datetime
import yunomi
from bson import json_util
import requests
from rq import Queue

db = constants.db
prefix = 'https://storage.googleapis.com/tg-training/'
images_q = Queue('start_synced_pipeline', connection=constants.redis_conn)
post_q = Queue('post_it', connection=constants.redis_conn)
relevancy_relation = 0.1
API_URL = 'http://api.trendi.guru/images'


def overflow_test(batch_size=10):
    # create list of ir/relevant images urls from db.irrelevant_/images &
    relevant_urls = get_first_x_images_from_collection(2000, 'images') + \
                    get_urls_from_gs("gs://tg-training/doorman/relevant")
    print "done collecting relevant-images, total {0}".format(len(relevant_urls))
    irrelevant_urls = get_urls_from_gs("gs://tg-training/doorman/irrelevant/irrelevant_images_for_doorman")
    print "done collecting irrelevant-images, total {0}".format(len(irrelevant_urls))

    rel_num = int(relevancy_relation*batch_size)
    irrel_num = batch_size-rel_num
    queue_file = open('/home/nadav/test_queue_log.txt', 'w')
    # MaiN LooP
    rel_cnt = yunomi.Meter()
    irrel_cnt = yunomi.Meter()
    inter = time.time()
    while len(relevant_urls) or len(irrelevant_urls):
        # create batch by the relations given
        urls_batch = []
        if len(relevant_urls):
            urls_batch = [relevant_urls.pop(random.randint(0, len(relevant_urls))) for i in xrange(0, rel_num)]
        if len(irrelevant_urls):
            urls_batch += [irrelevant_urls.pop() for i in xrange(0, irrel_num)]

        # simulate reasonable POST requests tempo to https://api.trendi.guru/images
        data = {"pageUrl": "overflow_test", "imageList": urls_batch}
        post_q.enqueue_call(func=post_it, args=(data,))
        time.sleep(1)

        # get a few measurements and print to log file:
        rel_cnt.mark(images_q.count)
        irrel_cnt.mark(db.irrelevant_images.count())

        if time.time()-inter > 10:
            inter = time.time()
            queue_file.write("{0}: total images on queue: {1}\n"
                             "".format(str(datetime.datetime.now()), rel_cnt.get_count()))
    queue_file.close()


def get_first_x_images_from_collection(x, collection):
    curs = db[collection].find({}, {'image_urls': 1}).sort('_id', pymongo.ASCENDING).limit(x)
    return [doc['image_urls'][0] for doc in curs]


def get_urls_from_gs(storage_lib):
    p = subprocess.Popen(["gsutil", "ls", storage_lib], stdout=subprocess.PIPE)
    output, err = p.communicate()
    output = [prefix+url[17:] for url in output.split('\n')]
    return output


def post_it(data):
    requests.post(API_URL, data=json_util.dumps(data))
