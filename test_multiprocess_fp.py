__author__ = 'liorsabag'

import multiprocessing
import pymongo
import time
import random
from Utils import ThreadSafeCounter


TOTAL = 0
CURRENT = ThreadSafeCounter()


def fake_fp(product):
    global CURRENT
    CURRENT.increment()
    print "{0} working on product {1}. {2} of {3}...\n".format(str(multiprocessing.current_process().pid), product["id"], CURRENT.value, TOTAL)
    # print str(current_process().pid) + " working on: " + str(product["id"]) + "\n"
    start_time = time.time()
    while time.time() < start_time + (random.random() * 5):
        start_time * start_time
        # print "done: " + str(product["id"])


class FpWorker(multiprocessing.Process):
    def __init__(self):



if __name__ == "__main__":

    print "Hello World"

    db = pymongo.MongoClient().mydb
    mini_skirt_cursor = db.products.find({"categories": {"$elemMatch": {"id": "mini-skirts"}}},
                                         {"id": 1, "images": 1}).batch_size(12000)

    TOTAL = mini_skirt_cursor.count()
    # first_ten = [mini_skirt_cursor[i] for i in range(0, 20)]

