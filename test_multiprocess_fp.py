__author__ = 'liorsabag'

from multiprocessing import Pool, Lock, Value, current_process
import pymongo
import time
import random


class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value


print "Hello World"

TOTAL = 0
CURRENT = Counter()


def fake_fp(product):
    global TOTAL, CURRENT
    CURRENT.increment()
    print "{0} working on product {1}. {2} of {3}...\n".format(str(current_process().pid), product["id"], CURRENT.value, TOTAL)
    # print str(current_process().pid) + " working on: " + str(product["id"]) + "\n"
    start_time = time.time()
    while time.time() < start_time + (random.random() * 5):
        start_time * start_time
        # print "done: " + str(product["id"])


db = pymongo.MongoClient().mydb
mini_skirt_cursor = db.products.find({"categories": {"$elemMatch": {"id": "mini-skirts"}}},
                                     {"id": 1, "images": 1}).batch_size(12000)

TOTAL = mini_skirt_cursor.count()
# first_ten = [mini_skirt_cursor[i] for i in range(0, 20)]


p = Pool(6)
p.map(fake_fp, mini_skirt_cursor)
