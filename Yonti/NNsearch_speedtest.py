from time import time
import numpy as np
import pymongo
from ..constants import db


collection = 'amazon_US_Female'
annoy_top_results = list(np.random.randint(480000, size=1000))
category = 'dress'


def test1():
    entries = db[collection].find({"AnnoyIndex": {"$in": annoy_top_results}, 'categories': category},
                                  {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1},
                                  cursor_type=pymongo.cursor.CursorType.EXHAUST)

    for i,x in enumerate(entries):
        print i


def timeit(f):
    t1 = time()
    f()
    t2 = time()
    print (t2-t1)


timeit(test1)
