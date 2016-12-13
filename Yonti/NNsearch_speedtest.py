import timeit
import numpy as np
import pymongo
from ..constants import db


collection = 'amazon_US_Female'
annoy_top_results = np.random.randint(10000, size=100)
category = 'dress'


def test1():
    entries = db[collection].find({"AnnoyIndex": {"$in": annoy_top_results}, 'categories': category},
                                  {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1},
                                  cursor_type=pymongo.cursor.CursorType.EXHAUST)

    for i in entries:
        print i

timeit.timeit("""
entries = db[collection].find({"AnnoyIndex": {"$in": annoy_top_results}, 'categories': category},
                                  {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1},
                                  cursor_type=pymongo.cursor.CursorType.EXHAUST)

    for i in entries:
        print i
""", number=1)

