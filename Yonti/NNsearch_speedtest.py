from time import time
import numpy as np
import pymongo
from ..constants import db


collection = 'amazon_US_Female'
annoy_top_results = list(np.random.randint(480000, size=1000))
category = 'dress'


def timeit(f, number, name='function'):
    t1 = time()
    for i in number:
        f()
    t2 = time()
    print '{} duration = {}'.format(name,(t2-t1)/float(number))


def test1():
    """with exhaust"""
    entries = db[collection].find({"AnnoyIndex": {"$in": annoy_top_results}, 'categories': category},
                                  {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1},
                                  cursor_type=pymongo.cursor.CursorType.EXHAUST)
    for i in entries:
        pass


def test2():
    """without exhaust"""
    entries = db[collection].find({"AnnoyIndex": {"$in": annoy_top_results}, 'categories': category},
                                  {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
    for i in entries:
        pass


def test3():
    """with exhaust - divided to 10 subgroups"""

    for i in range(10):
        small_list = annoy_top_results[i*100:(i+1)*100]
        len(annoy_top_results)
        entries = db[collection].find({"AnnoyIndex": {"$in": small_list}, 'categories': category},
                                      {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1},
                                      cursor_type=pymongo.cursor.CursorType.EXHAUST)
        for i in entries:
            pass


def test4():
    """without exhaust - divided to 10 subgroups"""

    for i in range(10):
        small_list = annoy_top_results[i * 100:(i + 1) * 100]
        len(annoy_top_results)
        entries = db[collection].find({"AnnoyIndex": {"$in": small_list}, 'categories': category},
                                      {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
        for i in entries:
            pass


def test5():
    """with exhaust - divided to 20 subgroups"""

    for i in range(20):
        small_list = annoy_top_results[i * 50:(i + 1) * 50]
        len(annoy_top_results)
        entries = db[collection].find({"AnnoyIndex": {"$in": small_list}, 'categories': category},
                                      {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1},
                                      cursor_type=pymongo.cursor.CursorType.EXHAUST)
        for i in entries:
            pass


def test6():
    """without exhaust - divided to 20 subgroups"""

    for i in range(20):
        small_list = annoy_top_results[i * 50:(i + 1) * 50]
        len(annoy_top_results)
        entries = db[collection].find({"AnnoyIndex": {"$in": small_list}, 'categories': category},
                                      {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
        for i in entries:
            pass


def test7():
    """with exhaust - divided to 50 subgroups"""

    for i in range(50):
        small_list = annoy_top_results[i * 20:(i + 1) * 20]
        len(annoy_top_results)
        entries = db[collection].find({"AnnoyIndex": {"$in": small_list}, 'categories': category},
                                      {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1},
                                      cursor_type=pymongo.cursor.CursorType.EXHAUST)
        for i in entries:
            pass


def test8():
    """without exhaust - divided to 50 subgroups"""

    for i in range(50):
        small_list = annoy_top_results[i * 20:(i + 1) * 20]
        len(annoy_top_results)
        entries = db[collection].find({"AnnoyIndex": {"$in": small_list}, 'categories': category},
                                      {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
        for i in entries:
            pass


def test9():
    """with exhaust - divided to 100 subgroups"""

    for i in range(100):
        small_list = annoy_top_results[i * 10:(i + 1) * 10]
        len(annoy_top_results)
        entries = db[collection].find({"AnnoyIndex": {"$in": small_list}, 'categories': category},
                                      {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1},
                                      cursor_type=pymongo.cursor.CursorType.EXHAUST)
        for i in entries:
            pass


def test10():
    """without exhaust - divided to 100 subgroups"""

    for i in range(100):
        small_list = annoy_top_results[i * 10:(i + 1) * 10]
        len(annoy_top_results)
        entries = db[collection].find({"AnnoyIndex": {"$in": small_list}, 'categories': category},
                                      {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
        for i in entries:
            pass

n=1
timeit(test1, number=n, name='with EXHAUST')
timeit(test2, number=n, name='without EXHAUST')
timeit(test3, number=n, name='with EXHAUST /10')
timeit(test4, number=n, name='without EXHAUST /10')
timeit(test5, number=n, name='with EXHAUST /30')
timeit(test6, number=n, name='without EXHAUST /20')
timeit(test7, number=n, name='with EXHAUST /50')
timeit(test8, number=n, name='without EXHAUST /50')
timeit(test9, number=n, name='with EXHAUST /100')
timeit(test10, number=n, name='without EXHAUST /100')
