from time import time
import numpy as np
import pymongo
from ..constants import db
import gevent
from gevent import Greenlet
import random

collection = 'amazon_US_Female'
category = 'dress'
annoy_top_results = list(np.random.randint(450000, size=1000))
annoy_new = ['skirt_{}'.format(i) for i in annoy_top_results]


def timeit(f, number, name='function'):
    global annoy_top_results, annoy_new
    rndstate = random.getstate()
    random.setstate(rndstate)
    annoy_top_results = list(np.random.randint(450000, size=1000))
    annoy_new = ['dress_{}'.format(ii) for ii in annoy_top_results]
    t1 = time()
    f(number)
    t2 = time()
    print '{} duration = {}'.format(name, (t2-t1))


def withH(b):
    entries = db[collection].find({"AnnoyIndex": {"$in": annoy_new}},
                                  {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1},
                                  cursor_type=pymongo.cursor.CursorType.EXHAUST)
    w = 0
    ec = entries.count() - 5
    for ee in entries:
        w += 1
        if w > ec:
            print w


def without(b):
    entries = db[collection].find({"AnnoyIndex": {"$in": annoy_new}},
                                  {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
    w = 0
    ec = entries.count() - 5
    for ee in entries:
        w += 1
        if w > ec:
            print w


def get_batchWH(batch):
    entries = db[collection].find({"AnnoyIndex": {"$in": batch}},
                                  {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1},
                                  cursor_type=pymongo.cursor.CursorType.EXHAUST)
    for ee in entries:
        # print ee['id']
        pass

    return {'d':'done'}


def get_batchWO(batch):
    entries = db[collection].find({"AnnoyIndex": {"$in": batch}},
                                  {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
    for ee in entries:
        # print ee['id']
        pass

    return {'d': 'done'}


def diviWH(b):
    bs = 1000/b
    queries = {q: Greenlet.spawn(get_batchWH, annoy_new[q*bs:(q+1)*bs]) for q in range(b)}
    gevent.joinall(queries.values())
    d = {k: v for k, v in queries.iteritems()}


def diviWO(b):
    bs = 1000 / b
    queries = {q: Greenlet.spawn(get_batchWO, annoy_new[q * bs:(q + 1) * bs]) for q in range(b)}
    gevent.joinall(queries.values())
    d = {k: v for k, v in queries.iteritems()}


def nofp(b):

    bs = 1000 / b
    for i in range(b):
        small_list = annoy_top_results[i * bs:(i + 1) * bs]
        entries = db[collection].find({"AnnoyIndex": {"$in": small_list}, 'categories': category},
                                      {"id": 1, "images.XLarge": 1, "clickUrl": 1})
        for ee in entries:
            # print ee['id']
            pass


def annoy_new_w(b):

    entries = db[collection].find({"AnnoyIndex_new": {"$in": annoy_new}},
                                  {"id": 1, "images.XLarge": 1, "clickUrl": 1, "fingerprint":1},
                                  cursor_type=pymongo.cursor.CursorType.EXHAUST)
    w = 0
    ec = entries.count() - 5
    for ee in entries:
        w += 1
        if w > ec:
            print w


def annoy_new_wo(b):

    entries = db[collection].find({"AnnoyIndex_new": {"$in": annoy_new}},
                                  {"id": 1, "images.XLarge": 1, "clickUrl": 1, "fingerprint":1})
    w = 0
    ec = entries.count() - 5
    for ee in entries:
        w += 1
        if w > ec:
            print w

# timeit(withH, number=1, name='with EXHAUST')
# timeit(without, number=1, name='without EXHAUST')
# timeit(withH, number=10, name='with EXHAUST /10')
# timeit(without, number=10, name='without EXHAUST /10')
# timeit(withH, number=20, name='with EXHAUST /20')
# timeit(without, number=20, name='without EXHAUST /20')
# timeit(withH, number=50, name='with EXHAUST /50')
# timeit(without, number=50, name='without EXHAUST /50')
# timeit(withH, number=100, name='with EXHAUST /100')
# timeit(without, number=100, name='without EXHAUST /100')
#
timeit(diviWH, number=10, name='divi with EXHAUST /10')
timeit(diviWO, number=10, name='divi without EXHAUST /10')
timeit(diviWH, number=50, name='divi with EXHAUST /50')
timeit(diviWO, number=50, name='divi without EXHAUST /50')
timeit(withH, number=1, name='with EXHAUST /1000')
timeit(without, number=1, name='without EXHAUST /1000')
# timeit(diviWH, number=1000, name='divi with EXHAUST /1000')
# timeit(diviWO, number=1000, name='divi without EXHAUST /1000')
# timeit(nofp, number=1, name='nofp')
# timeit(nofp, number=10, name='nofp 10')
# timeit(annoy_new_w, number=1, name='new with')
# timeit(annoy_new_wo, number=1, name='new without')

