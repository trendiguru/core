__author__ = 'yonatan'

import time

import constants

db = constants.db
relevantCaffeLabels = constants.caffeRelevantLabels


def isImgRelevant(url):
    '''

    :param url: in string format!!!
    :return: True if relevant / False o.w
    '''
    tic = time.time()
    db.caffeQ.insert_one({"url": url})

    while (db.caffeResults.find({"url": url}).count() == 0):
        time.sleep(0.25)
    toc = time.time()
    print toc - tic
    results = db.caffeResults.find_one({"url": url})

    catID = results["results"]
    intersection = [i for i in catID if i in relevantCaffeLabels]
    if intersection is None:
        return False
    results = db.caffeResults.delete_one({"url": url})
    return True
