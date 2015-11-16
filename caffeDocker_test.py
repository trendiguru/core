__author__ = 'yonatan'

import time

import cv2
import numpy as np

import constants

db = constants.db
relevant_caffe_labels = constants.caffeRelevantLabels


def is_person_in_img(method, path, k):
    '''

    :param type: what is the input type -path or url
            src: the exctual path/urlin string format!!!
            k: numner of results
    :return: True if relevant / False o.w
    '''
    tic = time.time()
    if method == "url":
        src = path
    elif method == "img":
        src = np.array(cv2.imread(path))
        src = src.astype(float) / 255
        src = src.tolist()
    else:
        print "bad input"
        return ("bad input", [])

    db.caffeQ.insert_one({"method": method, "src": src, "k": k})
    while db.caffeResults.find({"src": src}).count() == 0:
        time.sleep(0.25)
    toc = time.time()
    print "Total time of caffe: {0}".format(toc - tic)
    results = db.caffeResults.find_one({"src": src})
    catID = results["results"]
    intersection = [i for i in catID if i in relevant_caffe_labels]
    db.caffeResults.delete_one({"src": src})

    if len(intersection) == 0:
        retTuple = (False, catID)
    else:
        retTuple = (True, catID)

    return retTuple
