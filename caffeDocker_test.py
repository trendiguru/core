__author__ = 'yonatan'

import time
import collections

import cv2
import numpy as np

import constants

db = constants.db
relevant_caffe_labels = constants.caffeRelevantLabels


def is_person_in_img(method, path, k=10):
    '''

    :param type: what is the input type -path or url
            src: the exctual path/urlin string format!!!
            k: number of results
    :return: namedTuple CaffeAnswer with 2 fields: binary is_person and list of detected categories
    '''

    CaffeAnswer = collections.namedtuple('caffe_answer', 'is_person categories')
    tic = time.time()
    if method == "url":
        src = path
    elif method == "img":
        src = np.array(cv2.imread(path))
        toc = time.time()
        print (toc - tic)
        src = src.astype(float) / 255
        toc = time.time()
        print (toc - tic)
        src = src.tolist()
        toc = time.time()
        print (toc - tic)
    else:
        raise IOError("bad input was inserted to caffe!")

    id = db.caffeQ.insert_one({"method": method, "src": src, "k": k})
    toc = time.time()
    print (toc - tic)
    while db.caffeResults.find_one({"_id": str(id)}).count() == 0:
        time.sleep(0.25)
        if db.caffeQ.find({"_id": str(id)}).count() == 0:
            break
    toc = time.time()
    print "Total time of caffe: {0}".format(toc - tic)
    try:
        results = db.caffeResults.find_one({"_id": str(id)})
        catID = results["results"]
        intersection = [i for i in catID if i in relevant_caffe_labels]
        db.caffeResults.delete_one({"_id": str(id)})

        if len(intersection) == 0:
            return CaffeAnswer(False, catID)
        else:
            return CaffeAnswer(True, catID)
    except:
        return CaffeAnswer(False, [])
