__author__ = 'yonatan'

import time
import collections

import cv2

import Utils
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
    if method == "url":
        if path[-4:] == 'webp':
            src = Utils.get_cv2_img_array(path)
            try:
                print src.shape
            except:
                print "src is None!"
            src = cv2.resize(src, (227, 227))
            src = src.astype(float) / 255
            # src = base64.b64encode(src)
            src = src.tolist()
            method = 'img'
        else:
            src = path
    elif method == "img":
        src = Utils.get_cv2_img_array(path)
        src = cv2.resize(src, (227, 227))
        src = src.astype(float) / 255
        src = src.tolist()
        # src = base64.b64encode(src)
    else:
        raise IOError("bad input was inserted to caffe!")
    tic = time.time()
    ans = db.caffeQ.insert_one({"method": method, "src": src, "k": k})
    id = ans.inserted_id
    itr = 0
    while db.caffeResults.find({"id": str(id)}).count() == 0:
        itr += 1
        time.sleep(0.25)
        if itr > 20:
            break

    toc = time.time()
    print "Total time of caffe: {0}".format(toc - tic)
    try:
        results = db.caffeResults.find_one({"id": str(id)})
        catID = results["results"]
        intersection = [i for i in catID if i in relevant_caffe_labels]
        db.caffeResults.delete_one({"id": str(id)})

        if len(intersection) == 0:
            return CaffeAnswer(False, catID)
        else:
            return CaffeAnswer(True, catID)
    except:
        db.caffeResults.delete_many({})
        return CaffeAnswer(False, [])
