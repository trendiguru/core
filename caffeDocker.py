__author__ = 'yonatan'

import time

import constants

db = constants.db
relevant_caffe_labels = constants.caffe_relevant


def is_person_in_img(url):
    '''

    :param url: in string format!!!
    :return: True if relevant / False o.w
    '''
    tic = time.time()
    db.caffeQ.insert_one({"url": url})
    while db.caffeResults.find({"url": url}).count() == 0:
        time.sleep(0.25)
    toc = time.time()
    print "Total time of caffe: {0}".format(toc - tic)
    results = db.caffeResults.find_one({"url": url})
    catID = results["results"]
    intersection = [i for i in catID if i in relevant_caffe_labels]
    if len(intersection) == 0:
        return False
    db.caffeResults.delete_one({"url": url})
    return True
