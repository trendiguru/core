__author__ = 'yonatan'

import time

import skimage
import numpy as np

import constants

db = constants.db
relevant_caffe_labels = constants.caffeRelevantLabels


def is_person_in_img(method, src):
    '''

    :param type: what is the input type -path or url
            src: the exctual path/urlin string format!!!
    :return: True if relevant / False o.w
    '''
    tic = time.time()
    if method == "url":
        db.caffeQ.insert_one({"method": method, "src": src})
    else:
        src = load_image(src)
        src = src.tolist()
        db.caffeQ.insert_one({"method": method, "src": src})
    while db.caffeResults.find({"src": src}).count() == 0:
        time.sleep(0.25)
    toc = time.time()
    print "Total time of caffe: {0}".format(toc - tic)
    results = db.caffeResults.find_one({"src": src})
    catID = results["results"]
    intersection = [i for i in catID if i in relevant_caffe_labels]
    db.caffeResults.delete_one({"src": src})
    if len(intersection) == 0:
        return False
    return True


def load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.

    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).

    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = skimage.img_as_float(skimage.io.imread(filename)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img
