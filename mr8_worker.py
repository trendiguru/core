__author__ = 'yonatan'
'''
workers for the  mr8 testing
'''
import logging

import numpy as np
import cv2

import background_removal
import Utils
import constants
import fp_yuli_MR8

db = constants.db


# define a example function
def add_new_field(doc, x):
    image_url = doc["images"]["XLarge"]

    image = Utils.get_cv2_img_array(image_url)
    if not Utils.is_valid_image(image):
        logging.warning("image is None. url: {url}".format(url=image_url))
        return
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = background_removal.image_is_relevant(image)
    print faces
    if faces[0] is False:
        return
    fc = faces[1]
    print fc
    if len(fc) == 0:
        return
    x0, y0, w, h = fc[0]

    s_size = np.amin(np.asarray(w, h))

    sample = gray_img[y0 + 3 * s_size:y0 + 4 * s_size, x0:x0 + s_size]
    print sample.shape
    d = 40
    if s_size < d:
        return
    resized_sample = cv2.resize(sample, (d, d))
    response = fp_yuli_MR8.yuli_fp(resized_sample, d / 2)
    print len(response)

    ms_response = []
    for idx, val in enumerate(response):
        ms_response.append(fp_yuli_MR8.mean_std_pooling(val, 5))
    print (ms_response)
    print ("shape: " + ms_response[0].shape)

    try:
        doc["mr8"] = ms_response
        db.mr8_testing.insert(doc)
    except Exception as ex:
        logging.warning("Exception caught while inserting element #" + str(x) + " to the collection".format(ex))

    return x
