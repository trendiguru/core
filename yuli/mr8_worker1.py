__author__ = 'yonatan'
'''
workers for the  mr8 testing
'''
import logging

import numpy as np
import cv2

import paperdoll.paperdoll_parse_enqueue
import background_removal
import Utils
import constants
import fp_yuli_MR8

db = constants.db


# define a example function
def add_new_field(doc, x):
    image_url = doc["images"]["XLarge"]

    image = Utils.get_cv2_img_array(image_url)
    print "image high, width:", image.shape[0], image.shape[1]
    if not Utils.is_valid_image(image):
        logging.warning("image is None. url: {url}".format(url=image_url))
        return


    ms_response = []
    for idx, val in enumerate(response):
        ms_response = ms_response + fp_yuli_MR8.mean_std_pooling(val, 5)
    # print (ms_response)
    # print ("shape: " + str(ms_response[0].shape))
    doc["mr8"] = ms_response
    # try:
    db.mr8_testing.insert_one(doc)
    # except Exception as ex:
    #     logging.warning("Exception caught while inserting element #" + str(x) + " to the collection".format(ex))
    #     raw_input('boom!')
    # return x


def mr8_4_demo(img, fc = None):

    mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(img, async=False).result[:3]
    final_mask = paperdolls.after_pd_conclusions(mask, labels, person['face'])
    for num in np.unique(final_mask):
	category = list(labels.keys())[list(labels.values()).index(num)]
	if category in constants.paperdoll_shopstyle_women.keys():
		item_mask = 255 * np.array(final_mask == num, dtype=np.uint8)

    faces = background_removal.image_is_relevant(image)

    if faces[0] is True and len(faces[1]) != 0:
        x0, y0, w, h = faces[1][0]
        s_size = np.amin(np.asarray(w, h))
        print "s_size:", s_size
	#sample = gray_img[y0 + 3 * s_size:y0 + 4 * s_size, x0:x0 + s_size]
    else:
        s_size = np.asarray(0.1*image.shape[0])
        print "s_size:", s_size
         
    trimmed_mask = fp_yuli_MR8.trim_mask(image, mask) 
    response = fp_yuli_MR8.yuli_fp(trimmed_mask, s_size)
    print len(response)


    ms_response = []
    for idx, val in enumerate(response):
        ms_response = ms_response + fp_yuli_MR8.mean_std_pooling(val, 5)
    return ms_response
