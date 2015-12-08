__author__ = 'yonatan'
'''
workers for the  mr8 testing
'''

import numpy as np

import constants
import fp_yuli_MR8

db = constants.db


# define a example function
# def add_new_field(doc, x):
#     image_url = doc["images"]["XLarge"]
#
#     image = Utils.get_cv2_img_array(image_url)
#     print "image high, width:", image.shape[0], image.shape[1]
#     if not Utils.is_valid_image(image):
#         logging.warning("image is None. url: {url}".format(url=image_url))
#         return
#
#
#     ms_response = []
#     for idx, val in enumerate(response):
#         ms_response = ms_response + fp_yuli_MR8.mean_std_pooling(val, 5)
#     # print (ms_response)
#     # print ("shape: " + str(ms_response[0].shape))
#     doc["mr8"] = ms_response
#     # try:
#     db.mr8_testing.insert_one(doc)
#     # except Exception as ex:
#     #     logging.warning("Exception caught while inserting element #" + str(x) + " to the collection".format(ex))
#     #     raw_input('boom!')
#     # return x


def mr8_4_demo(img, fc, mask):
    if len(fc) != 0:
        x0, y0, w, h = fc
        s_size = np.amin([w, h])
        print "s_size:", s_size
    # sample = gray_img[y0 + 3 * s_size:y0 + 4 * s_size, x0:x0 + s_size]
    else:
        s_size = np.asarray(0.1 * img.shape[0])
        print "s_size:", s_size
    trimmed_mask = fp_yuli_MR8.trim_mask(img, mask)
    response = fp_yuli_MR8.yuli_fp(trimmed_mask, s_size)
    print len(response)

    ms_response = []
    for idx, val in enumerate(response):
        ms_response = ms_response + fp_yuli_MR8.mean_std_pooling(val, 5)
    return ms_response
