__author__ = 'yonatan'
'''
workers for the  mr8 testing
'''

import logging

import numpy as np

from ... import constants
from ... import new_finger_print
from ... import Utils
from ... import background_removal

db = constants.db


def person_isolation(image, face):
    x, y, w, h = face
    print (face)
    raw_input("boom!")
    image_copy = np.zeros(image.shape, dtype=np.uint8)
    x_back = np.max([x - 1.5 * w, 0])
    x_ahead = np.min([x + 2.5 * w, image.shape[1] - 2])
    image_copy[:, int(x_back):int(x_ahead), :] = image[:, int(x_back):int(x_ahead), :]
    return image_copy


# define a example function
def add_new_field(doc, x):
    image_url = doc["images"]["XLarge"]

    image = Utils.get_cv2_img_array(image_url)
    # image is BGR?
    print "image height, width:", image.shape[0], image.shape[1]
    if not Utils.is_valid_image(image):
        logging.warning("image is None. url: {url}".format(url=image_url))
        return
    try:
        small_image, resize_ratio = background_removal.standard_resize(image, 400)
        mask = background_removal.get_fg_mask(small_image)
    except:
        print("mask error")
        return
    # relevance = background_removal.image_is_relevant(image, True, image_url)
    # if relevance.is_relevant:
    #     if len(relevance.faces):
    #         if not isinstance(relevance.faces, list):
    #             relevant_faces = relevance.faces.tolist()
    #         else:
    #             relevant_faces = relevance.faces
    #         for face in relevant_faces:
    #             image_copy = person_isolation(image, face)
    #             mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(image_copy, async=False).result[:3]
    #             final_mask = after_pd_conclusions(mask, labels, face)
    #             break
    #     else:
    #         # no faces, only general positive human detection
    #         mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(image, async=False).result[:3]
    #         final_mask = after_pd_conclusions(mask, labels)
    #
    #     item_mask = 255 * np.array(final_mask == final_mask, dtype=np.uint8)
    #
    # else:  # if not relevant
    #     print("item " + str(x) + " not relevent!")
    #     return
    image = small_image
    item_mask = mask
    # try:
    specio = new_finger_print.spaciogram_finger_print(image, item_mask)
    doc["specio"] = specio.tolist()
    # except:
    #     print("specio specio specio scpecio failed")
    #     return
    # try:
    #     histo = new_finger_print.histogram_stack_finger_print(image, item_mask)
    #     doc["histo"] = histo.tolist()
    # except:
    #     print("histo histo histo histo failed")
    #     return
    db.nate_testing.find_one_and_update({'id': doc['id']},
                                        {"$set": {"specio": doc["specio"]}})
    # db.nate_testing.insert_one(doc)
    print("item " + str(x) + " updated/inserted with success!")
    return

# def mr8_4_demo(img, fc, mask):
#     # print (fc)
#     # if len(fc) == 4:
#     #     x0, y0, w, h = fc
#     #     s_size = min(w, h)
#     #     # while divmod(s_size, 5)[1] != 0:
#     #     #     s_size -= 1
#     #     #     print "s_size:", s_size
#     # # sample = gray_img[y0 + 3 * s_size:y0 + 4 * s_size, x0:x0 + s_size]
#     # else:
#     #     s_size = np.asarray(0.1 * img.shape[0])
#     #     print "s_size:", s_size
#     s_size = fc[3]
#     trimmed_mask = fp_yuli_MR8.trim_mask(img, mask)
#     response = fp_yuli_MR8.yuli_fp(trimmed_mask, s_size)
#     print len(response)
#
#     ms_response = []
#     for idx, val in enumerate(response):
#         ms_response.append(fp_yuli_MR8.mean_std_pooling(val, 5).tolist())
#     return ms_response
