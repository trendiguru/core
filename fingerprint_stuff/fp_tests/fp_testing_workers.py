__author__ = 'yonti'

'''
workers for the new_fp_testing function
'''

import logging

import fingerprint_core as fp
import background_removal
import utils_tg
import constants

fingerprint_length = constants.fingerprint_length
histograms_length = constants.histograms_length

db = constants.db_name


# define a example function
def add_new_fp(doc, x):
    image_url = doc["image"]["sizes"]["XLarge"]["url"]

    image = utils_tg.get_cv2_img_array(image_url)
    if not utils_tg.is_valid_image(image):
        logging.warning("image is None. url: {url}".format(url=image_url))
        return

    small_image, resize_ratio = background_removal.standard_resize(image, 400)
    del image

    mask = fp.generate_mask_and_insert(image_url=None, db_doc=doc, save_to_db=False, mask_only=True)
    try:
        fingerprint = fp.fp(small_image, mask=mask)
        doc["new_fp"] = fingerprint.tolist()
        doc["fp_version"] = 999
        db.fp_testing.insert(doc)
    except Exception as ex:
        logging.warning("Exception caught while inserting element #" + str(x) + " to the collection".format(ex))

    return x
