__author__ = 'yonti'
'''
all the old fingerprint testing functions were moved
to old_fingerprint_stuff
'''
import logging

import cv2
import numpy as np

import background_removal
import Utils
import constants

fingerprint_length = constants.fingerprint_length
histograms_length = constants.histograms_length
db = constants.db


# collection = constants.update_collection_name


def fp(img, bins=histograms_length, fp_length=fingerprint_length, mask=None):
    if mask is None or cv2.countNonZero(mask) == 0:
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
        print "mask shape: " + str(mask.shape)
        print "image shape: " + str(img.shape)
        print str(mask.shape[0] / float(mask.shape[1])) + ',' + str(img.shape[0] / float(img.shape[1]))
        raise ValueError('trouble with mask size, resetting to image size')
    n_pixels = cv2.countNonZero(mask)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    # histograms
    hist_hue = cv2.calcHist([hsv], [0], mask, [bins[0]], [0, 180])
    hist_hue = [item for sublist in hist_hue for item in sublist]  # flatten nested
    hist_hue = np.divide(hist_hue, n_pixels)

    hist_sat = cv2.calcHist([hsv], [1], mask, [bins[1]], [0, 255])
    hist_sat = [item for sublist in hist_sat for item in sublist]
    hist_sat = np.divide(hist_sat, n_pixels)

    hist_int = cv2.calcHist([hsv], [2], mask, [bins[2]], [0, 255])
    hist_int = [item for sublist in hist_int for item in sublist]  # flatten nested list
    hist_int = np.divide(hist_int, n_pixels)

    # Uniformity  t(5)=sum(p.^ 2);
    hue_uniformity = np.dot(hist_hue, hist_hue)
    sat_uniformity = np.dot(hist_sat, hist_sat)
    int_uniformity = np.dot(hist_int, hist_int)

    # Entropy   t(6)=-sum(p. *(log2(p+ eps)));
    eps = 1e-15
    max_log_value = np.log2(bins)  # this is same as sum of p log p
    l_hue = -np.log2(hist_hue + eps) / max_log_value[0]
    hue_entropy = np.dot(hist_hue, l_hue)
    l_sat = -np.log2(hist_sat + eps) / max_log_value[1]
    sat_entropy = np.dot(hist_sat, l_sat)
    l_int = -np.log2(hist_int + eps) / max_log_value[2]
    int_entropy = np.dot(hist_int, l_int)

    result_vector = [hue_uniformity, sat_uniformity, int_uniformity, hue_entropy, sat_entropy, int_entropy]
    result_vector = np.concatenate((result_vector, hist_hue, hist_sat, hist_int), axis=0)

    return result_vector[:fp_length]


def generate_mask_and_insert(doc, image_url=None, fp_date=None, coll="products"):
    """
    Takes an image + whatever else you give it, and handles all the logic (using/finding/creating a bb, then a mask)
    Work in progress...
    :param image_url:
    :param doc: ShopStyle DB doc
    :return:
    """
    image_url = image_url or doc["image"]["sizes"]["XLarge"]["url"]
    collection = coll
    image = Utils.get_cv2_img_array(image_url)
    if not Utils.is_valid_image(image):
        logging.warning("image is None. url: {url}".format(url=image_url))
        return
    small_image, resize_ratio = background_removal.standard_resize(image, 400)
    del image

    if not Utils.is_valid_image(small_image):
        logging.warning("small_image is Bad. {img}".format(img=small_image))
        return

    mask = background_removal.get_fg_mask(small_image)
    fingerprint = fp(small_image, mask=mask)

    fp_as_list = fingerprint.tolist()
    doc["fingerprint"] = fp_as_list
    doc["download_data"]["first_dl"] = fp_date
    doc["download_data"]["dl_version"] = fp_date
    doc["download_data"]["fp_version"] = constants.fingerprint_version
    try:
        db[collection].insert_one(doc)
        print "prod inserted successfully"
        # db.fp_in_process.delete_one({"id": doc["id"]})
    except:
        # db.download_data.find_one_and_update({"criteria": collection},
        #                                      {'$inc': {"errors": 1}})
        print "error inserting"

    return fp_as_list

