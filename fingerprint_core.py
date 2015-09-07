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

db = constants.db_name
collection = constants.update_collection

def fp(img, bins=histograms_length, fp_length=fingerprint_length, mask=None):
    if mask is None or cv2.countNonZero(mask) == 0:
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
        print('trouble with mask size, resetting to image size')
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
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


def generate_mask_and_insert(image_url=None, doc=None, save_to_db=False, mask_only=False):
    """
    Takes an image + whatever else you give it, and handles all the logic (using/finding/creating a bb, then a mask)
    Work in progress...
    :param image_url:
    :param doc: ShopStyle DB doc
    :return:
    """
    image_url = image_url or doc["image"]["sizes"]["XLarge"]["url"]

    image = Utils.get_cv2_img_array(image_url)
    if not Utils.is_valid_image(image):
        logging.warning("image is None. url: {url}".format(url=image_url))
        return
    small_image, resize_ratio = background_removal.standard_resize(image, 400)
    # I think we can delete the image... memory management FTW??
    del image
    # print "Image URL: {0}".format(image_url)

    CLASSIFIER_FOR_CATEGORY = {}

    if "bounding_box" in doc.keys() and doc["bounding_box"] != [0, 0, 0, 0] and doc["bounding_box"] is not None:
        chosen_bounding_box = doc["bounding_box"]
        chosen_bounding_box = [int(b) for b in (np.array(chosen_bounding_box) / resize_ratio)]
        mask = background_removal.get_fg_mask(small_image, chosen_bounding_box)
        logging.debug("Human bb found: {bb} for item: {id}".format(bb=chosen_bounding_box, id=doc["id"]))
    # otherwise use the largest of possibly many classifier bb's
    else:
        if "categories" in doc:
            classifier = CLASSIFIER_FOR_CATEGORY.get(doc["categories"][0]["id"], "")
        else:
            classifier = None
        if not Utils.is_valid_image(small_image):
            logging.warning("small_image is Bad. {img}".format(img=small_image))
            return
        mask = background_removal.get_fg_mask(small_image)
        bounding_box_list = []
        if classifier and not classifier.empty():
            # then - try to classify the image (white backgrounded and get a more accurate bb
            white_bckgnd_image = background_removal.image_white_bckgnd(small_image, mask)
            try:
                bounding_box_list = classifier.detectMultiScale(white_bckgnd_image)
            except KeyError:
                logging.info("Could not classify with {0}".format(classifier))
        # choosing the biggest bounding box if there are a few
        max_bb_area = 0
        chosen_bounding_box = None
        for possible_bb in bounding_box_list:
            if possible_bb[2] * possible_bb[3] > max_bb_area:
                chosen_bounding_box = possible_bb
                max_bb_area = possible_bb[2] * possible_bb[3]
        if chosen_bounding_box is None:
            logging.info("No Bounding Box found, using the whole image. "
                         "Document id: {0}, BB_list: {1}".format(doc.get("id"), str(bounding_box_list)))
        else:
            mask = background_removal.get_fg_mask(small_image, chosen_bounding_box)

    if mask_only:
        return mask

    fingerprint = fp(small_image, mask=mask)

    # fp_as_list = fingerprint.tolist()

    if save_to_db:
        db[collection].update_one({"_id": doc["_id"]},
                                  {"$set": {"fingerprint": fingerprint,
                                            "fp_version": constants.fingerprint_version}})
    return fingerprint
