__author__ = 'yonti'

import cv2
import logging
from gevent import Greenlet, joinall
import numpy as np

from ..recruit.recruit_constants import recruit2category_idx
from ...falcon import sleeve_client, length_client
from ...features import color
from ...paperdoll import neurodoll_falcon_client as nfc
from ... import Utils, constants, background_removal
from ...features_api import classifier_client
from termcolor import colored
from ...utils_tg.imutils import resize_keep_aspect

fingerprint_length = constants.fingerprint_length
histograms_length = constants.histograms_length
db = constants.db


def neurodoll(image, category_idx):
    dic = nfc.pd(image, category_idx)
    if not dic['success']:
        return False, []
    neuro_mask = dic['mask']
    #img = cv2.resize(image,(256,256))
    # rect = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    mask = np.zeros(image.shape[:2], np.uint8)
    med = np.median(neuro_mask)
    mask[neuro_mask > med] = 3
    mask[neuro_mask < med] = 2
    try:
        cv2.grabCut(image, mask, None, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
    except:
        return False, []
    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype(np.uint8)
    return True, mask2


def dict_fp(fing, image, mask, category):
    if category in constants.features_per_category:
        fp_features = constants.features_per_category[category]
    else:
        fp_features = constants.features_per_category['other']
    if fing is None:
        fp_keys = []
        fing = {}
    else:
        fp_keys = fing.keys()
        for f in fp_keys:
            if f == 'collar':
                if fing['collar'] is None or len(fing['collar']) == 9:
                    fing.pop('collar', None)
            elif fing[f] is None:
                fing.pop(f, None)
        fp_keys = fing.keys()
    fingerprint = {feature: Greenlet.spawn(get_feature_fp, feature, image, mask) for feature in fp_features if feature not in fp_keys}
    joinall(fingerprint.values())
    fingerprint = {k: v.value for k, v in fingerprint.iteritems()}
    if len(fingerprint.keys()):
        fingerprint = dict(fingerprint, **fing)
        return fingerprint, True
    else:
        return fing, False
    # fingerprint = {feature: get_feature_fp(image, mask, feature) for feature in fp_features}


def get_feature_fp(feature, image, mask=None):
    if feature == 'color':
        print 'color'
        return color.execute(image, histograms_length, fingerprint_length, mask)
    img = np.copy(image)
    img = resize_keep_aspect(img, output_size=(224, 224))
    res = classifier_client.get(feature, img)
    if isinstance(res, dict) and 'data' in res:
        return res['data']
    else:
        return res


def fp(img, bins=histograms_length, fp_length=fingerprint_length, mask=None):
    if mask is None or cv2.countNonZero(mask) == 0:
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
        print "mask shape: " + str(mask.shape)
        print "image shape: " + str(img.shape)
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


def refresh_fp(fingerprint, collection_name, item_id, category, image, small_image):

    collection = db[collection_name]

    if "recruit" in collection_name:
        category_idx = recruit2category_idx[category]
        success, neuro_mask = neurodoll(image, category_idx)
        if not success:
            print "error neurodolling"
            return []
        small_mask = cv2.resize(neuro_mask, (400, 400))

    else:
        small_mask = background_removal.get_fg_mask(small_image)

    if type(fingerprint) == list:
        fingerprint = {'color': fingerprint}

    fingerprint, any_change = dict_fp(fingerprint, small_image, small_mask, category)
    print 'fingerprint done'
    if any_change:
        print colored('!!!!!!!!!!!!!!!!!!!!changed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', 'yellow')
        try:
            collection.update_one({'_id': item_id}, {'$set': {'fingerprint': fingerprint}})
            print "successfull"
            # db.fp_in_process.delete_one({"id": doc["id"]})
        except:
            # db.download_data.find_one_and_update({"criteria": collection},
            #                                      {'$inc': {"errors": 1}})
            collection.delete_one({'_id': item_id})
            print "failed"
    else:
        print 'same'
    print 'done!'



