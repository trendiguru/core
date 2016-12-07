__author__ = 'yonti'

import cv2
import hashlib
import logging
from jaweson import msgpack
import numpy as np
import gevent
from gevent import Greenlet
import requests

# OURS
import Utils
import background_removal
import constants
from db_stuff.recruit.recruit_constants import recruit2category_idx
from utils.imutils import resize_keep_aspect
#from .falcon import sleeve_client, length_client
from .features_api import classifier_client
from .features import color

from .paperdoll import neurodoll_falcon_client as nfc

fingerprint_length = constants.fingerprint_length
histograms_length = constants.histograms_length
db = constants.db

#FEATURES_CLIENT_ADDRESS = "http://37.58.101.173:8084/"


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


def dict_fp(image, mask, category):
    print 'dict fp'
    if category in constants.features_per_category:
        fp_features = constants.features_per_category[category]
    else:
        fp_features = constants.features_per_category['other']
    fingerprint = {feature: Greenlet.spawn(get_feature_fp, feature, image, mask) for feature in fp_features}
    gevent.joinall(fingerprint.values())
    fingerprint = {k: v.value for k, v in fingerprint.iteritems()}
    # fingerprint = {feature: get_feature_fp(image, mask, feature) for feature in fp_features}
    return fingerprint


def get_feature_fp(feature, image, mask=None):
    if feature == 'color':
        print 'color'
        return color.execute(image, histograms_length, fingerprint_length, mask)
    img = np.copy(image)
    img = resize_keep_aspect(img, output_size=(224,224))
    res = classifier_client.get(feature, img)
    if isinstance(res, dict) and 'data' in res:
        return res['data']
    else:
        return res
    
#    data = msgpack.dumps({"image_or_url": image, "mask": mask})
#    resp = requests.post(FEATURES_CLIENT_ADDRESS+feature, data=data)
#    return msgpack.loads(resp.content)['data']

    # if feature == 'color':
    #     print 'color'
    #     return color.execute(image, histograms_length, fingerprint_length, mask)
    # elif feature == 'sleeve_length':
    #     print 'sleeve_length'
    #     return sleeve_client.get_sleeve(image)['data']
    # elif feature == 'length':
    #     print 'length'
    #     return length_client.get_length(image)['data']
    # else:
    #     return []


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


def generate_mask_and_insert(doc, image_url=None, fp_date=None, coll="products", img=None, neuro=False):
    """
    Takes an image + whatever else you give it, and handles all the logic (using/finding/creating a bb, then a mask)
    Work in progress...
    :param image_url:
    :param doc: ShopStyle DB doc
    :return:
    """
    image_url = image_url or doc["image"]["sizes"]["XLarge"]["url"]
    collection = coll
    if neuro or img is not None:
        image = img
    else:
        image = Utils.get_cv2_img_array(image_url)
    if not Utils.is_valid_image(image):
        logging.warning("image is None. url: {url}".format(url=image_url))
        return
    # img_hash = get_hash(image)
    # if db[coll].find_one({'img_hash': img_hash}):
    #     return
    small_image, resize_ratio = background_removal.standard_resize(image, 400)
    # del image

    if not Utils.is_valid_image(small_image):
        logging.warning("small_image is Bad. {img}".format(img=small_image))
        return
    category = doc['categories']
    print category
    if neuro:
        category_idx = recruit2category_idx[category]
        success, neuro_mask = neurodoll(image, category_idx)
        if not success:
            print "error neurodolling"
            return []
        small_mask = cv2.resize(neuro_mask, (400, 400))

    else:
        small_mask = background_removal.get_fg_mask(small_image)

    fingerprint = dict_fp(small_image, small_mask, category)
    print 'fingerprint done'
    doc["fingerprint"] = fingerprint
    doc["download_data"]["first_dl"] = fp_date
    doc["download_data"]["dl_version"] = fp_date
    doc["download_data"]["fp_version"] = constants.fingerprint_version
    print "prod insert ..."
    try:
        db[collection].insert_one(doc)
        print "successfull"
        # db.fp_in_process.delete_one({"id": doc["id"]})
    except:
        # db.download_data.find_one_and_update({"criteria": collection},
        #                                      {'$inc': {"errors": 1}})
        print "failed"

    return fingerprint['color']


def get_hash(image):
    m = hashlib.md5()
    m.update(image)
    url_hash = m.hexdigest()
    return url_hash


