__author__ = 'liorsabag'

import numpy as np
import cv2
import string
import logging
import constants
"""
import classify_core
import Utils

DEFAULT_CLASSIFIERS = ["/home/www-data/web2py/applications/fingerPrint/modules/shirtClassifier.xml",
                       "/home/www-data/web2py/applications/fingerPrint/modules/pantsClassifier.xml",
                       "/home/www-data/web2py/applications/fingerPrint/modules/dressClassifier.xml"]
"""

fingerprint_length = constants.fingerprint_length

def crop_image_to_bb(img, bb_coordinates_string_or_array):
    if isinstance(bb_coordinates_string_or_array, basestring):
        bb_array = [int(bb) for bb in string.split(bb_coordinates_string_or_array)]
    else:
        bb_array = bb_coordinates_string_or_array

    x = bb_array[0]
    y = bb_array[1]
    w = bb_array[2]
    h = bb_array[3]
    hh, ww, d = img.shape
    if (x + w <= ww) and (y + h <= hh):
	cropped_img = img[y:y+h,x:x+w]
    else:
        cropped_img = img
        logging.warning('Could not crop. Bad bounding box: imsize:' + str(ww) + ',' + str(hh) +
                        ' vs.:' + str(x + w) + ',' + str(y + h))

    return cropped_img

def fp(img, bounding_box=None, weights = np.ones(fingerprint_length) , histogram_length=25, use_intensity_histogram=False):
    if (bounding_box is not None) and (bounding_box != np.array([0, 0, 0, 0])).all():
        img = crop_image_to_bb(img, bounding_box)
    #crop out the outer 1/s of the image for color/texture-based features
    if img is None:
        return None
    s = 5
    h = img.shape[1]
    w = img.shape[0]

    if h==0 or w==0:
        return None

    r = [h / s, w / s, h - 2 * h / s, w - 2 * w / s]
#    roi = np.zeros((r[3], r[2], 3), np.uint8)
    roi = crop_image_to_bb(img,r)
    n_pixels = roi.shape[0] * roi.shape[1]


    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    #histograms
    bins = histogram_length

    hist_hue = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_hue = [item for sublist in hist_hue for item in sublist]  #flatten nested
    hist_hue = np.divide(hist_hue, n_pixels)

    hist_sat = cv2.calcHist([hsv], [1], None, [bins], [0, 255])
    hist_sat = [item for sublist in hist_sat for item in sublist]
    hist_sat = np.divide(hist_sat, n_pixels)

    hist_int = cv2.calcHist([hsv], [2], None, [bins], [0, 255])
    hist_int = [item for sublist in hist_int for item in sublist]  #flatten nested list
    hist_int = np.divide(hist_int, n_pixels)

    #Uniformity  t(5)=sum(p.^ 2);
    hue_uniformity = np.dot(hist_hue, hist_hue)
    sat_uniformity = np.dot(hist_sat, hist_sat)
    int_uniformity = np.dot(hist_int, hist_int)

    #Entropy   t(6)=-sum(p. *(log2(p+ eps)));
    eps = 1e-15
    max_log_value = np.log2(bins)  #this is same as sum of p log p
    l_hue = -np.log2(hist_hue + eps)/max_log_value
    hue_entropy = np.dot(hist_hue, l_hue)
    l_sat = -np.log2(hist_sat + eps)/max_log_value
    sat_entropy = np.dot(hist_sat, l_sat)
    l_int = -np.log2(hist_int + eps)/max_log_value
    int_entropy = np.dot(hist_int, l_int)

    result_vector = [hue_uniformity, sat_uniformity, int_uniformity, hue_entropy, sat_entropy, int_entropy]
    result_vector = np.concatenate((result_vector, hist_hue, hist_sat), axis=0)
    if use_intensity_histogram:
        result_vector = np.concatenate((result_vector,hist_int), axis=0)

    result_vector = result_vector * weights
    return result_vector


def my_range(start, stop, inc):
    r = start
    while r < stop:
        yield r
        r += inc

