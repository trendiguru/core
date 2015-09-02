__author__ = 'yonti'

import numpy as np
import cv2

import constants

fingerprint_length = constants.fingerprint_length
histograms_length = constants.histograms_length


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
