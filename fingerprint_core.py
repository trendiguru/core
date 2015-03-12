__author__ = 'liorsabag'

import numpy as np
import cv2
import string
import logging

"""
import classify_core
import Utils

DEFAULT_CLASSIFIERS = ["/home/www-data/web2py/applications/fingerPrint/modules/shirtClassifier.xml",
                       "/home/www-data/web2py/applications/fingerPrint/modules/pantsClassifier.xml",
                       "/home/www-data/web2py/applications/fingerPrint/modules/dressClassifier.xml"]
"""

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
        #rectok=True
        r = [x, y, w, h]
        #allRects.append(r)
        mask = np.zeros(img.shape[:2], np.uint8)
        cropped_img = np.zeros((r[3], r[2], 3), np.uint8)
        mask[r[0]:r[2], r[1]:r[3]] = 255

        for xx in range(r[2]):
            for yy in range(r[3]):
                cropped_img[yy, xx, :] = img[yy + r[1], xx + r[0], :]

    else:
        cropped_img = img
        logging.warning('Could not crop. Bad bounding box: imsize:' + str(ww) + ',' + str(hh) +
                        ' vs.:' + str(x + w) + ',' + str(y + h))

    return cropped_img

def fp(img, bounding_box=None):
    if (bounding_box is not None) and (bounding_box != np.array([0, 0, 0, 0])).all():
        img = crop_image_to_bb(img, bounding_box)
    #crop out the outer 1/s of the image for color/texture-based features
    s = 5
    h = img.shape[1]
    w = img.shape[0]
    r = [h / s, w / s, h - 2 * h / s, w - 2 * w / s]

    roi = np.zeros((r[3], r[2], 3), np.uint8)
    # should use imageop.crop here instead, its prob. faster
    for xx in range(r[2]):
        for yy in range(r[3]):
            roi[yy, xx, :] = img[yy + r[1], xx + r[0], :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    #histograms
    bins = 25
    n_pixels = roi.shape[0] * roi.shape[1]

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
    max_log_value = log2(bins)  #this is same as sum of p log p 
    l_hue = -np.log2(hist_hue + eps)/max_log_value
    hue_entropy = np.dot(hist_hue, l_hue)
    l_sat = -np.log2(hist_sat + eps)/max_log_value
    sat_entropy = np.dot(hist_sat, l_sat)
    l_int = -np.log2(hist_int + eps)/max_log_value
    int_entropy = np.dot(hist_int, l_int)

    result_vector = [hue_uniformity, sat_uniformity, int_uniformity, hue_entropy, sat_entropy, int_entropy]
    result_vector = np.concatenate((result_vector, hist_hue, hist_sat), axis=0)

    return result_vector


def my_range(start, stop, inc):
    r = start
    while r < stop:
        yield r
        r += inc


def gaussian1(x, x0, c, sigma):
    return c * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

"""
def classify_and_fingerprint(path_to_image_file):

    img = Utils.get_cv2_img_array(path_to_image_file)
    roi = []
    classification_dict = classify_core.classify_image_with_classifiers(img, *DEFAULT_CLASSIFIERS)
    bb_coordinates = classification_dict["/home/www-data/web2py/applications/fingerPrint/modules/shirtClassifier.xml"]
    if len(bb_coordinates) > 0:
        if bb_coordinates[0] is not None:
            shirt_found = True
            roi = crop_image_to_bb(img, bb_coordinates[0])
        else:
            roi = img
            bb_coordinates = [[0, 0, 0, 0]]
            logging.debug("in fingerprint_core.py: len(bb_coordinates)>0 but BB[0] is None -- REALLY WEIRD:")
            print(bb_coordinates)

    else:
        #        roi = None  #if no BB was formed , don't return an answer!!!!
        print("in fingerPrint2.py:bad roi (could not crop?)-using entire img - len(BB)!>0")
        roi = img
        bb_coordinates = [[0, 0, 0, 0]]
    if roi is not None:
        fingerprint = fp(roi)
    else:
        print("in fingerPrint2.py:bad roi (could not crop?) - using entire img (again)")
        fingerprint_length = 56
        fingerprint = [0 for x in range(fingerprint_length)]
        fingerprint[0] = -1
        print("in fingerPrint2.py:fp=" + str(fingerprint))


    # right now, we're only doing shirts, so it's binary
    # 0 - nothing found, 1 - at least one shirt found.
    # even within this simplified case, still need to figure out
    # how to deal with multiple shirts in an image
    classification_list = [0]

    logging.debug('in fingerPrint2.py:classify_and_fingerprint:fingerprint=' + str(fingerprint))
    logging.debug('in fingerPrint2.py:classify_and_fingerprint:BB=' + str(bb_coordinates))
    return classification_list, fingerprint, bb_coordinates


#takes classifier file path
def classify_and_fingerprint_with_classifier(path_to_image_file, classifier_xml):
    #pdb.set_trace()
    print("in fingerPrint2:classify_and_fingerprint:path_to_image_file: " + str(path_to_image_file))
    REMOTE_FILE = False
    item_found = False
    fingerprint = ""
    classification_dict = {}
    #read arg from command line if none given to function
    if path_to_image_file is not None:
        PATH_TO_FILE = path_to_image_file
        #else:
        #wtf is supposed to happen here - who is calling this from command line??

    #we want to be able to read URL as well as local file path
    if "://" in path_to_image_file:
        FILENAME = path_to_image_file.split('/')[-1].split('#')[0].split('?')[0]
        res = urllib.urlretrieve(PATH_TO_FILE, FILENAME)
        PATH_TO_FILE = FILENAME
        REMOTE_FILE = True

    #pdb.set_trace()
    #main prog starts here
    img = cv2.imread(PATH_TO_FILE)
    roi = []
    classification_dict = classify_core.classify_image_with_classifiers(img, classifier_xml)
    BB_coordinates = classification_dict[classifier_xml]
    if len(BB_coordinates) > 0:
        if BB_coordinates[0] is not None:
            item_found = True
            roi = crop_image_to_bb(img, BB_coordinates[0])
        else:
            roi = img
            BB_coordinates = [[0, 0, 0, 0]]
            print("in fingerPrint2.py: len(BB_coordinates)>0 but BB[0] is None -- REALLY WEIRD:")
            print(BB_coordinates)

    else:
        #        roi = None  #if no BB was formed , don't return an answer!!!!
        print("in fingerPrint2.py:bad roi (could not crop?)-using entire img - len(BB)!>0")
        roi = img
        BB_coordinates = [[0, 0, 0, 0]]
    if roi is not None:
        fingerprint = fp(roi)
    else:
        print("in fingerPrint2.py:bad roi (could not crop?) - using entire img (again)")
        fingerprint_length = 56
        fingerprint = [0 for x in range(fingerprint_length)]
        fingerprint[0] = -1
        print("in fingerPrint2.py:fp=" + str(fingerprint))

    if REMOTE_FILE:
        os.remove(PATH_TO_FILE)

    # right now, we're only doing shirts, so it's binary
    # 0 - nothing found, 1 - at least one shirt found.
    # even within this simplified case, still need to figure out
    # how to deal with multiple shirts in an image
    classification_list = []
    if item_found:
        classification_list.append(1)
    else:
        classification_list.append(0)

    print('in fingerPrint2.py:classify_and_fingerprint:fingerprint=' + str(fingerprint))
    print('in fingerPrint2.py:classify_and_fingerprint:BB=' + str(BB_coordinates))
    return classification_list, fingerprint, BB_coordinates


#  classification_list, fingerprint, bounding_box_list = fp2.classify_and_fingerprint_with_classifier_and_human_bb(row.IMAGEURL, classifier_xml,boundin
#takes classifier file path and human 'known good' bounding box
def classify_and_fingerprint_with_classifier_and_human_bb(path_to_image_file, classifier_xml, human_bounding_box):

    img = Utils.get_cv2_img_array(path_to_image_file)
    roi = []
    classification_dict = classify_core.classify_image_with_classifiers(img, classifier_xml)
    BB_coordinates = [human_bounding_box]
    if len(BB_coordinates) > 0:
        if BB_coordinates[0] is not None:
            item_found = True
            roi = crop_image_to_bb(img, BB_coordinates[0])
        else:
            roi = img
            BB_coordinates = [[0, 0, 0, 0]]
            print(
                "in fingerPrint2:classify_and_fp_withClassifierAndHumanBB: len(BB_coordinates)>0 but BB[0] is None -- REALLY WEIRD:")
            print(BB_coordinates)

    else:
        #        roi = None  #if no BB was formed , don't return an answer!!!!
        print(
            "in fingerPrint2.py:classify_and_fp+withCLassifierAndHumanBB:bad roi (could not crop?)-using entire img - len(BB)!>0")
        roi = img
        BB_coordinates = [[0, 0, 0, 0]]
    if roi is not None:
        fingerprint = fp(roi)
    else:
        print("in fingerPrint2.py:classify_and_fpWithClassifierAndHumanBB:bad roi (could not crop?) - using entire img")
        fingerprint_length = 56
        fingerprint = [0 for x in range(fingerprint_length)]
        fingerprint[0] = -1
        print("in fingerPrint2.py:fp=" + str(fingerprint))

    if REMOTE_FILE:
        os.remove(PATH_TO_FILE)

    # right now, we're only doing shirts, so it's binary
    # 0 - nothing found, 1 - at least one shirt found.
    # even within this simplified case, still need to figure out
    # how to deal with multiple shirts in an image
    classification_list = []
    if item_found:
        classification_list.append(1)
    else:
        classification_list.append(0)

    print('in fingerPrint2.py:classify_and_fingerprint_with_classifier_and_humanBB:fingerprint=' + str(fingerprint))
    print('in fingerPrint2.py:classify_and_fingerprint_with_classifier_and_humanBB:BB=' + str(BB_coordinates))
    return classification_list, fingerprint, BB_coordinates


def classify_and_fingerprint_dresses(path_to_image_file):
    #pdb.set_trace()
    print("in fingerPrint2:classify_and_fingerprint_dresses:path_to_image_file: " + str(path_to_image_file))
    REMOTE_FILE = False
    item_found = False
    fingerprint = ""
    classification_dict = {}
    #read arg from command line if none given to function
    if path_to_image_file is not None:
        PATH_TO_FILE = path_to_image_file
        #else:
        #wtf is supposed to happen here - who is calling this from command line??

    #we want to be able to read URL as well as local file path
    if "://" in path_to_image_file:
        FILENAME = path_to_image_file.split('/')[-1].split('#')[0].split('?')[0]
        res = urllib.urlretrieve(PATH_TO_FILE, FILENAME)
        PATH_TO_FILE = FILENAME
        REMOTE_FILE = True

    #pdb.set_trace()
    #main prog starts here
    img = cv2.imread(PATH_TO_FILE)
    roi = []
    classification_dict = classify.classify_image(img)
    BB_coordinates = classification_dict[
        "/home/www-data/web2py/applications/fingerPrint/modules/dressClassifier001.xml"]
    if len(BB_coordinates) > 0:
        if BB_coordinates[0] is not None:
            item_found = True
            roi = crop_image_to_bb(img, BB_coordinates[0])
        else:
            roi = img
            BB_coordinates = [[0, 0, 0, 0]]
            print("in fingerPrint2.py: len(BB_coordinates)>0 but BB[0] is None -- REALLY WEIRD:")
            print(BB_coordinates)

    else:
        #        roi = None  #if no BB was formed , don't return an answer!!!!
        print("in fingerPrint2.py:bad roi (could not crop?)-using entire img - len(BB)!>0")
        roi = img
        BB_coordinates = [[0, 0, 0, 0]]
    if roi is not None:
        fingerprint = fp(roi)
    else:
        print("in fingerPrint2.py:bad roi (could not crop?) - using entire img (again)")
        fingerprint_length = 56
        fingerprint = [0 for x in range(fingerprint_length)]
        fingerprint[0] = -1
        print("in fingerPrint2.py:fp=" + str(fingerprint))

    if REMOTE_FILE:
        os.remove(PATH_TO_FILE)

    # right now, we're only doing shirts, so it's binary
    # 0 - nothing found, 1 - at least one shirt found.
    # even within this simplified case, still need to figure out
    # how to deal with multiple shirts in an image
    classification_list = []
    if item_found:
        classification_list.append(2)
    else:
        classification_list.append(0)

    print('in fingerPrint2.py:classify_and_fingerprint_dresses:fingerprint=' + str(fingerprint))
    print('in fingerPrint2.py:classify_and_fingerprint_dresses:BB=' + str(BB_coordinates))
    return classification_list, fingerprint, BB_coordinates
"""
