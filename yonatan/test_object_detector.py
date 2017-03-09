#!/usr/bin/env python

import numpy as np
import cv2
import time
import dlib
from trendi.yonatan import functions


dress_detector_045 = dlib.simple_object_detector("/data/detector_0.45_C_40_symmetry.svm")
dress_detector_07 = dlib.simple_object_detector("/data/detector_0.7_C_40_symmetry.svm")

print "Starting the dress detector testing!"


def find_dress_dlib(image, detector, max_num_of_faces=10):
    # start = time.time()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    # can also put 0.
    if detector == 0.45:
        dets = dress_detector_045(image, 1)
    elif detector == 0.7:
        dets = dress_detector_07(image, 1)

    print "number of dresses found: {0}".format(len(dets))
    # print("Done in %.3f s." % (time.time() - start))

    return dets


def print_image_with_bb(image, bb_coordinates, label, path, color, to_print=True, limit_height=False, limit_width=False):

    if not bb_coordinates:
        return False

    if color == 'red':
        rgb = (0, 0, 255)
    else:
        rgb = (0, 255, 0)

    h, w, d = image.shape

    for d in bb_coordinates:
        if d.left() < 0:
            left = 0
        else:
            left = d.left()

        if d.top() < 0:
            top = 0
        else:
            top = d.top()

        if d.right() > w - 1:
            right = w - 1
        else:
            right = d.right()

        if d.bottom() > h - 1:
            bottom = h - 1
        else:
            bottom = d.bottom()

        width = d.right()-left
        height = d.bottom()-d.top()

        if height < limit_height * h:
            print "bb to short"
            return False

        print "d.left: {0}, d.top: {1}, d.right: {2}, d.bottom: {3}\nwidth: {4}, height: {5}\n".format(d.left(), d.top(), d.right(), d.bottom(), width, height)
        cv2.rectangle(image, (left, top), (right, bottom), rgb, 3)

    if to_print:
        print cv2.imwrite(path + str(label) + ".jpg", image)

    return True


def detect(url_or_np_array, label=0):

    full_image = functions.url_to_np_array(url_or_np_array)

    if full_image is None:
        return None

    print "dress detector 0.45!"
    dets = find_dress_dlib(full_image, 0.45)

    print "dress detector 0.7!"
    dets2 = find_dress_dlib(full_image, 0.7)

    if dets and dets2:
        return print_image_with_bb(full_image, dets, label, "/data/yonatan/linked_to_web/svm_C_40_symmetry/unique/dress_detector_result_045_", 'red', to_print=False, limit_height=0.25, limit_width=False) +\
               print_image_with_bb(full_image, dets2, label, "/data/yonatan/linked_to_web/svm_C_40_symmetry/unique/dress_detector_result_07_", 'green', limit_height=0.25, limit_width=False)
    elif dets:
        return print_image_with_bb(full_image, dets, label, "/data/yonatan/linked_to_web/svm_C_40_symmetry/dress_detector_result_045_", 'red', limit_height=0.25, limit_width=False)
    elif dets2:
        return print_image_with_bb(full_image, dets2, label, "/data/yonatan/linked_to_web/svm_C_40_symmetry/dress_detector_result_07_", 'green', limit_height=0.25, limit_width=False)
    else:
        return False
