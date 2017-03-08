#!/usr/bin/env python

import numpy as np
import cv2
import time
import dlib
from trendi.yonatan import functions


dress_detector_045 = dlib.simple_object_detector("/data/detector_0.45_C_40_symmetry.svm")
dress_detector_07 = dlib.simple_object_detector("/data/detector_0.7_C_40_symmetry.svm")


def find_dress_dlib(image, detector, max_num_of_faces=10):
    start = time.time()

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    if len(dets) == 0:
        print "no dress!!"
        return dets
    else:
        print "great success!"
        print len(dets)
        return dets

    #     img = io.imread(f)
    #     dets = dress_detector(img)


def print_image_with_bb(image, bb_coordinates, label, path, limit_height=False, limit_width=False):

    if not bb_coordinates:
        return False

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

        if d.right() < 0:
            right = 0
        else:
            right = d.right()

        if d.bottom() < 0:
            bottom = 0
        else:
            bottom = d.bottom()

        width = d.right()-left
        height = d.bottom()-d.top()

        if height < limit_height * h:
            print "bb to short"
            return False

        print "d.left: {0}, d.top: {1}, d.right: {2}, d.bottom: {3}\nwidth: {4}, height: {5}\n".format(d.left(), d.top(), d.right(), d.bottom(), width, height)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 3)

        print cv2.imwrite(path + str(label) + ".jpg", image)

        return True


def pad(array, reference, offsets):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros(reference.shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result


def detect(url_or_np_array, label=0):

    print "Starting the dress detector testing!"

    full_image = functions.url_to_np_array(url_or_np_array)

    if full_image is None:
        return None



    # h, w, d = full_image.shape
    # new_h = h + 30
    # new_w = w + 30
    #
    # padded_image = np.zeros((new_h, new_w, d))
    # h_offset = 15
    # w_offset = 15
    # padded_image[h_offset:h + h_offset, w_offset:w + w_offset, :] = full_image
    #
    # padded_image2 = padded_image.copy()
    #
    # print "image.shape: {0}".format(padded_image.shape)

    print "dress detector 0.45!"
    dets = find_dress_dlib(full_image, 0.45)

    # for d in dets:
    #     if d.left() < 0:
    #         left = d.left() + 15
    #     else:
    #         left = d.left()
    #
    #     width = d.right()-left
    #     height = d.bottom()-d.top()
    #
    #     if height < 0.25 * h:
    #         dets = 0
    #
    #     print "d.left: {0}, d.top: {1}, d.right: {2}, d.bottom: {3}\nwidth: {4}, height: {5}\n".format(left, d.top(), d.right(), d.bottom(), width, height)
    #     cv2.rectangle(padded_image, (left, d.top()), (d.right(), d.bottom()), (0, 0, 255), 3)

    print "dress detector 0.7!"
    dets2 = find_dress_dlib(full_image, 0.7)
    # for d in dets2:
    #     if d.left() < 0:
    #         left = d.left() + 15
    #     else:
    #         left = d.left()
    #
    #     width = d.right() - left
    #     height = d.bottom() - d.top()
    #
    #     if height < 0.25 * h:
    #         dets2 = 0
    #
    #     print "d.left: {0}, d.top: {1}, d.right: {2}, d.bottom: {3}\nwidth: {4}, height: {5}\n".format(left, d.top(), d.right(), d.bottom(), width, height)
    #     cv2.rectangle(padded_image2, (left, d.top()), (d.right(), d.bottom()), (0, 255, 0), 3)

    if dets and dets2:
        print_image_with_bb(full_image, dets, label, "/data/yonatan/linked_to_web/svm_C_40_symmetry/unique/dress_detector_result_045_", limit_height = 0.25, limit_width = False)
        print_image_with_bb(full_image, dets2, label, "/data/yonatan/linked_to_web/svm_C_40_symmetry/unique/dress_detector_result_07_", limit_height = 0.25, limit_width = False)
        # print cv2.imwrite("/data/yonatan/linked_to_web/svm_C_60/unique/dress_detector_result_045_" + str(label) + ".jpg", padded_image)
        # print cv2.imwrite("/data/yonatan/linked_to_web/svm_C_60/unique/dress_detector_result_07_" + str(label) + ".jpg", padded_image2)
        return 2
    elif dets:
        print_image_with_bb(full_image, dets, label, "/data/yonatan/linked_to_web/svm_C_40_symmetry/dress_detector_result_045_", limit_height = 0.25, limit_width = False)
        # print cv2.imwrite("/data/yonatan/linked_to_web/svm_C_60/dress_detector_result_045_" + str(label) + ".jpg", padded_image)
        return True
    elif dets2:
        print_image_with_bb(full_image, dets2, label, "/data/yonatan/linked_to_web/svm_C_40_symmetry/dress_detector_result_07_", limit_height = 0.25, limit_width = False)
        # print cv2.imwrite("/data/yonatan/linked_to_web/svm_C_60/dress_detector_result_07_" + str(label) + ".jpg", padded_image2)
        return True
    else:
        return False
