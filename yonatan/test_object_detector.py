#!/usr/bin/env python

import numpy as np
import os
import caffe
from trendi import background_removal, Utils, constants
import cv2
import sys
import argparse
import glob
import time
import urllib
import requests
import dlib
from skimage import io
from ..utils import imutils
# import yonatan_classifier
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt


# detector = dlib.get_frontal_face_detector()
# dress_detector = dlib.simple_object_detector("/data/detector2.svm")
dress_detector = dlib.simple_object_detector("/data/detector5.svm")


def find_dress_dlib(image, max_num_of_faces=10):
    start = time.time()

    ## faces, scores, idx = detector.run(image, 1, -1) - gives more results, those that add low confidence percentage ##
    ## faces, scores, idx = detector.run(image, 1, 1) - gives less results, doesn't show the lowest confidence percentage results ##
    ## i can get only the faces locations with: faces = detector(image, 1) ##

    # dets, scores, idx = dress_detector(image, 1)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    dets = dress_detector(image, 0)

    # print "image.shape: {0}".format(image.shape)

    print "number of dresses found: {0}".format(len(dets))

    if len(dets) == 0:
        print "no dress!!"
        return dets
    else:
        print "great success!"
        print len(dets)
        return dets

        # win_det = dlib.image_window()
    # win_det.set_image(dress_detector)
    #
    # dlib.hit_enter_to_continue()
    #
    # print("Showing detections on the images in the faces folder...")
    # win = dlib.image_window()
    # for f in glob.glob(os.path.join(faces_folder, "*.jpg")):
    #     print("Processing file: {}".format(f))
    #     img = io.imread(f)
    #     dets = dress_detector(img)
    #     print("Number of faces detected: {}".format(len(dets)))
    #     for k, d in enumerate(dets):
    #         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    #             k, d.left(), d.top(), d.right(), d.bottom()))
    #
    #     win.clear_overlay()
    #     win.set_image(img)
    #     win.add_overlay(dets)
    #     dlib.hit_enter_to_continue()


    # for i, d in enumerate(dresses):
    #     print("Detection {}, score: {}, face_type:{}".format(
    #         d, scores[i], idx[i]))
    #
    # print("Done in %.3f s." % (time.time() - start))
    #
    # faces = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(dresses)]
    # if not len(dresses):
    #     return {'are_dresses': False, 'dresses': []}
    # #final_faces = choose_faces(image, faces, max_num_of_faces)
    # print "number of faces: {0}\n".format(len(dresses))
    # return {'are_dresses': len(dresses) > 0, 'dresses': dresses, 'scores': scores}


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


def theDetector(url_or_np_array):

    print "Starting the dress detector testing!"
    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        #full_image = url_to_image(url_or_np_array)
        response = requests.get(url_or_np_array)  # download
        full_image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    elif type(url_or_np_array) == np.ndarray:
        full_image = url_or_np_array
    else:
        return None

    #checks if the face coordinates are inside the image
    if full_image is None:
        print "not a good image"
        return None

    # full_image = imutils.resize_keep_aspect(full_image, output_size=(500, 500))

    # faces = background_removal.find_face_dlib(full_image)

    x, y, z = full_image.shape
    print (x, y, z)
    new_x = x + 30
    new_y = y + 30

    padded_image = np.zeros((new_x, new_y, z))
    print padded_image.shape
    x_offset = 15
    y_offset = 15
    padded_image[x_offset:x + x_offset, y_offset:y + y_offset, :] = full_image

    print "image.shape: {0}".format(padded_image.shape)

    # (h, w) = full_image.shape[:2]
    # center = (w / 2, h / 2)
    # M = cv2.getRotationMatrix2D(center, 90, 1.0)

    # rotate_image = np.rot90(full_image, 1)
    # rotate_image2 = np.rot90(full_image, 3)
    #
    # resized_image = imutils.resize_keep_aspect(full_image, output_size=(300, 300))

    dets = find_dress_dlib(full_image)
    # dets2 = find_dress_dlib(rotate_image)
    # dets3 = find_dress_dlib(rotate_image2)
    # dets4 = find_dress_dlib(resized_image)
    #
    # for i in range(0, len(dets)):
    #     print "left: {0}, top: {1}, right: {2}, bottom: {3} -- regular".format(dets[i].left(), dets[i].top(), dets[i].right(), dets[i].bottom())
    #
    # for i in range(0, len(dets2)):
    #     print "left: {0}, top: {1}, right: {2}, bottom: {3} -- rotate90".format(dets2[i].left(), dets2[i].top(), dets2[i].right(), dets2[i].bottom())
    #
    # for i in range(0, len(dets3)):
    #     print "left: {0}, top: {1}, right: {2}, bottom: {3} -- rotate270".format(dets3[i].left(), dets3[i].top(), dets3[i].right(), dets3[i].bottom())
    #
    # for i in range(0, len(dets4)):
    #     print "left: {0}, top: {1}, right: {2}, bottom: {3} -- resize".format(dets4[i].left(), dets4[i].top(), dets4[i].right(), dets4[i].bottom())

    for d in dets:
        if d.left() < 0:
            left = d.left() + 15
        else:
            left = d.left()
        print "d.left: {0}, d.top: {1}, d.right: {2}, d.bottom: {3}\n".format(left, d.top(), d.right(), d.bottom())
        cv2.rectangle(padded_image, (left, d.top()), (d.right(), d.bottom()), (0, 0, 255), 3)

    print cv2.imwrite("/data/yonatan/linked_to_web/dress_detector_testing2.jpg", padded_image)

    # if not dresses["are_dresses"]:
    #     print "didn't find any dresses"
    #     return None
    #
    # height, width, channels = full_image.shape
    #
    # for i in range(0, len(dresses['dresses'])):
    #
    #     x, y, w, h = dresses['dresses'][i]
    #
    #     if x > width or x + w > width or y > height or y + h > height:
    #         print "\ndress out of image boundaries\n"
    #         return None
    #
    #
    # # if full_image.shape[0] - (y + h) >= 5 * h:
    # #     cv2.rectangle(full_image, (x, y + h), (x + w, y + (6 * h)), (0, 255, 0), 3)
    # #
    # # if full_image.shape[0] - (y + h) >= 6 * h:
    # #     cv2.rectangle(full_image, (x, y + h), (x + w, y + (7 * h)), (0, 0, 255), 3)
    # #
    # # if full_image.shape[0] - (y + h) >= 7 * h:
    # #     cv2.rectangle(full_image, (x, y + h), (x + w, y + (8 * h)), (0, 130, 130), 3)
    # #
    # # if full_image.shape[0] - (y + h) >= 8 * h:
    # #     cv2.rectangle(full_image, (x, y + h), (x + w, y + (9 * h)), (130, 0, 130), 3)
    #
    # cv2.rectangle(full_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    #
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(full_image,'{:.3f}'.format(dresses['scores'][i]),(int(x), int(y + 18)), font, 1,(0,255,0),2,cv2.LINE_AA)
    #
    # print cv2.imwrite("/data/yonatan/linked_to_web/dress_testing.jpg", full_image)
