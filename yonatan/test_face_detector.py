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
import skimage
import requests
import dlib
from ..utils import imutils
# import yonatan_classifier
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt


detector = dlib.get_frontal_face_detector()


def find_face_dlib(image, max_num_of_faces=10):
    start = time.time()

    ## faces, scores, idx = detector.run(image, 1, -1) - gives more results, those that add low confidence percentage ##
    ## faces, scores, idx = detector.run(image, 1, 1) - gives less results, doesn't show the lowest confidence percentage results ##
    ## i can get only the faces locations with: faces = detector(image, 1) ##

    faces, scores, idx = detector.run(image, 1)

    for i, d in enumerate(faces):
        print("Detection {}, score: {}, face_type:{}".format(
            d, scores[i], idx[i]))

    print("Done in %.3f s." % (time.time() - start))

    faces = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(faces)]
    if not len(faces):
        return {'are_faces': False, 'faces': []}
    #final_faces = choose_faces(image, faces, max_num_of_faces)
    print "number of faces: {0}\n".format(len(faces))
    return {'are_faces': len(faces) > 0, 'faces': faces, 'scores': scores}


def theDetector(url_or_np_array):

    print "Starting the face detector testing!"
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

    full_image = imutils.resize_keep_aspect(full_image, output_size=(500, 500))

    # faces = background_removal.find_face_dlib(full_image)

    faces = find_face_dlib(full_image, 10)

    if not faces["are_faces"]:
        print "didn't find any faces"
        return None

    height, width, channels = full_image.shape

    for i in range(0, len(faces['faces'])):

        x, y, w, h = faces['faces'][i]

        if x > width or x + w > width or y > height or y + h > height:
            print "\nface out of image boundaries\n"
            return None


        if faces["are_faces"]:
            if len(faces['faces']) == 1:
                full_image = full_image[y + h:, :]  # Crop the face from the image
                # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
            else:
                continue

        # face_image = full_image[y: y + h, x: x + w]

        # cv2.rectangle(full_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        #
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(full_image,'{:.3f}'.format(faces['scores'][i]),(int(x), int(y + 18)), font, 1,(0,255,0),2,cv2.LINE_AA)

    print cv2.imwrite("/data/yonatan/linked_to_web/face_testing.jpg", full_image)
