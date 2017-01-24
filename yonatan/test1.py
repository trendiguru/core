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
# from ..utils import imutils
# import yonatan_classifier
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt


detector = dlib.get_frontal_face_detector()

### print the deep fashion 'fabric' categories ###
# import yonatan_constants
#
# dict = yonatan_constants.attribute_type_dict
#
# counter = 0
#
# for key, value in dict.iteritems():
#
#     if value[1] == "fabric":
#         counter += 1
#         print value[0]
#
# print "\ncounter: {0}".format(counter)


def find_face_dlib(image, max_num_of_faces=10):
    start = time.time()
    faces = detector(image, 1)
    print("Done in %.3f s." % (time.time() - start))

    print faces
    faces = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(faces)]
    if not len(faces):
        return {'are_faces': False, 'faces': []}
    #final_faces = choose_faces(image, faces, max_num_of_faces)
    print "number of faces: {0}\n".format(len(faces))
    print faces[0]
    print faces[1]
    print faces[2]
    print faces[3]
    return {'are_faces': len(faces) > 0, 'faces': faces}


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

    #resized_image = imutils.resize_keep_aspect(full_image, output_size=(124, 124))

    # faces = background_removal.find_face_dlib(full_image)

    faces = find_face_dlib(full_image, 10)

    if not faces["are_faces"]:
        print "didn't find any faces"
        return None

    # print faces["faces"][0] # just checking if the face that found seems in the right place

    height, width, channels = full_image.shape

    for i in range(0, len(faces['faces'])):
        print "faces['faces'][i]: {0}".format(faces['faces'][i])
        x, y, w, h = faces['faces'][i]

        if x > width or x + w > width or y > height or y + h > height:
            print "\nface out of image boundaries\n"
            return None

        # face_image = full_image[y: y + h, x: x + w]

        cv2.rectangle(full_image, (x, y), (x + w, y + h), (255, 0, 0), 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(full_image,'{:.3f}'.format(score),(int(x), int(y + 18)), font, 1,(0,255,0),2,cv2.LINE_AA)

    print cv2.imwrite("/data/yonatan/linked_to_web/face_testing.jpg", full_image)
