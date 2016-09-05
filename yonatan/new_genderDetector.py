#!/usr/bin/env python

__author__ = 'yonatan_guy'

import numpy as np
import os
import caffe
import sys
import argparse
import glob
import time
from trendi import background_removal, Utils, constants
import cv2
import urllib
import skimage
import requests
import dlib
from ..utils import imutils
import yonatan_classifier


MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_50_gender_by_face/ResNet-50-deploy.prototxt"
PRETRAINED = "/home/yonatan/caffe_resnet50_snapshot_sgd_genfder_by_face_iter_10000.caffemodel"
caffe.set_mode_gpu()
image_dims = [224, 224]
mean, input_scale = np.array([120, 120, 120]), None
channel_swap = [2, 1, 0]
raw_scale = 255.0

# Make classifier.
classifier = yonatan_classifier.Classifier(MODLE_FILE, PRETRAINED,
                              image_dims=image_dims, mean=mean,
                              input_scale=input_scale, raw_scale=raw_scale,
                              channel_swap=channel_swap)

print "Done initializing!"


def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


def theDetector(url_or_np_array, face_coordinates):

    print "Starting the genderism!"
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

    height, width, channels = full_image.shape

    x, y, w, h = face_coordinates

    if x > width or x + w > width or y > height or y + h > height:
        return None

    face_image = full_image[y: y + h, x: x + w]

    face_for_caffe = [cv2_image_to_caffe(face_image)]
    #face_for_caffe = [caffe.io.load_image(face_image)]

    if face_for_caffe is None:
        return None

    # Classify.
    start = time.time()
    predictions = classifier.predict(face_for_caffe)
    print("Done in %.2f s." % (time.time() - start))

    if predictions[0][1] > 0.7:
        print predictions[0][1]
        return 'Male'
    else:
        print predictions[0][0]
        return 'Female'
