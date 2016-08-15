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
import yonatan_classifier


MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_50_dress_sleeve/ResNet-50-deploy.prototxt"
PRETRAINED = "/home/yonatan/resnet50_caffemodels/caffe_resnet50_snapshot_50_sgd_iter_5000.caffemodel"
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


def theDetector(url_or_np_array):

    print "Starting the genderism!"
    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        response = requests.get(url_or_np_array)  # download
        full_image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    elif type(url_or_np_array) == np.ndarray:
        full_image = url_or_np_array
    else:
        return None

    if full_image is None:
        print "not a good image"
        return None

    image_for_caffe = [cv2_image_to_caffe(full_image)]

    if image_for_caffe is None:
        return None

    # Classify.
    start = time.time()
    predictions = classifier.predict(image_for_caffe)
    print("Done in %.2f s." % (time.time() - start))

    return predictions[0]


def distance(v1, v2):
    if len(v1) != 8 or len(v2) != 8:
        print "length of v1 or v2 is not 8!"
        return None
    return np.linalg.norm(v1 - v2)
    #
    # #max_result = max(predictions[0])
    #
    # max_result_index = np.argmax(predictions[0])
    #
    # predict_label = int(max_result_index)
    #
    # if predict_label == 0:
    #     return 'strapless'
    # elif predict_label == 1:
    #     return 'spaghetti_straps'
    # elif predict_label == 2:
    #     return 'regular_straps'
    # elif predict_label == 3:
    #     return 'sleeveless'
    # elif predict_label == 4:
    #     return 'cap_sleeve'
    # elif predict_label == 5:
    #     return 'short_sleeve'
    # elif predict_label == 6:
    #     return 'midi_sleeve'
    # elif predict_label == 7:
    #     return 'long_sleeve'
    #
    # #print predictions[0][predict_label]

