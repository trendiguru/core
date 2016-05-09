#!/usr/bin/env python

__author__ = 'jeremy, yonatan guy'

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


MODEL_FILE = "/home/yonatan/neuro_doorman/deploy.prototxt"
PRETRAINED = "/home/yonatan/neuro_doorman/_iter_8078.caffemodel"
caffe.set_mode_gpu()
image_dims = [227, 227]
mean = np.array([107, 117, 123])
# the training was without mean subtraction
#mean = None
input_scale = None
channel_swap = [2, 1, 0]
raw_scale = 255.0

# Make classifier.
classifier = caffe.Classifier(MODEL_FILE, PRETRAINED,
                              image_dims=image_dims, mean=mean,
                              input_scale=input_scale, raw_scale=raw_scale,
                              channel_swap=channel_swap)


def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format

    if url.count('jpg') > 1:
        return None

    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if image.size == 0:
        return None
    new_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return new_image


def theDetector(url_or_np_array):
    '''
    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    else:
        return None

    image_for_caffe = [cv2_image_to_caffe(image)]
    #image_for_caffe = [caffe.io.load_image(image)]
    '''

    image_for_caffe = [caffe.io.load_image('/home/yonatan/Gingerbread_House_Essex_CT.jpg')]

    if image_for_caffe is None:
        return None

    # Classify.
    start = time.time()
    predictions = classifier.predict(image_for_caffe)

    print("predictions %s Done in %.2f s." % (str(predictions), (time.time() - start)))

    if predictions[0][1] > predictions[0][0]:
        print predictions[0][1]
        # relevant
        return True
    else:
        print predictions[0][0]
        # irrelevant
        return False
