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


MODLE_FILE = "/home/yonatan/trendi/yonatan/categories_of_irrelevancy/Alexnet_1000_deploy.prototxt"
PRETRAINED = "/home/yonatan/bvlc_alexnet.caffemodel"
caffe.set_mode_gpu()
image_dims = [227, 227]
mean, input_scale = np.array([120, 120, 120]), None
channel_swap = [2, 1, 0]
raw_scale = 255.0

# Make classifier.
classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
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


#def theDetector(image):
def theDetector(url_or_np_array):

    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        full_image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        full_image = url_or_np_array
    else:
        return None

    img_for_caffe = [cv2_image_to_caffe(full_image)]
    #face_for_caffe = [caffe.io.load_image(face_image)]

    if img_for_caffe is None:
        return None

    # Classify.
    start = time.time()
    predictions = classifier.predict(img_for_caffe)
    print("Done in %.2f s." % (time.time() - start))

    print 'maximum guess value: {0}, located in index: {1}'.format(max(predictions[0]), np.argmax(predictions[0]))



