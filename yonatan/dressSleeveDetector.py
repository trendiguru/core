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


MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_50_dress_sleeve/ResNet-50-deploy.prototxt"
PRETRAINED = "/home/yonatan/resnet50_caffemodels/caffe_resnet50_snapshot_50_sgd_iter_5000.caffemodel"
caffe.set_mode_gpu()
image_dims = [224, 224]
mean, input_scale = np.array([120, 120, 120]), None
channel_swap = [2, 1, 0]
raw_scale = 255.0

# Make classifier.
classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
                              image_dims=image_dims, mean=mean,
                              input_scale=input_scale, raw_scale=raw_scale,
                              channel_swap=channel_swap)

print "Done initializing!"

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

    #height, width, channels = full_image.shape

    #x, y, w, h = face_coordinates

    #if x > width or x + w > width or y > height or y + h > height:
    #    return None

    #face_image = full_image[y: y + h, x: x + w]

    face_for_caffe = [cv2_image_to_caffe(full_image)]
    #face_for_caffe = [caffe.io.load_image(face_image)]

    if face_for_caffe is None:
        return None

    # Classify.
    start = time.time()
    predictions = classifier.predict(face_for_caffe)
    print("Done in %.2f s." % (time.time() - start))

    #max_result = max(predictions[0])

    max_result_index = np.argmax(predictions[0])

    predict_label = int(max_result_index)

    if predict_label == 0:
        return 'strapless'
    elif predict_label == 1:
        return 'spaghetti_straps'
    elif predict_label == 2:
        return 'regular_straps'
    elif predict_label == 3:
        return 'sleeveless'
    elif predict_label == 4:
        return 'cap_sleeve'
    elif predict_label == 5:
        return 'short_sleeve'
    elif predict_label == 6:
        return 'midi_sleeve'
    elif predict_label == 7:
        return 'long_sleeve'

    print predictions[0][predict_label]

