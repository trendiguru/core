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

# ## style ##
# MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_152_style/ResNet-152-deploy.prototxt"
# PRETRAINED = "/home/yonatan/style_classifier/resnet152_caffemodels_7_11_16/caffe_resnet152_snapshot_style_6_categories_iter_2500.caffemodel"

## collar ##
MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_152_collar_type/ResNet-152-deploy.prototxt"
PRETRAINED = "/home/yonatan/collar_classifier/resnet152_caffemodels_6_11_16/caffe_resnet152_snapshot_collar_9_categories_iter_2500.caffemodel"

# caffe.set_device(int(sys.argv[1]))
caffe.set_device(3)

caffe.set_mode_gpu()
image_dims = [224, 224]
mean, input_scale = np.array([104.0, 116.7, 122.7]), None
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


def distance(v1, v2):
    if len(v1) != 8 or len(v2) != 8:
        print "length of v1 or v2 is not 8!"
        return None
    return np.linalg.norm(v1 - v2)


def theDetector(url_or_np_array):

    label = ''

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

    #max_result = max(predictions[0])

    max_result_index = np.argmax(predictions[0])

    predict_label = int(max_result_index)

    # ## style ##
    # if predict_label == 0:
    #     label = 'casual'
    # elif predict_label == 1:
    #     label = 'prom'
    # elif predict_label == 2:
    #     label = 'tuxedos_and_suits'
    # elif predict_label == 3:
    #     label = 'bride_dress'
    # elif predict_label == 4:
    #     label = 'active'
    # elif predict_label == 5:
    #     label = 'swim'

    ## collar ##
    if predict_label == 0:
        label = 'crew_neck'
    elif predict_label == 1:
        label = 'scoop_neck'
    elif predict_label == 2:
        label = 'v_neck'
    elif predict_label == 3:
        label = 'deep_v_neck'
    elif predict_label == 4:
        label = 'Henley_t_shirts'
    elif predict_label == 5:
        label = 'polo_collar'
    elif predict_label == 6:
        label = 'tie_neck'
    elif predict_label == 7:
        label = 'turtleneck'
    elif predict_label == 8:
        label = 'Hooded_T_Shirt'


    print "label: {0}".format(label)

    print "prediction: {0}".format(predictions[0])
