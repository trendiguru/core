#!/usr/bin/env python

import caffe
import numpy as np
from .. import background_removal, Utils, constants
import cv2
import os
import sys
import argparse
import glob
import time
import skimage
from PIL import Image
from . import gender_detector
import random
import matplotlib.pyplot as plt


array_success = np.array([])
array_failure = np.array([])

text_file = open("db_dresses_test.txt", "r")

counter = 0

MODLE_FILE = "/home/yonatan/trendi/yonatan/Alexnet_deploy_for_dresses.prototxt"
PRETRAINED = "/home/yonatan/caffe_alexnet_db_dresses_sleeve_iter_4749.caffemodel"
caffe.set_mode_gpu()
image_dims = [256, 256]
mean, input_scale = np.array([120, 120, 120]), None
#mean, input_scale = None, None
#channel_swap = None
channel_swap = [2, 1, 0]
raw_scale = 255.0
ext = 'jpg'

# Make classifier.
classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
                              image_dims=image_dims, mean=mean,
                              input_scale=input_scale, raw_scale=raw_scale,
                              channel_swap=channel_swap)

success_counter = 0
failure_counter = 0
guessed_mini_instead_midi = 0
guessed_maxi_instead_midi = 0
guessed_midi_instead_mini = 0
guessed_maxi_instead_mini = 0
guessed_midi_instead_maxi = 0
guessed_mini_instead_maxi = 0

counter_99_percent = 0
counter_97_percent = 0
counter_95_percent = 0
counter_90_percent = 0

failure_above_98_percent = 0

#failure_current_result = 0
#success_current_result = 0

for line in text_file:
    counter += 1

    # split line to full path and label
    path = line.split()

    if path == []:
        continue

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    input_file = os.path.expanduser(path[0])
    inputs = [caffe.io.load_image(input_file)]
    #inputs = [Utils.get_cv2_img_array(input_file)]

    print("Classifying %d inputs." % len(inputs))

    # Classify.
    start = time.time()
    predictions = classifier.predict(inputs)
    print("Done in %.2f s." % (time.time() - start))

    strapless_predict = predictions[0][0]
    spaghetti_straps_predict = predictions[0][1]
    straps_predict = predictions[0][2]
    sleeveless_predict = predictions[0][3]
    cap_sleeve_predict = predictions[0][4]
    short_sleeve_predict = predictions[0][5]
    midi_sleeve_predict = predictions[0][6]
    long_sleeve_predict = predictions[0][7]

    max_result = max(predictions[0])

    max_result_index = np.argmax(predictions[0])

    if max_result_index == path[1]:
        array_success = np.append(array_success, max_result)
    else:
        array_failure = np.append(array_failure, max_result)


success = len(array_success)
failure = len(array_failure)
if success == 0 or failure == 0:
    print "wrong!"
else:
    print 'accuracy percent: {0}'.format(float(success) / (success + failure))

histogram = plt.figure(1)

plt.hist(array_success, bins=100, range=(0, 1), color='blue', label='array_success')
plt.legend()

plt.hist(array_failure, bins=100, range=(0, 1), color='red', label='array_failure')
plt.legend()

histogram.savefig('db_dresses_histogram_iter_5000.png')
