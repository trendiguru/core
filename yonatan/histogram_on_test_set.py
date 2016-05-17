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


#path = '/home/yonatan/55k_test_set'
array_success = np.array([])
array_failure = np.array([])

text_file = open("1500_dresses_test_list.txt", "r")

counter = 0

MODLE_FILE = "/home/yonatan/trendi/yonatan/Alexnet_deploy_for_dresses.prototxt"
PRETRAINED = "/home/yonatan/caffe_alexnet_train_on_9000_dresses_iter_10000.caffemodel"
caffe.set_mode_gpu()
image_dims = [100, 200]
mean, input_scale = np.array([120, 120, 120]), None
channel_swap = None
#channel_swap = [2, 1, 0]
raw_scale = 255.0
ext = 'jpg'

# Make classifier.
classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
                              image_dims=image_dims, mean=mean,
                              input_scale=input_scale, raw_scale=raw_scale,
                              channel_swap=channel_swap)

success_counter = 0
failure_counter = 0
guessed_f_instead_m = 0
guessed_m_instead_f = 0

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

    #if the gender_detector is right
    if (predictions[0][0] > predictions[0][1]) and (path[1] == '0'):
        array_success = np.append(array_success, predictions[0][0])
    elif (predictions[0][1] > predictions[0][0]) and (path[1] == '1'):
        array_success = np.append(array_success, predictions[0][1])
    # if the gender_detector is wrong
    elif (predictions[0][0] > predictions[0][1]) and (path[1] == '1'):
        array_failure = np.append(array_failure, predictions[0][0])
        print predictions
        guessed_f_instead_m += 1
    elif (predictions[0][1] > predictions[0][0]) and (path[1] == '0'):
        array_failure = np.append(array_failure, predictions[0][1])
        print predictions
        guessed_m_instead_f += 1

    print counter

print guessed_f_instead_m
print guessed_m_instead_f



histogram=plt.figure(1)

bins = np.linspace(-1000, 1000, 50)

plt.hist(array_success, alpha=0.5, label='array_success')
plt.legend()

plt.hist(array_failure, alpha=0.5, label='array_failure')
plt.legend()

histogram.savefig('9000_train_dresses_histogram.png')