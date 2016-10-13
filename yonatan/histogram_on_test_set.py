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
import yonatan_classifier


#path = '/home/yonatan/55k_test_set'
array_success = np.array([])
array_failure = np.array([])

#text_file = open("55k_face_test_list.txt", "r")
text_file = open("/home/yonatan/faces_stuff/55k_face_test_list.txt", "r")

counter = 0
test_flag = 0

MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_50_gender_by_face/ResNet-50-deploy.prototxt"
PRETRAINED = "/home/yonatan/faces_stuff/resnet_genderator_models_09_10_16/caffe_resnet50_snapshot_sgd_gender_by_face_iter_30000.caffemodel"
caffe.set_mode_gpu()
image_dims = [224, 224]
mean, input_scale = np.array([120, 120, 120]), None
#mean, input_scale = None, None
channel_swap = None
# channel_swap = [2, 1, 0]
raw_scale = 255.0
ext = 'jpg'

# Make classifier.
classifier = yonatan_classifier.Classifier(MODLE_FILE, PRETRAINED,
                              image_dims=image_dims, mean=mean,
                              input_scale=input_scale, raw_scale=raw_scale,
                              channel_swap=channel_swap)

success_counter = 0
failure_counter = 0
guessed_f_instead_m = 0
guessed_m_instead_f = 0

error_counter = 0

for line in text_file:
    counter += 1

    # split line to full path and label
    path = line.split()

    if path == []:
        continue

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    input_file = os.path.expanduser(path[0])
    try:
        inputs = [caffe.io.load_image(input_file)]
    except IOError:
        print "cannot identify image file"
        error_counter += 1
        continue
    #inputs = [Utils.get_cv2_img_array(input_file)]

    print("Classifying %d inputs." % len(inputs))

    # Classify.
    start = time.time()
    predictions = classifier.predict(inputs)
    print("Done in %.2f s." % (time.time() - start))
    print predictions[0]

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

    if test_flag:
        if counter > 500:
            print "test_flag is on"
            break

print "guessed_f_instead_m: {}".format(guessed_f_instead_m)
print "guessed_m_instead_f: {}".format(guessed_m_instead_f)

success = len(array_success)
failure = len(array_failure)

if success == 0 or failure == 0:
    print "wrong!"
else:
    print '\naccuracy percent: {0}'.format(float(success) / (success + failure))

print "error_counter: {0}".format(error_counter)

histogram=plt.figure(1)

#bins = np.linspace(-1000, 1000, 50)

plt.hist(array_success, bins=100, range=(0.9, 1), color='green', label='array_success')
plt.legend()

plt.hist(array_failure, bins=100, range=(0.9, 1), color='red', label='array_failure')
plt.legend()

histogram.savefig('new_genderator_histogram.png')