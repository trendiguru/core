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
from itertools import combinations, product

array_success_with_plus_minus_category = np.array([])
array_failure_with_plus_minus_category = np.array([])
array_success_without = np.array([])
array_failure_without = np.array([])

## style ##
text_file = open("/home/yonatan/collar_classifier/collar_images/collar_test_list.txt", "r")

MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_152_collar_type/ResNet-152-deploy.prototxt"
PRETRAINED = "/home/yonatan/collar_classifier/resnet152_caffemodels_4_12_16/caffe_resnet152_snapshot_collar_10_categories_iter_2500.caffemodel"

# caffe.set_device(int(sys.argv[1]))
caffe.set_device(3)

caffe.set_mode_gpu()
image_dims = [224, 224]
mean, input_scale = np.array([104.0, 116.7, 122.7]), None
#mean, input_scale = None, None
#channel_swap = None
channel_swap = [2, 1, 0]
raw_scale = 255.0
ext = 'jpg'

# Make classifier.
#classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
classifier = yonatan_classifier.Classifier(MODLE_FILE, PRETRAINED,
                              image_dims=image_dims, mean=mean,
                              input_scale=input_scale, raw_scale=raw_scale,
                              channel_swap=channel_swap)

counter = 0

success_counter = 0
failure_counter = 0

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

    max_result = max(predictions[0])

    max_result_index = np.argmax(predictions[0])

    true_label = int(path[1])
    predict_label = int(max_result_index)

    if predict_label == true_label:
        array_success_with_plus_minus_category = np.append(array_success_with_plus_minus_category, max_result)
        array_success_without = np.append(array_success_without, max_result)
    elif predict_label == 0 and true_label == 1:
        array_success_with_plus_minus_category = np.append(array_success_with_plus_minus_category, max_result)
        array_failure_without = np.append(array_failure_without, max_result)
    elif predict_label == 9 and true_label == 8:
        array_success_with_plus_minus_category = np.append(array_success_with_plus_minus_category, max_result)
        array_failure_without = np.append(array_failure_without, max_result)
    elif predict_label == (true_label + 1) or predict_label == (true_label - 1):
        array_success_with_plus_minus_category = np.append(array_success_with_plus_minus_category, max_result)
        array_failure_without = np.append(array_failure_without, max_result)
    else:
        array_failure_with_plus_minus_category = np.append(array_failure_with_plus_minus_category, max_result)
        array_failure_without = np.append(array_failure_without, max_result)
        print max_result

    print counter
    #print predictions

success_with = len(array_success_with_plus_minus_category)
failure_with = len(array_failure_with_plus_minus_category)

success_without = len(array_success_without)
failure_without = len(array_failure_without)

if success_with == 0 or failure_with == 0:
    print "wrong!"
else:
    print '\naccuracy percent with +-category: {0}'.format(float(success_with) / (success_with + failure_with))
    print 'accuracy percent without: {0}\n'.format(float(success_without) / (success_without + failure_without))

histogram = plt.figure(1)

plt.hist(array_success_with_plus_minus_category, bins=100, range=(0.9, 1), color='blue', label='array_success_with_plus_minus_category')
plt.legend()

plt.hist(array_failure_with_plus_minus_category, bins=100, range=(0.9, 1), color='red', label='array_failure_with_plus_minus_category')
plt.legend()

plt.hist(array_success_without, bins=100, range=(0.9, 1), color='green', label='array_success_without')
plt.legend()

plt.hist(array_failure_without, bins=100, range=(0.9, 1), color='pink', label='array_failure_without')
plt.legend()

histogram.savefig('db_dress_length_histogram_iter_5000_dress_with_person.png')
