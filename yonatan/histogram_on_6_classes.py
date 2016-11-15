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

array_mini_length = np.zeros(6)
array_above_knee = np.zeros(6)
array_knee_length = np.zeros(6)
array_tea_length = np.zeros(6)
array_ankle_length = np.zeros(6)
array_floor_length = np.zeros(6)

all_predictions = np.zeros(6)

# ## dress length ##
# text_file = open("db_dress_length_test.txt", "r")
#
# MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_50_dress_length/ResNet-50-deploy.prototxt"
# PRETRAINED = "/home/yonatan/resnet50_caffemodels/caffe_resnet50_snapshot_dress_length_3k_images_with_people_iter_5000.caffemodel"

## style ##
text_file = open("/home/yonatan/style_classifier/style_second_try/style_images/style_test_list.txt", "r")

MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_152_style/ResNet-152-deploy.prototxt"
PRETRAINED = "/home/yonatan/style_classifier/style_second_try/resnet152_caffemodels_8_11_16/caffe_resnet152_snapshot_style_5_categories_iter_2500.caffemodel"

# caffe.set_device(int(sys.argv[1]))
caffe.set_device(3)

caffe.set_mode_gpu()
image_dims = [224, 224]
mean, input_scale = np.array([120, 120, 120]), None
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

#counter_99_percent = 0
#counter_97_percent = 0
#counter_95_percent = 0
#counter_90_percent = 0

#failure_above_98_percent = 0

mean_vector = 0
variance_vector = 0


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

    mean_vector += predictions[0]

    mini_length_predict = predictions[0][0]
    above_knee_predict = predictions[0][1]
    knee_length_predict = predictions[0][2]
    tea_length_predict = predictions[0][3]
    ankle_length_predict = predictions[0][4]
    floor_length_predict = predictions[0][5]

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
    elif predict_label == 4 and true_label == 3:
        array_success_with_plus_minus_category = np.append(array_success_with_plus_minus_category, max_result)
        array_failure_without = np.append(array_failure_without, max_result)
    elif predict_label == (true_label + 1) or predict_label == (true_label - 1):
        array_success_with_plus_minus_category = np.append(array_success_with_plus_minus_category, max_result)
        array_failure_without = np.append(array_failure_without, max_result)
    else:
        array_failure_with_plus_minus_category = np.append(array_failure_with_plus_minus_category, max_result)
        array_failure_without = np.append(array_failure_without, max_result)
        print max_result

    if true_label == 0:
        array_mini_length = np.vstack((array_mini_length, predictions[0]))
    elif true_label == 1:
        array_above_knee = np.vstack((array_above_knee, predictions[0]))
    elif true_label == 2:
        array_knee_length = np.vstack((array_knee_length, predictions[0]))
    elif true_label == 3:
        array_tea_length = np.vstack((array_tea_length, predictions[0]))
    elif true_label == 4:
        array_ankle_length = np.vstack((array_ankle_length, predictions[0]))
    elif true_label == 5:
        array_floor_length = np.vstack((array_floor_length, predictions[0]))

    print counter
    #print predictions

    all_predictions = np.vstack((all_predictions, predictions[0]))

# pop the first vector - which is [0 0 0 0 0 0]
all_predictions = all_predictions[1:]

mean_vector = mean_vector / counter

for i in range(1, counter):
    max_result_index = np.argmax(all_predictions[i])
    variance_vector += np.square(all_predictions[i] - mean_vector)

variance_vector = variance_vector / counter

array_mini_length = array_mini_length[1:]
array_above_knee = array_above_knee[1:]
array_knee_length = array_knee_length[1:]
array_tea_length = array_tea_length[1:]
array_ankle_length = array_ankle_length[1:]
array_floor_length = array_floor_length[1:]

np.save('array_mini_length.npy', array_mini_length)
np.save('array_above_knee.npy', array_above_knee)
np.save('array_knee_length.npy', array_knee_length)
np.save('array_tea_length.npy', array_tea_length)
np.save('array_ankle_length.npy', array_ankle_length)
np.save('array_floor_length.npy', array_floor_length)

# to load the saved arrays: #
# np.load('array_mini_length.npy')
# np.load('array_above_knee.npy')
# np.load('array_knee_length.npy')
# np.load('array_tea_length.npy')
# np.load('array_ankle_length.npy')
# np.load('array_floor_length.npy')


# the order of the coralations being printed is:
# 0 and 1, 0 and 2, ...
# 1 and 2, 1 and 3, ...
# 2 and 3, 2 and 4, ...
# ...
print "\ncorrelation between:\n0 and 1, 0 and 2, ...\n1 and 2, 1 and 3, ...\n2 and 3, 2 and 4, ...\n"
for comb in combinations([array_mini_length, array_above_knee,array_knee_length,
                          array_tea_length, array_ankle_length, array_floor_length], 2):
    print np.min([np.linalg.norm(a-b) for a,b in product(*comb)])


success_with = len(array_success_with_plus_minus_category)
failure_with = len(array_failure_with_plus_minus_category)

success_without = len(array_success_without)
failure_without = len(array_failure_without)

if success_with == 0 or failure_with == 0:
    print "wrong!"
else:
    print '\naccuracy percent with +-category: {0}'.format(float(success_with) / (success_with + failure_with))
    print 'accuracy percent without: {0}\n'.format(float(success_without) / (success_without + failure_without))

#print "mean vector: {0}".format(mean_vector)

print "mean array_mini_length: {0}\nmean array_spaghetti: {1}\n" \
      "mean array_regular: {2}\nmean array_tea_length: {3}\n" \
      "mean array_ankle_length: {4}\nmean array_floor_length: {5}\n".format(
      np.mean(array_mini_length, 0), np.mean(array_above_knee, 0), np.mean(array_knee_length, 0), np.mean(array_tea_length, 0),
      np.mean(array_ankle_length, 0), np.mean(array_floor_length, 0))

print "variance array_mini_length: {0}\nvariance array_spaghetti: {1}\n" \
      "variance array_regular: {2}\nvariance array_tea_length: {3}\n" \
      "variance array_ankle_length: {4}\nvariance array_floor_length: {5}\n".format(
      np.var(array_mini_length, 0), np.var(array_above_knee, 0), np.var(array_knee_length, 0), np.var(array_tea_length, 0),
      np.var(array_ankle_length, 0), np.var(array_floor_length, 0))

#print "variance vector: {0}".format(variance_vector)

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
