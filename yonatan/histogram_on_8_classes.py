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

array_strapless = np.zeros(8)
array_spaghetti_straps = np.zeros(8)
array_regular_straps = np.zeros(8)
array_sleeveless = np.zeros(8)
array_cap_sleeve = np.zeros(8)
array_short_sleeve = np.zeros(8)
array_midi_sleeve = np.zeros(8)
array_long_sleeve = np.zeros(8)

all_predictions = np.zeros(8)

text_file = open("db_dress_sleeve_test.txt", "r")

MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_50_dress_sleeve/ResNet-50-deploy.prototxt"
PRETRAINED = "/home/yonatan/resnet50_caffemodels/caffe_resnet50_snapshot_50_sgd_iter_10000.caffemodel"
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

    true_label = int(path[1])
    predict_label = int(max_result_index)

    if predict_label == true_label:
        array_success_with_plus_minus_category = np.append(array_success_with_plus_minus_category, max_result)
        array_success_without = np.append(array_success_without, max_result)
    elif predict_label == 0 and true_label == 1:
        array_success_with_plus_minus_category = np.append(array_success_with_plus_minus_category, max_result)
        array_failure_without = np.append(array_failure_without, max_result)
    elif predict_label == 7 and true_label == 6:
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
        array_strapless = np.vstack((array_strapless, predictions[0]))
    elif true_label == 1:
        array_spaghetti_straps = np.vstack((array_spaghetti_straps, predictions[0]))
    elif true_label == 2:
        array_regular_straps = np.vstack((array_regular_straps, predictions[0]))
    elif true_label == 3:
        array_sleeveless = np.vstack((array_sleeveless, predictions[0]))
    elif true_label == 4:
        array_cap_sleeve = np.vstack((array_cap_sleeve, predictions[0]))
    elif true_label == 5:
        array_short_sleeve = np.vstack((array_short_sleeve, predictions[0]))
    elif true_label == 6:
        array_midi_sleeve = np.vstack((array_midi_sleeve, predictions[0]))
    elif true_label == 7:
        array_long_sleeve = np.vstack((array_long_sleeve, predictions[0]))

    print counter
    #print predictions

    all_predictions = np.vstack((all_predictions, predictions[0]))

# pop the first vector - which is [0 0 0 0 0 0 0 0 0]
all_predictions = all_predictions[1:]

mean_vector = mean_vector / counter

for i in range(1, counter):
    max_result_index = np.argmax(all_predictions[i])
    variance_vector += np.square(all_predictions[i] - mean_vector)

variance_vector = variance_vector / counter

array_strapless = array_strapless[1:]
array_spaghetti_straps = array_spaghetti_straps[1:]
array_regular_straps = array_regular_straps[1:]
array_sleeveless = array_sleeveless[1:]
array_cap_sleeve = array_cap_sleeve[1:]
array_short_sleeve = array_short_sleeve[1:]
array_midi_sleeve = array_midi_sleeve[1:]
array_long_sleeve = array_long_sleeve[1:]

np.savez(array_strapless, array_strapless)
np.save(array_spaghetti_straps, array_spaghetti_straps)
np.save(array_regular_straps, array_regular_straps)
np.save(array_sleeveless, array_sleeveless)
np.save(array_cap_sleeve, array_cap_sleeve)
np.save(array_short_sleeve, array_short_sleeve)
np.save(array_midi_sleeve, array_midi_sleeve)
np.save(array_long_sleeve, array_long_sleeve)


# the order of the coralations being printed is:
# 0 and 1, 0 and 2, ...
# 1 and 2, 1 and 3, ...
# 2 and 3, 2 and 4, ...
# ...
print "\ncorrelation between:\n 0 and 1, 0 and 2, ...\n1 and 2, 1 and 3, ...\n2 and 3, 2 and 4, ...\n"
for comb in combinations([array_strapless, array_spaghetti_straps,array_regular_straps,
                          array_sleeveless, array_cap_sleeve, array_short_sleeve,
                          array_midi_sleeve, array_long_sleeve], 2):
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

print "mean array_strapless: {0}\nmean array_spaghetti: {1}\n" \
      "mean array_regular: {2}\nmean array_sleeveless: {3}\n" \
      "mean array_cap_sleeve: {4}\nmean array_short_sleeve: {5}\n" \
      "mean array_midi_sleeve: {6}\nmean array_long_sleeve: {7}\n".format(
      np.mean(array_strapless, 0), np.mean(array_spaghetti_straps, 0), np.mean(array_regular_straps, 0), np.mean(array_sleeveless, 0),
      np.mean(array_cap_sleeve, 0), np.mean(array_short_sleeve, 0), np.mean(array_midi_sleeve, 0), np.mean(array_long_sleeve, 0))

print "variance array_strapless: {0}\nvariance array_spaghetti: {1}\n" \
      "variance array_regular: {2}\nvariance array_sleeveless: {3}\n" \
      "variance array_cap_sleeve: {4}\nvariance array_short_sleeve: {5}\n" \
      "variance array_midi_sleeve: {6}\nvariance array_long_sleeve: {7}\n".format(
      np.var(array_strapless, 0), np.var(array_spaghetti_straps, 0), np.var(array_regular_straps, 0), np.var(array_sleeveless, 0),
      np.var(array_cap_sleeve, 0), np.var(array_short_sleeve, 0), np.var(array_midi_sleeve, 0), np.var(array_long_sleeve, 0))

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

histogram.savefig('db_dresses_histogram_iter_5000.png')
