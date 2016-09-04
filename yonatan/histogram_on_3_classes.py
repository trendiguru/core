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


array_success = np.array([])
array_failure = np.array([])

counter = 0

text_file = open("dress_length_3_labels_sets/dress_length_3_labels_test.txt", "r")

MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_50_dress_length/ResNet-50-deploy.prototxt"
PRETRAINED = "/home/yonatan/resnet50_caffemodels/caffe_resnet50_snapshot_dress_length_3_categories_iter_5000.caffemodel"
caffe.set_mode_gpu()
image_dims = [224, 224]
mean, input_scale = np.array([120, 120, 120]), None
#mean, input_scale = None, None
#channel_swap = None
channel_swap = [2, 1, 0]
raw_scale = 255.0
ext = 'jpg'

# Make classifier.
classifier = yonatan_classifier.Classifier(MODLE_FILE, PRETRAINED,
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

    mini_predict = predictions[0][0]
    midi_predict = predictions[0][1]
    maxi_predict = predictions[0][2]

    max_result = max(predictions[0])

    if 0.90 <= max_result < 0.95:
        counter_90_percent += 1
        counter_95_percent += 1
        counter_97_percent += 1
        counter_99_percent += 1
    elif 0.95 <= max_result < 0.97:
        counter_90_percent += 1
        counter_95_percent += 1
    elif 0.97 <= max_result < 0.99:
        counter_90_percent += 1
        counter_95_percent += 1
        counter_97_percent += 1
    elif max_result >= 0.99:
        counter_90_percent += 1
        counter_95_percent += 1
        counter_97_percent += 1
        counter_99_percent += 1

    print mini_predict
    print midi_predict
    print maxi_predict

    #if the gender_detector is right
    if (mini_predict > midi_predict) and (mini_predict > maxi_predict) and (path[1] == '0'):
        array_success = np.append(array_success, mini_predict)
    elif (midi_predict > mini_predict) and (midi_predict > maxi_predict) and (path[1] == '1'):
        array_success = np.append(array_success, midi_predict)
    elif (maxi_predict > mini_predict) and (maxi_predict > midi_predict) and (path[1] == '2'):
        array_success = np.append(array_success, maxi_predict)
    # if the gender_detector is wrong
    elif (mini_predict > midi_predict) and (mini_predict > maxi_predict) and (path[1] == '1'):
        array_failure = np.append(array_failure, mini_predict)
        print predictions
        guessed_mini_instead_midi += 1
    elif (maxi_predict > midi_predict) and (maxi_predict > mini_predict) and (path[1] == '1'):
        array_failure = np.append(array_failure, maxi_predict)
        print predictions
        guessed_maxi_instead_midi += 1
    elif (midi_predict > mini_predict) and (midi_predict > maxi_predict) and (path[1] == '0'):
        array_failure = np.append(array_failure, midi_predict)
        print predictions
        guessed_midi_instead_mini += 1
    elif (maxi_predict > midi_predict) and (maxi_predict > mini_predict) and (path[1] == '0'):
        array_failure = np.append(array_failure, maxi_predict)
        print predictions
        guessed_maxi_instead_mini += 1
        #img = cv2.imread(input_file)
        #cv2.imshow('guessed_maxi_instead_mini', img)
        #cv2.waitKey(0)
    elif (midi_predict > mini_predict) and (midi_predict > maxi_predict) and (path[1] == '2'):
        array_failure = np.append(array_failure, midi_predict)
        print predictions
        guessed_midi_instead_maxi += 1
    elif (mini_predict > midi_predict) and (mini_predict > maxi_predict) and (path[1] == '2'):
        array_failure = np.append(array_failure, mini_predict)
        print predictions
        guessed_mini_instead_maxi += 1
        #img = cv2.imread(input_file)
        #cv2.imshow('guessed_mini_instead_maxi', img)
        #cv2.waitKey(0)
    print counter

print 'guessed_mini_instead_midi {0}'.format(guessed_mini_instead_midi)
print 'guessed_maxi_instead_midi {0}'.format(guessed_maxi_instead_midi)
print 'guessed_midi_instead_mini {0}'.format(guessed_midi_instead_mini)
print 'guessed_maxi_instead_mini {0}'.format(guessed_maxi_instead_mini)
print 'guessed_midi_instead_maxi {0}'.format(guessed_midi_instead_maxi)
print 'guessed_mini_instead_maxi {0}'.format(guessed_mini_instead_maxi)

print 'results equal or above 90%: {0}'.format(float(counter_90_percent) / counter)
print 'results equal or above 95%: {0}'.format(float(counter_95_percent) / counter)
print 'results equal or above 97%: {0}'.format(float(counter_97_percent) / counter)
print 'results equal or above 99%: {0}'.format(float(counter_99_percent) / counter)

success = len(array_success)
failure = len(array_failure)
if success == 0 or failure == 0:
    print "wrong!"
else:
    print 'accuracy percent: {0}'.format(float(success) / (success + failure))

for cell in array_failure:
    if cell >= 0.98:
        failure_above_98_percent += 1

print 'failure_above_98_percent: {0}'.format(float(failure_above_98_percent) / failure)

histogram = plt.figure(1)

plt.hist(array_success, bins=100, range=(0, 1), color='blue', label='array_success')
plt.legend()

plt.hist(array_failure, bins=100, range=(0, 1), color='red', label='array_failure')
plt.legend()

histogram.savefig('67000_train_dresses_histogram_iter_20000_only_dresses_on_models.png')
