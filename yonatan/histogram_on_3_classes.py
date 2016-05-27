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

text_file = open("dresses_test.txt", "r")

counter = 0

MODLE_FILE = "/home/yonatan/trendi/yonatan/Alexnet_deploy_for_dresses.prototxt"
PRETRAINED = "/home/yonatan/caffe_alexnet_train_on_74250_dresses_iter_45000.caffemodel"
caffe.set_mode_gpu()
image_dims = [256, 256]
mean, input_scale = np.array([120, 120, 120]), None
#mean, input_scale = None, None
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
guessed_mini_instead_midi = 0
guessed_maxi_instead_midi = 0
guessed_midi_instead_mini = 0
guessed_maxi_instead_mini = 0
guessed_midi_instead_maxi = 0
guessed_mini_instead_maxi = 0

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
    elif (midi_predict > mini_predict) and (midi_predict > maxi_predict) and (path[1] == '2'):
        array_failure = np.append(array_failure, midi_predict)
        print predictions
        guessed_midi_instead_maxi += 1
    elif (mini_predict > midi_predict) and (mini_predict > maxi_predict) and (path[1] == '2'):
        array_failure = np.append(array_failure, mini_predict)
        print predictions
        guessed_mini_instead_maxi += 1
    print counter

print 'guessed_mini_instead_midi {0}'.format(guessed_mini_instead_midi)
print 'guessed_maxi_instead_midi {0}'.format(guessed_maxi_instead_midi)
print 'guessed_midi_instead_mini {0}'.format(guessed_midi_instead_mini)
print 'guessed_maxi_instead_mini {0}'.format(guessed_maxi_instead_mini)
print 'guessed_midi_instead_maxi {0}'.format(guessed_midi_instead_maxi)
print 'guessed_mini_instead_maxi {0}'.format(guessed_mini_instead_maxi)

histogram = plt.figure(1)

bins = np.linspace(-1000, 1000, 50)

plt.hist(array_success, alpha=0.5, label='array_success')
plt.legend()

plt.hist(array_failure, alpha=0.5, label='array_failure')
plt.legend()

histogram.savefig('75000_train_dresses_histogram.png')