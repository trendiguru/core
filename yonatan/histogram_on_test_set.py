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

text_file = open("55k_face_test_list.txt", "r")

counter = 0

for line in file:
    counter += 1
    #split line to link and label
    words = line.split("/")

    if words == []:
        continue

    file_name = words[3].split()

    predictions = gender_detector.genderator(line)

    #if the gender_detector is right
    if (predictions[0][0] > predictions[0][1]) && (file_name[1] == 0):
        array_success = np.append(array_success, predictions[0][0])
    elif (predictions[0][1] > predictions[0][0]) && (file_name[1] == 1):
        array_success = np.append(array_success, predictions[0][1])
    # if the gender_detector is wrong
    if (predictions[0][0] > predictions[0][1]) & & (file_name[1] == 1):
        array_failure = np.append(array_failure, predictions[0][0])
    elif (predictions[0][1] > predictions[0][0]) & & (file_name[1] == 0):
        array_failure = np.append(array_failure, predictions[0][1])

print counter


histogram=plt.figure(1)

bins = np.linspace(-1000, 1000, 50)

plt.hist(array_success, alpha=0.5, label='array_success')
plt.legend()

plt.hist(array_failure, alpha=0.5, label='array_failure')
plt.legend()

histogram.savefig('imdb_test_image.png')