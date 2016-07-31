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


path = '/home/yonatan/55k_test_set'
array_boys_success = np.array([])
array_girls_success = np.array([])
array_boys_failure = np.array([])
array_girls_failure = np.array([])

female_count = 0
#text_file = open("face_testing.txt", "w")
for root, dirs, files in os.walk(path):
    for file in files:
        #if file.startswith("face-"):
            predictions = gender_detector.genderator(root + "/" + file)
            if predictions[0][0] > predictions[0][1]:
                array_boys_failure = np.append(array_boys_failure, predictions[0][0])
                array_girls_failure = np.append(array_girls_failure, predictions[0][1])
            else:
                array_boys_success=np.append(array_boys_success, predictions[0][0])
                array_girls_success=np.append(array_girls_success, predictions[0][1])
            female_count += 1
print ("female_count: %d" % (female_count))

histogram=plt.figure(1)

bins = np.linspace(-1000, 1000, 50)

plt.hist(array_boys_success, alpha=0.5, label='array_boys_success')
plt.hist(array_girls_success, alpha=0.5, label='array_girls_success')
plt.legend()

plt.hist(array_boys_failure, alpha=0.5, label='array_boys_failure')
plt.hist(array_girls_failure, alpha=0.5, label='array_girls_failure')
plt.legend()

histogram.savefig('test_image_for_faces.png')
