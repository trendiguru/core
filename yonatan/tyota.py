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


mypath_female = '/home/yonatan/test_set/female'
array_boys = np.array([])
array_girls = np.array([])

female_count = 0
text_file = open("face_testing.txt", "w")
for root, dirs, files in os.walk(mypath_female):
    for file in files:
        if file.startswith("face-"):
            predictions = gender_detector.genderator(root + "/" + file)
            np.append(array_boys, predictions[0][0])
            np.append(array_girls, predictions[0][1])
            female_count += 1
            print predictions
            print predictions[0][0]
            print array_boys
print ("female_count: %d" % (female_count))

print array_boys

histogram=plt.figure(1)

bins = np.linspace(-1000, 1000, 50)

plt.hist(array_boys, alpha=0.5, label='array_boys')
plt.hist(array_girls, alpha=0.5, label='array_girls')
plt.legend()

histogram.savefig('test_image.png')

'''
fig = plt.figure(1)
plt.hist(gaussian_numbers, bins=20, histtype='stepfilled', normed=True, color='b', label='Gaussian')
plt.hist(uniform_numbers, bins=20, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='Uniform')
plt.title("Gaussian/Uniform Histogram")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.legend()
plt.show()
# Save the figure to the current path
fig.savefig('test_image.png')
'''