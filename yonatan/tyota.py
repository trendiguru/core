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
print ("female_count: %d" % (female_count))

np.histogram(array_boys)
np.histogram(array_girls)