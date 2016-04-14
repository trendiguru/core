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


female_count = 0
for root, dirs, files in os.walk(mypath_female):
    for file in files:
        if file.startswith("face-"):
            gender_detector.genderator(file)
            female_count += 1
print ("female_count: %d" % (female_count))