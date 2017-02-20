#!/usr/bin/env python

import os
import sys
import glob
import numpy as np

import dlib
from skimage import io
import cv2


path = "/data/jeremy/image_dbs/pixlevel_images/pixlevel_fullsize_labels_v3"

dress_pixel_value = 13  # 13 is the pixel value represents a dress
dress_counter = 0
dress_list = []

for root, dirs, files in os.walk(path):

    for file in files:

        if not file.endswith(".png"):
            print "not a png file"
            continue

        labeled_image = cv2.imread(os.path.join(root, file))

        is_dress = np.any(labeled_image == dress_pixel_value)

        if is_dress:
            dress_counter += 1
            dress_list.append(file)
            print "found dress!"

            dress_and_zeros = np.where(labeled_image != dress_pixel_value, 0, labeled_image)

        else:
            print "no dress!"

print "number of dresses: {0}".format(dress_counter)



