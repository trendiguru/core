#!/usr/bin/env python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil
import yonatan_constants
from .. import background_removal, utils, constants
import sys
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
import caffe
from ..utils import imutils
import argparse
import glob
import time
import skimage
import urllib
from PIL import Image
import pymongo
import dlib

counter = 0
counter_train = 0
counter_cv = 0
counter_test = 0

train_dir_path = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_train_set'
cv_dir_path = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_cv_set'
test_dir_path = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_test_set'

source_dir = '/home/yonatan/dress_length_from_my_pc'


for root, dirs, files in os.walk(source_dir):
    file_count = len(files)

    counter = 0

    print file_count

    counter_train = file_count * 0.9
    counter_cv = file_count * 0.05
    counter_test = file_count * 0.05

    for file in files:

        old_file_location = root + '/' + file

        #input_file = os.path.expanduser(old_file_location)

        #img = cv2.imread(old_file_location)


        img = cv2.imread(old_file_location, 0)
        height, width = img.shape[:2]
        print width, height
        print counter

        resized_image = imutils.resize_keep_aspect(old_file_location, output_size = (224, 224))

        if counter < counter_train:
            #new_file_location = train_dir_path + file
            cv2.imwrite(os.path.join(train_dir_path, file), resized_image)
            #os.rename(old_file_location, new_file_location)
            counter += 1
        elif counter >= counter_train and counter < counter_train + counter_cv:
            #new_file_location = cv_dir_path + file
            cv2.imwrite(os.path.join(cv_dir_path, file), resized_image)
            #os.rename(old_file_location, new_file_location)
            counter += 1
        elif counter >= counter_train + counter_cv and counter < counter_train + counter_cv + counter_test:
            #new_file_location = test_dir_path + file
            cv2.imwrite(os.path.join(test_dir_path, file), resized_image)
            #os.rename(old_file_location, new_file_location)
            counter += 1
        else:
            print counter
            break

    print 'counter_train = {0}, counter_cv = {1}, counter_test = {2}, counter = {3}'.format(counter_train, counter_cv, counter_test, counter)


# sets = {'train', 'cv', 'test'}
#
# for set in sets:
#
#     source = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_' + set + '_set/'
#
#     imutils.resize_keep_aspect_dir(source, source, False, (224, 224))
