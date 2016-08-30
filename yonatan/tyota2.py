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

# test_text_file = open("55k_test_set_new.txt", "w")
#
# counter = 0
#
#
# dir_path = '/home/yonatan/55k_test_set'
#
# for root, dirs, files in os.walk(dir_path):
#     for file in files:
#         resized_image = imutils.resize_keep_aspect(file, output_size=(224, 224))
#
#         test_text_file.write("/home/yonatan/resized_test_dir/" + file + " 0\n")
#
#         cv2.imwrite(os.path.join('/home/yonatan/resized_test_dir', file), resized_image)
#
#         counter += 1
#         print counter
#
#
# def divide_data():

counter = 0
counter_train = 0
counter_cv = 0
counter_test = 0

train_dir_path = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_train_set/'
cv_dir_path = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_cv_set/'
test_dir_path = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_test_set/'

source_dir = '/home/yonatan/dress_length_from_my_pc'

for root, dirs, files in os.walk(source_dir):
    file_count = len(files)

    counter_train = file_count * 0.9
    counter_cv = file_count * 0.05
    counter_test = file_count * 0.05

    for file in files:

        old_file_location = source_dir + '/' + file

        input_file = os.path.expanduser(old_file_location)

        print input_file
        print type.input_file

        resized_image = imutils.resize_keep_aspect(input_file, output_size = (224, 224))



    #     if counter < counter_train:
    #         new_file_location = train_dir_path + file
    #         os.rename(old_file_location, new_file_location)
    #         counter += 1
    #     elif counter >= counter_train and counter < counter_train + counter_cv:
    #         new_file_location = cv_dir_path + file
    #         os.rename(old_file_location, new_file_location)
    #         counter += 1
    #     elif counter >= counter_train + counter_cv and counter < counter_train + counter_cv + counter_test:
    #         new_file_location = test_dir_path + file
    #         os.rename(old_file_location, new_file_location)
    #         counter += 1
    #     else:
    #         print counter
    #         break
    #
    # print 'counter_train = {0}, counter_cv = {1}, counter_test = {2}, counter = {3}'.format(counter_train, counter_cv, counter_test, counter)
