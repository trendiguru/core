#!/usr/bin/env python

import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
import os
import caffe
from .. import background_removal, Utils, constants
import cv2
import sys
import argparse
import glob
import time
import skimage
import urllib
from PIL import Image


def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if image.size == 0:
        print url
        return None
    new_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return new_image


train_set_number = 700
cv_set_number = 100
test_set_number = 100

width = 100
height = 200

dresses = {'mini', 'maxi'}

text_file_train = open("700_dresses_train_list.txt", "w")
text_file_cv = open("100_dresses_cv_list.txt", "w")
text_file_test = open("100_dresses_test_list.txt", "w")

counter = 0

for kind in dresses:
    if kind == 'mini':
        file_for_read = open('mini_900_dresses.txt', 'r')
    else:
        file_for_read = open('maxi_900_dresses.txt', 'r')

    counter = 0

    for line in file_for_read:
        counter += 1
        #split line to link and label
        words = line.split()

        if words == []:
            continue

        dress_image = url_to_image(words[0])
        if dress_image is None:
            continue

        # Resize it.
        resized_image = cv2.resize(dress_image, (width, height))

        image_file_name = 'resized_dress-' + str(counter) + '.jpg'

        print counter

        if counter <= train_set_number:
            cv2.imwrite(os.path.join('/home/yonatan/small_dresses_train_set', image_file_name), resized_image)
            text_file_train.write('/home/yonatan/small_dresses_train_set/' + image_file_name + ' ' + words[1] + '\n')
        elif counter > train_set_number and counter <= train_set_number + cv_set_number:
            cv2.imwrite(os.path.join('/home/yonatan/small_dresses_cv_set', image_file_name), resized_image)
            text_file_cv.write('/home/yonatan/small_dresses_cv_set/' + image_file_name + ' ' + words[1] + '\n')
        elif counter > train_set_number + cv_set_number:
            cv2.imwrite(os.path.join('/home/yonatan/small_dresses_test_set', image_file_name), resized_image)
            text_file_test.write('/home/yonatan/small_dresses_test_set/' + image_file_name + ' ' + words[1] + '\n')

text_file_train.flush()
text_file_cv.flush()
text_file_test.flush()
