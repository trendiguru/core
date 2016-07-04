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
import pymongo

db = constants.db

sleeve_dict = {
'strapless' : [db.yonatan_dresses.find({'sleeve_length': ['true', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}), 0],
'spaghetti_straps' : [db.yonatan_dresses.find({'sleeve_length': ['false', 'true', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}), 1],
'straps' : [db.yonatan_dresses.find({'sleeve_length': ['false', 'false', 'true', 'false', 'false', 'false', 'false', 'false', 'false']}), 2],
'sleeveless' : [db.yonatan_dresses.find({'sleeve_length': ['false', 'false', 'false', 'true', 'false', 'false', 'false', 'false', 'false']}), 3],
'cap_sleeve' : [db.yonatan_dresses.find({'sleeve_length': ['false', 'false', 'false', 'false', 'true', 'false', 'false', 'false', 'false']}), 4],
'short_sleeve' : [db.yonatan_dresses.find({'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'true', 'false', 'false', 'false']}), 5],
'midi_sleeve' : [db.yonatan_dresses.find({'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'true', 'false', 'false']}), 6],
'long_sleeve' : [db.yonatan_dresses.find({'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'true', 'false']}), 7]
# 'asymmetry' : [db.yonatan_dresses.find({'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'true']}), 8]
}


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


num_of_each_category = 900

width = 150
height = 300


for key, value in sleeve_dict.iteritems():
    text_file = open("850_dresses_" + key + "_list.txt", "w")
    for i in range(1, value[0].count()):
        if i > num_of_each_category:
            break

        link_to_image = value[0][i]['images']['XLarge']

        dress_image = url_to_image(link_to_image)
        if dress_image is None:
            continue

        # Resize it.
        resized_image = cv2.resize(dress_image, (width, height))

        image_file_name = key + '_dress-' + str(i) + '.jpg'

        print i

        cv2.imwrite(os.path.join('/home/yonatan/db_' + key + 'dresses', image_file_name), resized_image)
        text_file.write('/home/yonatan/yonatan/db_' + key + 'dresses/' + image_file_name + ' ' + str(value[1]) + '\n')

    text_file.flush()
