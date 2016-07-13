#!/usr/bin/env python

import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
import os
import caffe
from .. import background_removal, utils, constants
from ..utils import imutils
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

dress_sleeve_dict = {
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

dress_length_dict = {
    'mini_length': db.yonatan_dresses.count({'dress_length': ['true', 'false', 'false', 'false', 'false', 'false']}),
    'above_knee': db.yonatan_dresses.count({'dress_length': ['false', 'true', 'false', 'false', 'false', 'false']}),
    'knee_length': db.yonatan_dresses.count({'dress_length': ['false', 'false', 'true', 'false', 'false', 'false']}),
    'tea_length': db.yonatan_dresses.count({'dress_length': ['false', 'false', 'false', 'true', 'false', 'false']}),
    'ankle_length': db.yonatan_dresses.count({'dress_length': ['false', 'false', 'false', 'false', 'true', 'false']}),
    'floor_length': db.yonatan_dresses.count({'dress_length': ['false', 'false', 'false', 'false', 'false', 'true']})
}

men_shirt_sleeve_dict = {
    'straps': db.yonatan_men_shirts.count({'shirt_sleeve_length': ['true', 'false', 'false', 'false', 'false']}),
    'sleeveless': db.yonatan_men_shirts.count({'shirt_sleeve_length': ['false', 'true', 'false', 'false', 'false']}),
    'short_sleeve': db.yonatan_men_shirts.count({'shirt_sleeve_length': ['false', 'false', 'true', 'false', 'false']}),
    'midi_sleeve': db.yonatan_men_shirts.count({'shirt_sleeve_length': ['false', 'false', 'false', 'true', 'false']}),
    'long_sleeve': db.yonatan_men_shirts.count({'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'true']})
}

pants_length_dict = {
    'bermuda': db.yonatan_pants.count({'pants_length': ['true', 'false', 'false', 'false']}),
    'knee': db.yonatan_pants.count({'pants_length': ['false', 'true', 'false', 'false']}),
    'capri': db.yonatan_pants.count({'pants_length': ['false', 'false', 'true', 'false']}),
    'floor': db.yonatan_pants.count({'pants_length': ['false', 'false', 'false', 'true']})
}

women_shirt_sleeve_dict = {
    'strapless': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['true', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}),
    'spaghetti_straps': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'true', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}),
    'straps': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'true', 'false', 'false', 'false', 'false', 'false', 'false']}),
    'sleeveless': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'true', 'false', 'false', 'false', 'false', 'false']}),
    'cap_sleeve': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'true', 'false', 'false', 'false', 'false']}),
    'short_sleeve': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'true', 'false', 'false', 'false']}),
    'midi_sleeve': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'true', 'false', 'false']}),
    'long_sleeve': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'true', 'false']}),
    'asymmetry': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'true']})
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


for key, value in dress_sleeve_dict.iteritems():
    #text_file = open("all_dresses_" + key + "_list.txt", "w")
    for i in range(1, value[0].count()):
        #if i > num_of_each_category:
         #   break

        link_to_image = value[0][i]['images']['XLarge']

        fresh_image = url_to_image(link_to_image)
        if fresh_image is None:
            continue

        # Resize it.
        #resized_image = cv2.resize(fresh_image, (width, height))
        resized_image = imutils.resize_keep_aspect(fresh_image, output_size = (256, 256))

        image_file_name = key + '_dress-' + str(i) + '.jpg'

        print i

        cv2.imwrite(os.path.join('/home/yonatan/resized_db_' + key + '_dresses', image_file_name), resized_image)
        #text_file.write('/home/yonatan/resized_db_' + key + '_dresses/' + image_file_name + ' ' + str(value[1]) + '\n')

        print '/home/yonatan/resized_db_' + key + '_dresses/'

    #text_file.flush()
