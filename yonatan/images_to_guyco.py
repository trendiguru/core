#!/usr/bin/env python

import numpy as np
import skimage.io
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
import pymongo
import argparse
import yonatan_constants

# if i run this function on braini2:
# db = constants.db

# if i run this function on brainik80a:
# first run on shell: ssh -f -N -L 27017:mongodb1-instance-1:27017 -L 6379:redis1-redis-1-vm:6379 root@extremeli.trendi.guru
# then:
db = pymongo.MongoClient().mydb


# dress length #
dress_length_jeremy_dict = {
    'mini_length': [db.yonatan_dresses.find(
        {'user_name': 'jeremy', 'dress_length': ['true', 'false', 'false', 'false', 'false', 'false']}), 0],
    'above_knee': [db.yonatan_dresses.find(
        {'user_name': 'jeremy', 'dress_length': ['false', 'true', 'false', 'false', 'false', 'false']}), 0],
    'knee_length': [db.yonatan_dresses.find(
        {'user_name': 'jeremy', 'dress_length': ['false', 'false', 'true', 'false', 'false', 'false']}), 1],
    'tea_length': [db.yonatan_dresses.find(
        {'user_name': 'jeremy', 'dress_length': ['false', 'false', 'false', 'true', 'false', 'false']}), 2],
    'ankle_length': [db.yonatan_dresses.find(
        {'user_name': 'jeremy', 'dress_length': ['false', 'false', 'false', 'false', 'true', 'false']}), 3],
    'floor_length': [db.yonatan_dresses.find(
        {'user_name': 'jeremy', 'dress_length': ['false', 'false', 'false', 'false', 'false', 'true']}), 3]
}


def hand_picked():
    hand_picked_file = open("hand_picked_file.txt", "w")
    hand_picked_labeled_file = open("hand_picked_labeled_file.txt", "w")

    counter = 0
    # num_of_each_category = 0

    for key, value in dress_length_jeremy_dict.iteritems():
        print "starting " + str(key)
        for i in range(1, value[0].count()):
            # if i > num_of_each_category:
            #   break
            hand_picked_file.write(str(value[0][i]['images']['XLarge']) + "\n")
            hand_picked_labeled_file.write(str(value[0][i]['images']['XLarge']) + " " + str(value[1]) + "\n")

            counter += 1
            print counter

# hand_picked()


def labeled():
    labeled_images_file = open("labeled_images_file.txt", "w")
    labeled_images_with_label_file = open("labeled_images_with_label_file.txt", "w")

    counter = 0
    num_of_each_category = 500

    for key, value in yonatan_constants.dress_length_dict.iteritems():
        print "starting " + str(key)
        for i in range(1, value[0].count()):
            if i > num_of_each_category:
                break
            labeled_images_file.write(str(value[0][i]['images']['XLarge']) + "\n")
            labeled_images_with_label_file.write(str(value[0][i]['images']['XLarge']) + " " + str(value[1]) + "\n")

            counter += 1
            print counter

# labeled()


def not_catalog_images():
    not_catalog_images_file = open("not_catalog_images_file.txt", "w")

    counter = 0

    # only images with one person that have a dress
    live_dress_images_cursor = db.images.find({'num_of_people': 1, 'people.items.category': {'$in': ['dress']}})

    # somtimes there's more than one url, so i'm taking only the first
    for doc in live_dress_images_cursor:
        #not_catalog_images_file.write(str(doc['image_urls'][0]) + "\n")

        #got some
        url = doc['image_urls'][0].encode('ascii', 'ignore').decode('ascii')
        not_catalog_images_file.write(url + "\n")

        counter += 1
        print counter

# not_catalog_images()


# - 100 hand picked - i got 170 (the rest isn't good enough), it's not 'n' from each category.
# i got 2 files: with and without label (didn't know if you want to give them the answer or not):
# "hand_picked_file.txt"   and  "hand_picked_labeled_file.txt"
#
# - 3000 labeled - same: with and without label:
# "labeled_images_file.txt"   and   "labeled_images_with_label_file.txt"
#
# - 10,000 from not catalog images - so i took only images with one person (and of course with a dress), so there's only 4555 that's qualified.
# i also saw sometimes there's more than one link to an image, but (from what i checked) it's the same image in all of them, so i only took the first link.
# "not_catalog_images_file.txt"