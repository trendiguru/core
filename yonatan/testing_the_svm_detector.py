#!/usr/bin/env python

import numpy as np
import os
import caffe
from trendi import background_removal, Utils, constants
import cv2
import sys
import argparse
import glob
import time
import urllib
import requests
import dlib
import pymongo

from trendi.yonatan import test_object_detector, preparing_data_from_db


# if i run this function on braini2:
# db = constants.db

# if i run this function on brainik80a:
# db = pymongo.MongoClient().mydb

# if in_docker:
db = pymongo.MongoClient('localhost', port=27017).mydb

# else:
# db = constants.db

dict = db.irrelevant_images_distinct.find()

results_text_file = open("/data/yonatan/yonatan_files/trendi/yonatan/_results_irrelevant_db_images.txt", "w")

counter = 0
error_counter = 0
face_image_counter = 0

for i in range(1, dict.count()):

    # if i % 10 != 0:
    #     continue

    link_to_image = dict[i]['image_urls'][0]

    # check if i get a url (= string) or np.ndarray
    if isinstance(link_to_image, basestring):
        # full_image = url_to_image(url_or_np_array)
        try:
            response = requests.get(link_to_image, timeout=10)  # download
            full_image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
        except:
            print "couldn't open link"
            error_counter += 1
            continue
    elif type(link_to_image) == np.ndarray:
        full_image = link_to_image
    else:
        error_counter += 1
        continue

    # checks if the face coordinates are inside the image
    if full_image is None:
        print "not a good image"
        error_counter += 1
        continue

    # # if there's a head, cut it off
    faces = preparing_data_from_db.find_face_dlib(full_image)

    if faces["are_faces"]:
        face_image_counter += 1
        continue

    test_object_detector.detect(full_image, counter)




    print counter


    counter += 1

results_text_file.write(working_path + '/' + image_file_name + ' ' + str(value[1]) + '\n')

print "number of bad images: {0}".format(i - counter)

# text_file.flush()