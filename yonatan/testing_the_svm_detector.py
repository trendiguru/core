#!/usr/bin/env python

import numpy as np
import cv2
import requests
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

error_counter = 0
face_image_counter = 0
found_dress_counter = 0
no_dress_counter = 0

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

    if test_object_detector.detect(full_image, found_dress_counter):
        found_dress_counter += 1
    else:
        no_dress_counter += 1

    total = error_counter + face_image_counter + found_dress_counter + no_dress_counter
    added_images = found_dress_counter / float(total)
    print "error_counter = {0}, face_image_counter = {1}, found_dress_counter = {2}, no_dress_counter = {3}\ntotal = {4}, added_images = {5}".format(error_counter, face_image_counter, found_dress_counter, no_dress_counter, total, added_images)



results_text_file.write("error_counter = " + str(error_counter) + ", face_image_counter = " + str(face_image_counter) + ", found_dress_counter = " + str(found_dress_counter) + ", no_dress_counter = " + str(no_dress_counter) + "\ntotal = " + str(total) + ", added_images = " + str(added_images))


results_text_file.flush()
