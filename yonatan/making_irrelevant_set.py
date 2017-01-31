#!/usr/bin/env python

import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
import os
import caffe
# from .. import background_removal, utils, constants
# from ..utils import imutils
import cv2
import sys
import argparse
import glob
import time
import skimage
import urllib
from PIL import Image
import pymongo
import argparse
import shutil
import yonatan_constants
import dlib
import requests
import grabCut

detector = dlib.get_frontal_face_detector()

def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


def find_face_dlib(image, max_num_of_faces=10):
    faces = detector(image, 1)
    faces = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(faces)]
    if not len(faces):
        return {'are_faces': False, 'faces': []}
    #final_faces = choose_faces(image, faces, max_num_of_faces)
    return {'are_faces': len(faces) > 0, 'faces': faces}


def preparing_data_from_db(argv):

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="the argument should be one of those:"
             "\ndress_sleeve\ndress_length\nmen_shirt_sleeve\npants_length\nwomen_shirt_sleeve\nyonatan_dresses_test"
    )

    # if i run this function on braini2:
    # db = constants.db

    # if i run this function on brainik80a:
    # db = pymongo.MongoClient().mydb

    # if in_docker:
    db = pymongo.MongoClient('localhost', port=27017).mydb

    # else:
    # db = constants.db

    args = parser.parse_args()

    # irrelevant #
    if args.input_file == 'irrelevant':
        irrelevant = db.irrelevant_images_distinct.find()

    else:
        print "wrong input!"
        print "the argument should be one of those:\n{0}\n{1}\n{2}\n{3}\n{4}".format('dress_sleeve', 'dress_length', 'men_shirt_sleeve', 'pants_length', 'women_shirt_sleeve', 'yonatan_dresses_test')
        return

    counter = 0

    # irrelevant_text_file = open("/data/irrelevant/irrelevant_db_images.txt", "w")

    for i in range(1, irrelevant.count()):
        #if i > num_of_each_category:
         #   break

        link_to_image = irrelevant[i]['image_urls'][0]

        # check if i get a url (= string) or np.ndarray
        if isinstance(link_to_image, basestring):
            # full_image = url_to_image(url_or_np_array)
            response = requests.get(link_to_image)  # download
            full_image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
        elif type(link_to_image) == np.ndarray:
            full_image = link_to_image
        else:
            continue

        # checks if the face coordinates are inside the image
        if full_image is None:
            print "not a good image"
            continue

        image_file_name = 'irrelevant-' + str(i) + '.jpg'
        path = os.path.join('/data/irrelevant/images', image_file_name)

        print counter

        cv2.imwrite(path, full_image)

        # irrelevant_text_file.write(path + ' 0\n')

        counter += 1


    print "number of bad images: {0}".format(i - counter)

    # irrelevant_text_file.flush()


if __name__ == '__main__':
    preparing_data_from_db(sys.argv)




irrelevant = db.irrelevant_images.find().distinct('image_hash')

test = db.irrelevant_images.aggregate([{"$group" : {'_id' : "$image_hash"}}])
list = list(test)
len(list)


for i in range(0, irrelevant.count()):
    for j in range(0, len(list)):
        if irrelevant[i]['_id'] == list[j]:
            counter += 1
            print counter
