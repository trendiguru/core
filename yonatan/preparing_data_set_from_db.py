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

    # dress sleeve #
    if args.input_file == 'dress_sleeve':
        dictionary = yonatan_constants.dress_sleeve_dict

    # dress length #
    elif args.input_file == 'dress_length':
        dictionary = yonatan_constants.dress_length_dict

    # men shirt sleeve #
    elif args.input_file == 'men_shirt_sleeve':
        dictionary = yonatan_constants.men_shirt_sleeve_dict

    # pants length #
    elif args.input_file == 'pants_length':
        dictionary = yonatan_constants.pants_length_dict

    # women shirt sleeve #
    elif args.input_file == 'women_shirt_sleeve':
        dictionary = yonatan_constants.women_shirt_sleeve_dict

    # dress_doorman #
    elif args.input_file == 'yonatan_dresses_test':
        dresses = db.yonatan_dresses_test.find()

    else:
        print "wrong input!"
        print "the argument should be one of those:\n{0}\n{1}\n{2}\n{3}\n{4}".format('dress_sleeve', 'dress_length', 'men_shirt_sleeve', 'pants_length', 'women_shirt_sleeve', 'yonatan_dresses_test')
        return

    ## for all but dress_doorman ##
    # for key, value in dictionary.iteritems():
    #
    #     working_path = '/home/yonatan/resized_db_' + args.input_file + '_' + key
    #
    #     if os.path.isdir(working_path):
    #         if not os.listdir(working_path):
    #             print '\nfolder is empty'
    #         else:
    #             print "deleting directory content"
    #             shutil.rmtree(working_path)
    #             os.mkdir(working_path)
    #     else:
    #         print "creating new directory"
    #         os.mkdir(working_path)
    #
    #     #text_file = open("all_dresses_" + key + "_list.txt", "w")
    #     for i in range(1, value[0].count()):
    #         #if i > num_of_each_category:
    #          #   break
    #
    #         link_to_image = value[0][i]['images']['XLarge']

    images = []
    boxes = []

    counter = 0

    for i in range(1, dresses.count()):
        #if i > num_of_each_category:
         #   break

        link_to_image = dresses[i]['images']['XLarge']

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

        # # if there's a head, cut it of
        faces = find_face_dlib(full_image)

        if faces["are_faces"]:
            if len(faces['faces']) == 1:
                x, y, w, h = faces['faces'][0]
                full_image = full_image[y + h:, :]  # Crop the face from the image
                # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
            else:
                continue

        cropped_image = grabCut.grabcut(full_image)


        # Resize it.
        #resized_image = cv2.resize(full_image, (width, height))
        # resized_image = imutils.resize_keep_aspect(full_image, output_size = (224, 224))

        image_file_name = 'dress-' + str(i) + '.jpg'

        line_in_list_images = 'io.imread(/data/dress_detector/images/' + image_file_name
        line_in_list_boxes = '([dlib.rectangle(left=0, top=0, right=' + str(cropped_image.shape[1]) + ', bottom=' + str(cropped_image.shape[0]) + ')])'

        images.append(line_in_list_images)
        boxes.append(line_in_list_boxes)

        print counter

        cv2.imwrite(os.path.join('/data/dress_detector/images', image_file_name), cropped_image)
        #text_file.write(working_path + '/' + image_file_name + ' ' + str(value[1]) + '\n')

        counter += 1


    np.array(images).dump(open('/data/dress_detector/images.npy', 'wb'))
    np.array(boxes).dump(open('/data/dress_detector/boxes.npy', 'wb'))

    print "number of bad images: {0}".format(i - counter)

    #text_file.flush()


if __name__ == '__main__':
    preparing_data_from_db(sys.argv)
