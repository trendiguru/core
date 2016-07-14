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
import argparse
import shutil
import yonatan_constants


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


def preparing_data_from_db(argv):

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="the argument should be one of those:"
             "\ndress_sleeve\ndress_length\nmen_shirt_sleeve\npants_length\nwomen_shirt_sleeve"
    )

    db = constants.db

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

    else:
        print "wrong input!"
        print "the argument should be one of those:\n{0}\n{1}\n{2}\n{3}\n{4}".format('dress_sleeve',
                                                                                     'dress_length', 'men_shirt_sleeve',                                                                           'pants_length',
                                                                                     'women_shirt_sleeve')
        return


    num_of_each_category = 900

    for key, value in dictionary.iteritems():

        working_path = '/home/yonatan/resized_db_' + args.input_file + '_' + key

        if os.path.isdir(working_path):
            if not os.listdir(working_path):
                print '\nfolder is empty'
            else:
                print "deleting directory content"
                shutil.rmtree(working_path)
                os.mkdir(working_path)
        else:
            print "creating new directory"
            os.mkdir(working_path)

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

            image_file_name = key + '_' + args.input_file + '-' + str(i) + '.jpg'

            print i

            cv2.imwrite(os.path.join(working_path, image_file_name), resized_image)
            #text_file.write(working_path + '/' + image_file_name + ' ' + str(value[1]) + '\n')

            print working_path

        #text_file.flush()


if __name__ == '__main__':
    preparing_data_from_db(sys.argv)
