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
            hand_picked_labeled_file.write(str(value[0][i]['images']['XLarge']) + str(value[0]) + "\n")

            counter += 1
            print counter

hand_picked()

# def check_how_many(category_dict, yonatan_category_db, item_type):
#
#     dress_sleeve_dict = category_dict
#
#     sum_of_all = sum(value[0].count() for key, value in dress_sleeve_dict.iteritems())
#     sum_of_all_already_seen = yonatan_category_db.count({'already_seen_' + item_type: True})
#
#     deleted = sum_of_all_already_seen - sum_of_all
#
#     print "\n\n" + item_type + "\n"
#
#     for key, value in dress_sleeve_dict.iteritems():
#         category_value = value[0].count()
#         print '{0}: {1}, percent: {2}%'.format(key, category_value,
#                                                int(round(float(category_value) / sum_of_all, 2) * 100))
#     print 'sum of all: {0}'.format(sum_of_all)
#     print 'deleted: {0}'.format(deleted)
#
#
# def how_many(argv):
#
#     parser = argparse.ArgumentParser()
#     # Required arguments: input and output files.
#     parser.add_argument(
#         "input_file",
#         help="the argument should be one of those:"
#              "\ndress_sleeve\ndress_length\nmen_shirt_sleeve\npants_length\nwomen_shirt_sleeve\ncheck_all"
#     )
#
#     # if i run this function on braini2:
#     # db = constants.db
#
#     # if i run this function on brainik80a:
#     db = pymongo.MongoClient().mydb
#
#     args = parser.parse_args()
#
#     # dress sleeve #
#     if args.input_file == 'dress_sleeve':
#         check_how_many(yonatan_constants.dress_sleeve_dict, db.yonatan_dresses, 'dress_sleeve')
#
#     # dress length #
#     elif args.input_file == 'dress_length':
#         check_how_many(yonatan_constants.dress_length_dict, db.yonatan_dresses, 'dress_length')
#
#     # men shirt sleeve #
#     elif args.input_file == 'men_shirt_sleeve':
#         check_how_many(yonatan_constants.men_shirt_sleeve_dict, db.yonatan_men_shirts, 'shirt_sleeve')
#
#     # pants length #
#     elif args.input_file == 'pants_length':
#         check_how_many(yonatan_constants.pants_length_dict, db.yonatan_pants, 'pants_length')
#
#     # women shirt sleeve #
#     elif args.input_file == 'women_shirt_sleeve':
#         check_how_many(yonatan_constants.women_shirt_sleeve_dict, db.yonatan_women_shirts, 'shirt_sleeve')
#
#     elif args.input_file == 'check_all':
#         check_how_many(yonatan_constants.dress_sleeve_dict, db.yonatan_dresses, 'dress_sleeve')
#         check_how_many(yonatan_constants.dress_length_dict, db.yonatan_dresses, 'dress_length')
#         check_how_many(yonatan_constants.men_shirt_sleeve_dict, db.yonatan_men_shirts, 'shirt_sleeve')
#         check_how_many(yonatan_constants.pants_length_dict, db.yonatan_pants, 'pants_length')
#         check_how_many(yonatan_constants.women_shirt_sleeve_dict, db.yonatan_women_shirts, 'shirt_sleeve')
#
#     else:
#         print "wrong input!"
#         print "the argument should be one of those:\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}".format('dress_sleeve',
#                                                                                      'dress_length', 'men_shirt_sleeve', 'pants_length', 'women_shirt_sleeve', 'check_all')
#         return
#
# if __name__ == '__main__':
#     how_many(sys.argv)
