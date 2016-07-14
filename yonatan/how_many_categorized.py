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


def how_many(argv):

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
        dress_sleeve_dict = yonatan_constants.dress_sleeve_dict

        sum_of_all = sum(value[0].count() for key, value in dress_sleeve_dict.iteritems())
        sum_of_all_already_seen = db.yonatan_dresses.count({'already_seen_dress_sleeve': True})

        deleted = sum_of_all_already_seen - sum_of_all

        for key, value in dress_sleeve_dict.iteritems():
            category_value = value[0].count()
            print '{0}: {1}, percent: {2}%'.format(key, category_value, int(round(float(category_value) / sum_of_all, 2) * 100))

    # dress length #
    elif args.input_file == 'dress_length':
        dress_length_dict = yonatan_constants.dress_length_dict

        sum_of_all = sum(value[0].count() for key, value in dress_length_dict.iteritems())
        sum_of_all_already_seen = db.yonatan_dresses.count({'already_seen_dress_length': True})

        deleted = sum_of_all_already_seen - sum_of_all

        for key, value in dress_length_dict.iteritems():
            category_value = value[0].count()
            print '{0}: {1}, percent: {2}%'.format(key, category_value, int(round(float(category_value) / sum_of_all, 2) * 100))

    # men shirt sleeve #
    elif args.input_file == 'men_shirt_sleeve':
        men_shirt_sleeve_dict = yonatan_constants.men_shirt_sleeve_dict

        sum_of_all = sum(value[0].count() for key, value in men_shirt_sleeve_dict.iteritems())
        sum_of_all_already_seen = db.yonatan_men_shirts.count({'already_seen_shirt_sleeve': True})

        deleted = sum_of_all_already_seen - sum_of_all

        for key, value in men_shirt_sleeve_dict.iteritems():
            category_value = value[0].count()
            print '{0}: {1}, percent: {2}%'.format(key, category_value, int(round(float(category_value) / sum_of_all, 2) * 100))

    # pants length #
    elif args.input_file == 'pants_length':
        pants_length_dict = yonatan_constants.pants_length_dict

        sum_of_all = sum(value[0].count() for key, value in pants_length_dict.iteritems())
        sum_of_all_already_seen = db.yonatan_pants.count({'already_seen_pants_length': True})

        deleted = sum_of_all_already_seen - sum_of_all

        for key, value in pants_length_dict.iteritems():
            category_value = value[0].count()
            print '{0}: {1}, percent: {2}%'.format(key, category_value, int(round(float(category_value) / sum_of_all, 2) * 100))

    # women shirt sleeve #
    elif args.input_file == 'women_shirt_sleeve':
        women_shirt_sleeve_dict = yonatan_constants.women_shirt_sleeve_dict

        sum_of_all = sum(value[0].count() for key, value in women_shirt_sleeve_dict.iteritems())
        sum_of_all_already_seen = db.yonatan_women_shirts.count({'already_seen_shirt_sleeve': True})

        deleted = sum_of_all_already_seen - sum_of_all

        for key, value in women_shirt_sleeve_dict.iteritems():
            category_value = value[0].count()
            print '{0}: {1}, percent: {2}%'.format(key, category_value, int(round(float(category_value) / sum_of_all, 2) * 100))

    else:
        print "wrong input!"
        print "the argument should be one of those:\n{0}\n{1}\n{2}\n{3}\n{4}".format('dress_sleeve',
                                                                                     'dress_length', 'men_shirt_sleeve', 'pants_length', 'women_shirt_sleeve')
        return

    print 'sum of all: {0}'.format(sum_of_all)
    print 'deleted: {0}'.format(deleted)

if __name__ == '__main__':
    how_many(sys.argv)
