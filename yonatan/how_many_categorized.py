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


def check_how_many(category_dict, yonatan_category_db):

    dress_sleeve_dict = category_dict

    sum_of_all = sum(value[0].count() for key, value in dress_sleeve_dict.iteritems())
    sum_of_all_already_seen = yonatan_category_db.count({'already_seen_dress_sleeve': True})

    deleted = sum_of_all_already_seen - sum_of_all

    for key, value in dress_sleeve_dict.iteritems():
        category_value = value[0].count()
        print '{0}: {1}, percent: {2}%'.format(key, category_value,
                                               int(round(float(category_value) / sum_of_all, 2) * 100))
    print 'sum of all: {0}'.format(sum_of_all)
    print 'deleted: {0}'.format(deleted)


def how_many(argv):

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="the argument should be one of those:"
             "\ndress_sleeve\ndress_length\nmen_shirt_sleeve\npants_length\nwomen_shirt_sleeve\ncheck_all"
    )

    db = constants.db

    args = parser.parse_args()

    # dress sleeve #
    if args.input_file == 'dress_sleeve':
        check_how_many(yonatan_constants.dress_sleeve_dict, db.yonatan_dresses)

    # dress length #
    elif args.input_file == 'dress_length':
        check_how_many(yonatan_constants.dress_length_dict, db.yonatan_dresses)

    # men shirt sleeve #
    elif args.input_file == 'men_shirt_sleeve':
        check_how_many(yonatan_constants.men_shirt_sleeve_dict, db.yonatan_men_shirts)

    # pants length #
    elif args.input_file == 'pants_length':
        check_how_many(yonatan_constants.pants_length_dict, db.yonatan_pants)

    # women shirt sleeve #
    elif args.input_file == 'women_shirt_sleeve':
        check_how_many(yonatan_constants.women_shirt_sleeve_dict, db.yonatan_women_shirts)

    elif args.input_file == 'check_all':
        check_how_many(yonatan_constants.dress_sleeve_dict, db.yonatan_dresses)
        check_how_many(yonatan_constants.dress_length_dict, db.yonatan_dresses)
        check_how_many(yonatan_constants.men_shirt_sleeve_dict, db.yonatan_men_shirts)
        check_how_many(yonatan_constants.pants_length_dict, db.yonatan_pants)
        check_how_many(yonatan_constants.women_shirt_sleeve_dict, db.yonatan_women_shirts)

    else:
        print "wrong input!"
        print "the argument should be one of those:\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}".format('dress_sleeve',
                                                                                     'dress_length', 'men_shirt_sleeve', 'pants_length', 'women_shirt_sleeve', 'check_all')
        return

if __name__ == '__main__':
    how_many(sys.argv)
