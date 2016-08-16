#!/usr/bin/env python

import os
import cv2
import sys
from PIL import Image
from .. import background_removal, utils, constants
import numpy as np
import matplotlib.pyplot as plt
import yonatan_constants
import argparse
import shutil
import pymongo


def divide_data(argv):

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="the argument should be one of those:"
             "\ndress_sleeve\ndress_length\nmen_shirt_sleeve\npants_length\nwomen_shirt_sleeve"
    )

    # if i run this function on braini2:
    # db = constants.db

    # if i run this function on brainik80a:
    db = pymongo.MongoClient().mydb

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

    counter = 0
    counter_train = 0
    counter_cv = 0
    counter_test = 0

    train_dir_path = '/home/yonatan/db_' + args.input_file + '_train_set/'
    cv_dir_path = '/home/yonatan/db_' + args.input_file + '_cv_set/'
    test_dir_path = '/home/yonatan/db_' + args.input_file + '_test_set/'

    if os.path.isdir(train_dir_path):
        print "deleting directories content"
        shutil.rmtree(train_dir_path)
        shutil.rmtree(cv_dir_path)
        shutil.rmtree(test_dir_path)
    else:
        print "creating new directories"

    os.mkdir(train_dir_path)
    os.mkdir(cv_dir_path)
    os.mkdir(test_dir_path)

    for key, value in dictionary.iteritems():
        source_dir = '/home/yonatan/resized_db_' + args.input_file + '_' + key

        if os.path.isdir(source_dir):
            if not os.listdir(source_dir):
                print '\nfolder is empty'
                break
        else:
            print '\nfolder doesn\'t exist'
            break

        counter = 0

        for root, dirs, files in os.walk(source_dir):
            file_count = len(files)
            print '/home/yonatan/resized_db_' + args.input_file + '_' + key
            print "count: {0}, type: {1}".format(file_count, type(file_count))

            counter_train = file_count * 0.9
            counter_cv = file_count * 0.05
            counter_test = file_count * 0.05

            for file in files:

                old_file_location = source_dir + '/' + file

                if counter < counter_train:
                    new_file_location = train_dir_path + file
                    os.rename(old_file_location, new_file_location)
                    counter += 1
                elif counter >= counter_train and counter < counter_train + counter_cv:
                    new_file_location = cv_dir_path + file
                    os.rename(old_file_location, new_file_location)
                    counter += 1
                elif counter >= counter_train + counter_cv and counter < counter_train + counter_cv + counter_test:
                    new_file_location = test_dir_path + file
                    os.rename(old_file_location, new_file_location)
                    counter += 1
                else:
                    print counter
                    break

    print 'counter_train = {0}, counter_cv = {1}, counter_test = {2}, counter = {3}'.format(counter_train, counter_cv, counter_test, counter)

if __name__ == '__main__':
    divide_data(sys.argv)
