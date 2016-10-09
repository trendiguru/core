#!/usr/bin/env python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil
import yonatan_constants
from .. import background_removal, utils, constants
import sys


def create_txt_files(argv):

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
                                                                                     'dress_length',
                                                                                     'men_shirt_sleeve',
                                                                                     'pants_length',
                                                                                     'women_shirt_sleeve')
        return

    train_dir_path = '/home/yonatan/db_' + args.input_file + '_train_set/'
    cv_dir_path = '/home/yonatan/db_' + args.input_file + '_cv_set/'
    test_dir_path = '/home/yonatan/db_' + args.input_file + '_test_set/'

    if os.path.isdir(train_dir_path) and os.path.isdir(cv_dir_path) and os.path.isdir(test_dir_path):
        print "all good"
    else:
        print "no training set directory"
        return

    sets = {'train', 'cv', 'test'}

    train_text_file = open("db_" + args.input_file + "_train.txt", "w")
    cv_text_file = open("db_" + args.input_file + "_cv.txt", "w")
    test_text_file = open("db_" + args.input_file + "_test.txt", "w")

    counter = 0

    for set in sets:
        dir_path = '/home/yonatan/db_' + args.input_file + '_' + set + '_set'

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                for key, value in dictionary.iteritems():
                    if key in file:
                        if set == 'train':
                            train_text_file.write(root + "/" + file + " " + str(value[1]) + "\n")
                        elif set == 'cv':
                            cv_text_file.write(root + "/" + file + " " + str(value[1]) + "\n")
                        elif set == 'test':
                            test_text_file.write(root + "/" + file + " " + str(value[1]) + "\n")
                        break

                counter += 1
                print counter


def create_txt_files_no_mongo():

    train_dir_path = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_train_set'
    cv_dir_path = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_cv_set'
    test_dir_path = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_test_set'

    train_text_file = open("/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_train.txt", "w")
    cv_text_file = open("/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_cv.txt", "w")
    test_text_file = open("/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_test.txt", "w")

    counter = 0

    sets = {'train', 'cv', 'test'}

    for set in sets:
        dir_path = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_' + set + '_set'

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if "maxi" in file:
                    label = 2
                elif "midi" in file:
                    label = 1
                elif "mini" in file:
                    label = 0

                if set == 'train':
                    train_text_file.write(root + "/" + file + " " + str(label) + "\n")
                elif set == 'cv':
                    cv_text_file.write(root + "/" + file + " " + str(label) + "\n")
                elif set == 'test':
                    test_text_file.write(root + "/" + file + " " + str(label) + "\n")

                counter += 1
                print counter


def create_txt_files_by_adding_from_different_directories():

    train_text_file = open("/home/yonatan/faces_stuff/55k_face_train_list.txt", "a")
    cv_text_file = open("/home/yonatan/faces_stuff/55k_face_cv_list.txt", "a")
    test_text_file = open("/home/yonatan/faces_stuff/55k_face_test_list.txt", "a")

    counter = 0

    # man: 1, woman: 0
    genders = {'man', 'woman'}
    for gender in genders:

        if gender == 'man':
            label = "1"
        else:
            label = "0"

        dir_path = '/home/yonatan/faces_stuff/uniq_faces/' + gender
        for root, dirs, files in os.walk(dir_path):
            file_count = len(files)

            counter = 0

            print file_count

            counter_train = file_count * 0.9
            counter_cv = file_count * 0.05
            counter_test = file_count * 0.05

            for file in files:

                counter += 1
                print counter

                if counter < counter_train:
                    train_text_file.write(root + "/" + file + " " + label + "\n")
                elif counter >= counter_train and counter < counter_train + counter_cv:
                    cv_text_file.write(root + "/" + file + " " + label + "\n")
                elif counter >= counter_train + counter_cv and counter < counter_train + counter_cv + counter_test:
                    test_text_file.write(root + "/" + file + " " + label + "\n")
                else:
                    print "DONE"
                    break

            print 'counter_train = {0}, counter_cv = {1}, counter_test = {2}, counter = {3}'.format(counter_train, counter_cv,
                                                                                                    counter_test, counter)

if __name__ == '__main__':
    # create_txt_files(sys.argv)
    # create_txt_files_no_mongo()
    create_txt_files_by_adding_from_different_directories()
