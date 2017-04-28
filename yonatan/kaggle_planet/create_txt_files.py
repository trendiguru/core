#!/usr/bin/env python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil
import sys
import random


train_csv = open('/data/yonatan/kaggle_planet_data/train.csv')

train_text_file = open("/data/yonatan/kaggle_planet_data/kaggke_planet_multilabel_train.txt", "w")
cv_text_file = open("/data/yonatan/kaggle_planet_data/kaggke_planet_multilabel_cv.txt", "w")

path = '/data/yonatan/kaggle_planet_data/train-jpg/'

lines_in_train_csv = train_csv.readlines()

all_labels = ['partly_cloudy', 'haze', 'clear',
              'primary', 'water', 'habitation', 'agriculture', 'cultivation', 'road', 'bare_ground',
              'slash_burn', 'selective_logging', 'blooming', 'conventional_mine', 'artisinal_mine', 'blow_down']

labels = '0'

for line in lines_in_train_csv:

    labels = '0'

    image_name_and_labels = line.split(',')
    image_name = image_name_and_labels[0]
    labels_names = image_name_and_labels[1]

    if 'cloudy' in labels_names:
        labels = "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
    else:
        for i in range(0, len(all_labels)):
            if all_labels[i] in labels_names:
                labels += " 1"
            else:
                labels += " 0"

    print path + image_name_and_labels[0] + " " + labels

    train_text_file.write(path + image_name_and_labels[0] + " " + labels + "\n")

train_text_file.close()



# def create_txt_files_no_mongo():
#     # from the times when i first made txt files for all categories,
#     # #now i make train, cv, test directly from all categories
#
#     train_dir_path = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_train_set'
#     cv_dir_path = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_cv_set'
#     test_dir_path = '/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_test_set'
#
#     train_text_file = open("/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_train.txt", "w")
#     cv_text_file = open("/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_cv.txt", "w")
#     test_text_file = open("/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_test.txt", "w")
#                                                                                                     counter)
#
#
# def create_txt_files_from_different_directories(feature_name, source_dir, labels):
#     train_text_file = open(source_dir + "/" + feature_name + "_train_list.txt", "w")
#     cv_text_file = open(source_dir + "/" + feature_name + "_cv_list.txt", "w")
#     test_text_file = open(source_dir + "/" + feature_name + "_test_list.txt", "w")
#
#     error_counter = 0
#
#     for key, value in labels.iteritems():
#         label_dir = source_dir + "/" + key
#         label = str(value)
#
#         if os.path.isdir(label_dir):
#             if not os.listdir(label_dir):
#                 print '\nfolder is empty: ' + key
#                 break
#         else:
#             print '\nfolder doesn\'t exist: ' + key
#             break
#
#         for root, dirs, files in os.walk(label_dir):
#             file_count = len(files)
#
#             counter = 0
#
#             print file_count
#
#             counter_train = file_count * 0.9
#             counter_cv = file_count * 0.05
#             counter_test = file_count * 0.05
#
#             for file in files:
#
#                 if "._" in file:
#                     continue
#
#                 counter += 1
#                 print counter
#
#                 try:
#                     if counter < counter_train:
#                         train_text_file.write(root + "/" + file + " " + label + "\n")
#                     elif counter >= counter_train and counter < counter_train + counter_cv:
#                         cv_text_file.write(root + "/" + file + " " + label + "\n")
#                     elif counter >= counter_train + counter_cv and counter < counter_train + counter_cv + counter_test:
#                         test_text_file.write(root + "/" + file + " " + label + "\n")
#                     else:
#                         print "DONE" + value
#                         break
#
#                 except:
#                     print "something ain't good"
#                     error_counter += 1
#                     continue
#
#             print 'counter_train = {0}, counter_cv = {1}, counter_test = {2}, counter = {3}, key = {4}, error_counter = {5}'.format(
#                 counter_train, counter_cv,
#                 counter_test, counter, key, error_counter)
#
#     train_text_file.close()
#     cv_text_file.close()
#     test_text_file.close()
#
#     train_lines = open(source_dir + "/" + feature_name + "_train_list.txt").readlines()
#     cv_lines = open(source_dir + "/" + feature_name + "_cv_list.txt").readlines()
#     test_lines = open(source_dir + "/" + feature_name + "_test_list.txt").readlines()
#     random.shuffle(train_lines)
#     random.shuffle(cv_lines)
#     random.shuffle(test_lines)
#     open(source_dir + "/" + feature_name + "_train_list.txt", "w").writelines(train_lines)
#     open(source_dir + "/" + feature_name + "_cv_list.txt", "w").writelines(cv_lines)
#     open(source_dir + "/" + feature_name + "_test_list.txt", "w").writelines(test_lines)
#
#
# def edit_existing_gender_txt_files():
#     train_txt_file = open("/home/yonatan/faces_stuff/55k_face_train_list.txt", "r+")
#     cv_txt_file = open("/home/yonatan/faces_stuff/55k_face_cv_list.txt", "r+")
#     test_txt_file = open("/home/yonatan/faces_stuff/55k_face_test_list.txt", "r+")
#
#     copy_train_file = train_txt_file.readlines()
#     copy_cv_file = cv_txt_file.readlines()
#     copy_test_file = test_txt_file.readlines()
#
#     train_txt_file.seek(0)
#     cv_txt_file.seek(0)
#     test_txt_file.seek(0)
#
#     for line in copy_train_file:
#         if line.startswith("/home/yonatan/faces_stuff"):
#             train_txt_file.write(line)
#         else:
#             words = line.split('/')
#             # for example:
#             # ['', 'home', 'yonatan', '55k_faces_cv_set_224', 'resized_face-9921.jpg 1']
#             new_line = "/home/yonatan/faces_stuff/" + words[3] + '/' + words[4]
#             train_txt_file.write(new_line)
#
#     for line in copy_cv_file:
#         if line.startswith("/home/yonatan/faces_stuff"):
#             cv_txt_file.write(line)
#         else:
#             words = line.split('/')
#             # for example:
#             # ['', 'home', 'yonatan', '55k_faces_cv_set_224', 'resized_face-9921.jpg 1']
#             new_line = "/home/yonatan/faces_stuff/" + words[3] + '/' + words[4]
#             cv_txt_file.write(new_line)
#
#     for line in copy_test_file:
#         if line.startswith("/home/yonatan/faces_stuff"):
#             test_txt_file.write(line)
#         else:
#             words = line.split('/')
#             # for example:
#             # ['', 'home', 'yonatan', '55k_faces_cv_set_224', 'resized_face-9921.jpg 1']
#             new_line = "/home/yonatan/faces_stuff/" + words[3] + '/' + words[4]
#             test_txt_file.write(new_line)
#
#     train_txt_file.truncate()
#     cv_txt_file.truncate()
#     test_txt_file.truncate()
#     train_txt_file.close()
#     cv_txt_file.close()
#     test_txt_file.close()
#
#
# def shuffle_all_lines():
#     train_lines = open('/home/yonatan/faces_stuff/55k_face_train_list.txt').readlines()
#     cv_lines = open('/home/yonatan/faces_stuff/55k_face_cv_list.txt').readlines()
#     test_lines = open('/home/yonatan/faces_stuff/55k_face_test_list.txt').readlines()
#
#     random.shuffle(train_lines)
#     random.shuffle(cv_lines)
#     random.shuffle(test_lines)
#
#     train_text_file = open("/home/yonatan/faces_stuff/55k_face_train_list.txt", "w").writelines(train_lines)
#     cv_text_file = open("/home/yonatan/faces_stuff/55k_face_cv_list.txt", "w").writelines(cv_lines)
#     test_text_file = open("/home/yonatan/faces_stuff/55k_face_test_list.txt", "w").writelines(test_lines)
#
#
# if __name__ == '__main__':
#     # create_txt_files(sys.argv)
#     # create_txt_files_no_mongo()
#     # edit_existing_gender_txt_files()
#     # create_txt_files_by_adding_from_different_directories()
#     # shuffle_all_lines()
#     create_txt_files_from_different_directories()
