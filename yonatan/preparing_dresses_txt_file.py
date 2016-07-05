#!/usr/bin/env python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

sets = {'train', 'cv', 'test'}

train_text_file = open("db_dresses_train.txt", "w")
cv_text_file = open("db_dresses_cv.txt", "w")
test_text_file = open("db_dresses_test.txt", "w")

counter = 0

for set in sets:
    dir_path = '/home/yonatan/dresses_' + set + '_set'

    for root, dirs, files in os.walk(dir_path):
        for file in files:

            sleeve_dress = {'strapless', 'spaghetti_straps', 'straps', 'sleeveless',
                            'cap_sleeve', 'short_sleeve', 'midi_sleeve', 'long_sleeve'}

            if 'strapless' in file:
                if set == 'train':
                    train_text_file.write(root + "/" + file + " 0\n")
                elif set == 'cv':
                    cv_text_file.write(root + "/" + file + " 0\n")
                elif set == 'test':
                    test_text_file.write(root + "/" + file + " 0\n")
            elif 'spaghetti_straps' in file:
                if set == 'train':
                    train_text_file.write(root + "/" + file + " 1\n")
                elif set == 'cv':
                    cv_text_file.write(root + "/" + file + " 1\n")
                elif set == 'test':
                    test_text_file.write(root + "/" + file + " 1\n")
            elif 'straps' in file:
                if set == 'train':
                    train_text_file.write(root + "/" + file + " 2\n")
                elif set == 'cv':
                    cv_text_file.write(root + "/" + file + " 2\n")
                elif set == 'test':
                    test_text_file.write(root + "/" + file + " 2\n")
            elif 'sleeveless' in file:
                if set == 'train':
                    train_text_file.write(root + "/" + file + " 2\n")
                elif set == 'cv':
                    cv_text_file.write(root + "/" + file + " 2\n")
                elif set == 'test':
                    test_text_file.write(root + "/" + file + " 2\n")
            elif 'cap_sleeve' in file:
                if set == 'train':
                    train_text_file.write(root + "/" + file + " 2\n")
                elif set == 'cv':
                    cv_text_file.write(root + "/" + file + " 2\n")
                elif set == 'test':
                    test_text_file.write(root + "/" + file + " 2\n")
            elif 'short_sleeve' in file:
                if set == 'train':
                    train_text_file.write(root + "/" + file + " 2\n")
                elif set == 'cv':
                    cv_text_file.write(root + "/" + file + " 2\n")
                elif set == 'test':
                    test_text_file.write(root + "/" + file + " 2\n")
            elif 'midi_sleeve' in file:
                if set == 'train':
                    train_text_file.write(root + "/" + file + " 2\n")
                elif set == 'cv':
                    cv_text_file.write(root + "/" + file + " 2\n")
                elif set == 'test':
                    test_text_file.write(root + "/" + file + " 2\n")
            elif 'long_sleeve' in file:
                if set == 'train':
                    train_text_file.write(root + "/" + file + " 2\n")
                elif set == 'cv':
                    cv_text_file.write(root + "/" + file + " 2\n")
                elif set == 'test':
                    test_text_file.write(root + "/" + file + " 2\n")

            counter += 1
            print counter
