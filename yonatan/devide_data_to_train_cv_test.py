#!/usr/bin/env python

import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sleeve_dress = {'strapless', 'spaghetti_straps', 'straps', 'sleeveless',
                'cap_sleeve', 'short_sleeve', 'midi_sleeve', 'long_sleeve'}

#sets = {'train', 'cv', 'test'}

counter = 0
counter_train = 640
counter_cv = 130
counter_test = 130

for dress in sleeve_dress:
    source_dir = '/home/yonatan/db_' + sleeve_dress + '_dresses'

    counter = 0

    for root, dirs, files in os.walk(source_dir):
        for file in files:

            old_file_location = source_dir + '/' + file

            if counter < counter_train:
                new_file_location = '/home/yonatan/db_dresses_train_set/' + file
                os.rename(old_file_location, new_file_location)
                counter += 1
            elif counter >= counter_train and counter < counter_train + counter_cv:
                new_file_location = '/home/yonatan/db_dresses_cv_set/' + file
                os.rename(old_file_location, new_file_location)
                counter += 1
            elif counter >= counter_train + counter_cv and counter < counter_train + counter_cv + counter_test:
                new_file_location = '/home/yonatan/db_dresses_test_set/' + file
                os.rename(old_file_location, new_file_location)
                counter += 1
            else:
                print counter
                break

print 'counter_train = {0}, counter_cv = {1}, counter_test = {2}, counter = {3}'.format(counter_train, counter_cv, counter_test, counter)
