#!/usr/bin/env python

import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dresses = {'mini', 'midi', 'maxi'}
sets = {'train', 'cv', 'test'}

counter = 0
counter_train = 750
counter_cv = 150
counter_test = 150

for dress in dresses:
    source_dir = '/home/yonatan/resized_' + dress + '_dresses'

    counter = 0

    for root, dirs, files in os.walk(source_dir):
        for file in files:

            old_file_location = source_dir + '/' + file

            if counter < counter_train:
                new_file_location = '/home/yonatan/dresses_train_set/' + file
                os.rename(old_file_location, new_file_location)
                counter_train += 1
                print counter_train
            elif counter >= counter_train and counter < counter_train + counter_cv:
                new_file_location = '/home/yonatan/dresses_cv_set/' + file
                os.rename(old_file_location, new_file_location)
                counter_cv += 1
                print counter_cv
            elif counter >= counter_train + counter_cv and counter < counter_train + counter_cv + counter_test:
                new_file_location = '/home/yonatan/dresses_test_set/' + file
                os.rename(old_file_location, new_file_location)
                counter_test += 1
                print counter_test
            else:
                break

print 'counter_mini = {0}, counter_midi = {1}, counter_maxi = {2}'.format(counter_train, counter_cv, counter_test)
