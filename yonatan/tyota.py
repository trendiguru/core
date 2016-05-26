#!/usr/bin/env python

import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dresses = {'mini', 'midi', 'maxi'}
sets = {'train', 'cv', 'test'}

dresses_train_set_from_each_catagory = 750
dresses_cv_set_from_each_catagory = 150
dresses_test_set_from_each_catagory = 150

counter_train = 0
counter_cv = 0
counter_test = 0

for dress in dresses:

    source_dir = '/home/yonatan/resized_' + dress + '_dresses'

    for set in sets:

        destination_dir = '/home/yonatan/dresses_' + set + '_set'

        for root, dirs, files in os.walk(source_dir):
            for file in files:

                old_file_location = source_dir + '/' + file
                new_file_location = destination_dir + '/' + file
                os.rename(old_file_location, new_file_location)

                if set == 'train':
                    counter_train += 1
                    print counter_train
                elif set == 'cv':
                    counter_cv += 1
                    print counter_cv
                elif set == 'test':
                    counter_test += 1
                    print counter_test

print 'counter_mini = {0}, counter_midi = {1}, counter_maxi = {2}'.format(counter_train, counter_cv, counter_test)
