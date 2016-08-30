#!/usr/bin/env python

import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

counter = 0

# dresses = {'mini', 'midi', 'maxi'}
#
# for dress in dresses:

source_dir = '/home/yonatan/dress_length_from_my_pc'

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if not file.endswith(".jpg"):
            print file
            old_file_name = source_dir + '/' + file
            new_file_name = source_dir + '/' + file + '.jpg'
            os.rename(old_file_name, new_file_name)
            counter += 1
            print counter



