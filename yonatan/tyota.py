#!/usr/bin/env python

import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dresses = {'mini', 'midi', 'maxi'}

counter_mini = 0
counter_midi = 0
counter_maxi = 0

for dress in dresses:

    source_dir = '/home/yonatan/resized_' + dress + '_dresses'

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".jpg"):
                if dress == 'mini':
                    counter_mini += 1
                    print counter_mini
                elif dress == 'midi':
                    counter_midi += 1
                    print counter_midi
                elif dress == 'maxi':
                    counter_maxi += 1
                    print counter_maxi

print 'counter_mini = {0}, counter_midi = {1}, counter_maxi = {2}'.format(counter_mini, counter_midi, counter_maxi)
