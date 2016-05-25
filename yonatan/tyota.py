#!/usr/bin/env python

import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dresses = {'mini', 'midi', 'maxi'}

counter = 0

for dress in dresses:

    # source_dir = '/home/yonatan/all_' + dress + '_dresses'
    source_dir = '/home/yonatan/tyota_dresses'

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if not file.endswith(".jpg"):
                cv2.imwrite(os.path.join(source_dir, file + '.jpg'), file)
                os.remove(file)
                counter += 1
                print counter



