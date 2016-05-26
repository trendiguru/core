#!/usr/bin/env python

import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sets = {'train', 'cv', 'test'}

counter = 0
counter_train = 750
counter_cv = 150
counter_test = 150

for set in sets:
    dir = '/home/yonatan/dresses_' + set + '_set'

    counter = 0

    for root, dirs, files in os.walk(dir):
        for file in files:

            counter += 1

    print counter