#!/usr/bin/env python

import numpy as np
import os
from PIL import Image
import caffe
from trendi import background_removal, Utils, constants
import cv2
import sys
import argparse
import glob
import time

counter = 0
ext = ["-mirrored.jpg", "-mirrored.png"]
mother_path = '/home/jeremy/image_dbs/colorful_fashion_parsing_data'

sets = {'images', 'labels', 'labels_200x150'}

for set in sets:
    path = os.path.join(mother_path, set)

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(tuple(ext)):
                os.remove(os.path.join(root, file))

                counter += 1
                print counter