#!/usr/bin/env python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil
import yonatan_constants
from .. import background_removal, utils, constants
import sys
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
import caffe
from ..utils import imutils
import argparse
import glob
import time
import skimage
import urllib
from PIL import Image
import pymongo
import dlib

test_text_file = open("55k_test_set_new.txt", "w")

counter = 0


dir_path = '/home/yonatan/55k_test_set'

for root, dirs, files in os.walk(dir_path):
    for file in files:
        resized_image = imutils.resize_keep_aspect(file, output_size=(224, 224))

        test_text_file.write("/home/yonatan/resized_test_dir/" + file + " 0\n")

        cv2.imwrite(os.path.join('/home/yonatan/resized_test_dir', file), resized_image)

        counter += 1
        print counter
