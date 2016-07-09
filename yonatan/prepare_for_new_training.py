#!/usr/bin/env python

import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
import os
import caffe
from .. import background_removal, Utils, constants
import cv2
import sys
import argparse
import glob
import time
import skimage
import urllib
from PIL import Image
import pymongo
import shutil

sleeve_dress = {'strapless', 'spaghetti_straps', 'straps', 'sleeveless',
                'cap_sleeve', 'short_sleeve', 'midi_sleeve', 'long_sleeve', 'asymmetry'}

sets = {'train', 'cv', 'test'}


for dress_type in sleeve_dress:
    shutil.rmtree('/home/yonatan/db_' + dress_type + '_dresses')
    shutil.rmtree('/home/yonatan/resized_db_' + dress_type + '_dresses')
    os.mkdir('/home/yonatan/db_' + dress_type + '_dresses')
    os.mkdir('/home/yonatan/resized_db_' + dress_type + '_dresses')

for set in sets:
    shutil.rmtree('/home/yonatan/dresses_' + set + '_set')
    os.mkdir('/home/yonatan/dresses_' + set + '_set')
    #os.remove('/home/yonatan/db_dresses_' + set + '.txt')


