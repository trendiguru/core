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
import pymongo; db = constants.db
import random

male_count = db.yonatan_gender.count({'gender': 'Male'})
female_count = db.yonatan_gender.count({'gender': 'Female'})

male_curser = db.yonatan_gender.find({'gender': 'Male'})
female_curser = db.yonatan_gender.find({'gender': 'Female'})

# open txt file
text_file = open("live_data_set_links.txt", "w")

for m_counter in xrange(male_count):
    url = male_curser[m_counter]['url']
    if isinstance(url, list):
        url = url[0]
    face = male_curser[m_counter]['face']
    if isinstance(face, list):
        face = face[0]
    # writing to txt file
    text_file.write(str(url) + ' 1 ' + str(male_curser[m_counter]['face']) + '\n')
    print m_counter

for f_counter in xrange(male_count):
    url = female_curser[f_counter]['url']
    if isinstance(url, list):
        url = url[0]
    face = female_curser[f_counter]['face']
    if isinstance(face, list):
        face = face[0]
    # writing to txt file
    text_file.write(str(url) + ' 0 ' + str(female_curser[f_counter]['face']) + '\n')
    print f_counter

text_file.flush()

# shuffle the txt file
lines = open('live_data_set_links.txt').readlines()
random.shuffle(lines)
open('live_data_set_links.txt', 'w').writelines(lines)

