#!/usr/bin/env python

import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
import os
# import caffe
from .. import background_removal, utils, constants
from ..utils import imutils
import cv2
import sys
import argparse
import glob
import time
import skimage
import urllib
from PIL import Image
import pymongo
import argparse
import shutil


# if i run this function on braini2:
# db = constants.db

# if i run this function on brainik80a:
db = pymongo.MongoClient().mydb


dress_sleeve_dict = {
    'strapless': [db.yonatan_dresses.find(
        {'sleeve_length': ['true', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}), np.log(1)],
    'spaghetti_straps': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'true', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}), np.log(2)],
    'regular_straps': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'false', 'true', 'false', 'false', 'false', 'false', 'false', 'false']}), np.log(7)],
    'sleeveless': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'false', 'false', 'true', 'false', 'false', 'false', 'false', 'false']}), np.log(10)],
    'cap_sleeve': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'false', 'false', 'false', 'true', 'false', 'false', 'false', 'false']}), np.log(15)],
    'short_sleeve': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'true', 'false', 'false', 'false']}), np.log(45)],
    'midi_sleeve': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'true', 'false', 'false']}), np.log(72)],
    'long_sleeve': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'true', 'false']}), np.log(100)]
    # 'asymmetry' : [db.yonatan_dresses.find(
    #   {'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'true']}), 8]
}

# dress length #
dress_length_dict = {
    'mini_length': [db.yonatan_dresses.find(
        {'dress_length': ['true', 'false', 'false', 'false', 'false', 'false']}), 0],
    'above_knee': [db.yonatan_dresses.find(
        {'dress_length': ['false', 'true', 'false', 'false', 'false', 'false']}), 0],
    'knee_length': [db.yonatan_dresses.find(
        {'dress_length': ['false', 'false', 'true', 'false', 'false', 'false']}), 1],
    'tea_length': [db.yonatan_dresses.find(
        {'dress_length': ['false', 'false', 'false', 'true', 'false', 'false']}), 2],
    'ankle_length': [db.yonatan_dresses.find(
        {'dress_length': ['false', 'false', 'false', 'false', 'true', 'false']}), 3],
    'floor_length': [db.yonatan_dresses.find(
        {'dress_length': ['false', 'false', 'false', 'false', 'false', 'true']}), 3]
}

# men shirt sleeve #
men_shirt_sleeve_dict = {
    'regular_straps': [db.yonatan_men_shirts.find(
        {'shirt_sleeve_length': ['true', 'false', 'false', 'false', 'false']}), 0],
    'sleeveless': [db.yonatan_men_shirts.find(
        {'shirt_sleeve_length': ['false', 'true', 'false', 'false', 'false']}), 1],
    'short_sleeve': [db.yonatan_men_shirts.find(
        {'shirt_sleeve_length': ['false', 'false', 'true', 'false', 'false']}), 2],
    'midi_sleeve': [db.yonatan_men_shirts.find(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'true', 'false']}), 3],
    'long_sleeve': [db.yonatan_men_shirts.find(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'true']}), 4]
}

# pants length #
pants_length_dict = {
    'bermuda': [db.yonatan_pants.find({'pants_length': ['true', 'false', 'false', 'false']}), 0],
    'knee': [db.yonatan_pants.find({'pants_length': ['false', 'true', 'false', 'false']}), 1],
    'capri': [db.yonatan_pants.find({'pants_length': ['false', 'false', 'true', 'false']}), 2],
    'floor': [db.yonatan_pants.find({'pants_length': ['false', 'false', 'false', 'true']}), 3]
}

# women shirt sleeve #
women_shirt_sleeve_dict = {
    'strapless': [db.yonatan_women_shirts.find(
        {'shirt_sleeve_length': ['true', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}), 0],
    'spaghetti_straps': [db.yonatan_women_shirts.find(
        {'shirt_sleeve_length': ['false', 'true', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}), 1],
    'regular_straps': [db.yonatan_women_shirts.find(
        {'shirt_sleeve_length': ['false', 'false', 'true', 'false', 'false', 'false', 'false', 'false', 'false']}), 2],
    'sleeveless': [db.yonatan_women_shirts.find(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'true', 'false', 'false', 'false', 'false', 'false']}), 3],
    'cap_sleeve': [db.yonatan_women_shirts.find(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'true', 'false', 'false', 'false', 'false']}), 4],
    'short_sleeve': [db.yonatan_women_shirts.find(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'true', 'false', 'false', 'false']}), 5],
    'midi_sleeve': [db.yonatan_women_shirts.find(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'true', 'false', 'false']}), 6],
    'long_sleeve': [db.yonatan_women_shirts.find(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'true', 'false']}), 7]
    # 'asymmetry': [db.yonatan_women_shirts.find(
    #    {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'true']}), 8]
}

# collar basic #
collar_basic_dict = {
    'crew_neck': 0,
    'scoop_neck': 1,
    'v_neck': 2,
    'deep_v_neck': 3,
    'Henley_t_shirts': 4,
    'polo_collar': 5,
    'tie_neck': 6,
    'turtleneck': 7,
    'Hooded_T_Shirt': 8
}

# style #
style_dict = {
    'swimsuit': 0,
    'sports': 1,
    'others': 2,
    'prom': 3,
    'bride_dress': 4
}
