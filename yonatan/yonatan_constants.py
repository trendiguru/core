#!/usr/bin/env python

import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
import os
import caffe
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

db = constants.db


dress_sleeve_dict = {
    'strapless': [db.yonatan_dresses.find(
        {'sleeve_length': ['true', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}), 0],
    'spaghetti_straps': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'true', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}), 1],
    'straps': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'false', 'true', 'false', 'false', 'false', 'false', 'false', 'false']}), 2],
    'sleeveless': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'false', 'false', 'true', 'false', 'false', 'false', 'false', 'false']}), 3],
    'cap_sleeve': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'false', 'false', 'false', 'true', 'false', 'false', 'false', 'false']}), 4],
    'short_sleeve': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'true', 'false', 'false', 'false']}), 5],
    'midi_sleeve': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'true', 'false', 'false']}), 6],
    'long_sleeve': [db.yonatan_dresses.find(
        {'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'true', 'false']}), 7]
    # 'asymmetry' : [db.yonatan_dresses.find({'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'true']}), 8]
}

# dress length #
dress_length_dict = {
    'mini_length': db.yonatan_dresses.count(
        {'dress_length': ['true', 'false', 'false', 'false', 'false', 'false']}),
    'above_knee': db.yonatan_dresses.count({'dress_length': ['false', 'true', 'false', 'false', 'false', 'false']}),
    'knee_length': db.yonatan_dresses.count(
        {'dress_length': ['false', 'false', 'true', 'false', 'false', 'false']}),
    'tea_length': db.yonatan_dresses.count({'dress_length': ['false', 'false', 'false', 'true', 'false', 'false']}),
    'ankle_length': db.yonatan_dresses.count(
        {'dress_length': ['false', 'false', 'false', 'false', 'true', 'false']}),
    'floor_length': db.yonatan_dresses.count(
        {'dress_length': ['false', 'false', 'false', 'false', 'false', 'true']})
}

# men shirt sleeve #
men_shirt_sleeve_dict = {
    'straps': db.yonatan_men_shirts.count({'shirt_sleeve_length': ['true', 'false', 'false', 'false', 'false']}),
    'sleeveless': db.yonatan_men_shirts.count(
        {'shirt_sleeve_length': ['false', 'true', 'false', 'false', 'false']}),
    'short_sleeve': db.yonatan_men_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'true', 'false', 'false']}),
    'midi_sleeve': db.yonatan_men_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'true', 'false']}),
    'long_sleeve': db.yonatan_men_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'true']})
}

# pants length #
pants_length_dict = {
    'bermuda': db.yonatan_pants.count({'pants_length': ['true', 'false', 'false', 'false']}),
    'knee': db.yonatan_pants.count({'pants_length': ['false', 'true', 'false', 'false']}),
    'capri': db.yonatan_pants.count({'pants_length': ['false', 'false', 'true', 'false']}),
    'floor': db.yonatan_pants.count({'pants_length': ['false', 'false', 'false', 'true']})
}

# women shirt sleeve #
women_shirt_sleeve_dict = {
    'strapless': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['true', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}),
    'spaghetti_straps': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'true', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}),
    'straps': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'true', 'false', 'false', 'false', 'false', 'false', 'false']}),
    'sleeveless': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'true', 'false', 'false', 'false', 'false', 'false']}),
    'cap_sleeve': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'true', 'false', 'false', 'false', 'false']}),
    'short_sleeve': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'true', 'false', 'false', 'false']}),
    'midi_sleeve': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'true', 'false', 'false']}),
    'long_sleeve': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'true', 'false']}),
    'asymmetry': db.yonatan_women_shirts.count(
        {'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'true']})
}
