#!/usr/bin/env python

import numpy as np
import skimage.io
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
import pymongo

db = constants.db

sleeve_dict = {
'strapless' : db.yonatan_dresses.count({'sleeve_length': ['true', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}),
'spaghetti_straps' : db.yonatan_dresses.count({'sleeve_length': ['false', 'true', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}),
'straps' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'true', 'false', 'false', 'false', 'false', 'false', 'false']}),
'sleeveless' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'false', 'true', 'false', 'false', 'false', 'false', 'false']}),
'cap_sleeve' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'false', 'false', 'true', 'false', 'false', 'false', 'false']}),
'short_sleeve' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'true', 'false', 'false', 'false']}),
'midi_sleeve' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'true', 'false', 'false']}),
'long_sleeve' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'true', 'false']}),
'asymmetry' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'true']})
}

sum_of_all = sleeve_dict['strapless'] + sleeve_dict['spaghetti_straps'] + sleeve_dict['straps'] + \
             sleeve_dict['sleeveless'] + sleeve_dict['cap_sleeve'] + sleeve_dict['short_sleeve'] + \
             sleeve_dict['midi_sleeve'] + sleeve_dict['long_sleeve'] + sleeve_dict['asymmetry']

sum_of_all_already_seen = db.yonatan_dresses.count({'already_seen_dress_sleeve': True})

deleted = sum_of_all_already_seen - sum_of_all

for key, value in sleeve_dict.iteritems():
    print '{0}: {1}, percent: {2}'.format(key, value, round(float(value) / sum_of_all, 2))
print 'sum of all: {0}'.format(sum_of_all)
print 'deleted: {0}'.format(deleted)
