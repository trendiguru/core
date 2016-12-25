#!/usr/bin/env python

__author__ = 'yonatan_guy'

import numpy as np
import os
import caffe
#import caffe_batch_infer import caffe
import sys
import argparse
import glob
import time
from trendi import background_removal, Utils, constants
import cv2
import urllib
import skimage
import requests



with open('/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/list_attr_cloth.txt', 'r') as handle:
    for line in handle:
        print line.split('  ')


# train_text_file = open("/home/yonatan/faces_stuff/55k_face_train_list.txt", "r")
#
#
# train_text_file.write("\"" + i + "\": [\'" + attribute + "\', \'" + type + "\']")
