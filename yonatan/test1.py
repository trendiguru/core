#!/usr/bin/env python

import numpy as np
import os
import caffe
from trendi import background_removal, Utils, constants
import cv2
import sys
import argparse
import glob
import time
import urllib
import skimage
import requests
import dlib
from ..utils import imutils
# import yonatan_classifier


### print the deep fashion 'fabric' categories ###
import yonatan_constants

dict = yonatan_constants.attribute_type_dict

counter = 0

for key, value in dict.iteritems():

    if value[1] == "fabric":
        counter += 1
        print value[0]

print "\ncounter: {0}".format(counter)