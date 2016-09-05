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
from trendi.paperdoll import neurodoll_falcon_client as nfc


# answer_dict = nfc.pd('/home/yonatan/dress_length_3_labels_sets/dress_length_3_labels_test_set/cute_midi_dress(87).jpg')
answer_dict = nfc.pd('http://g.nordstromimage.com/ImageGallery/store/product/Zoom/19/_12554259.jpg')

if not answer_dict['success']:
   print 'false'
neuro_mask = answer_dict['mask']

neuro_mask = (neuro_mask > 0) * 255

cv2.imwrite('only_background.jpg', neuro_mask)
