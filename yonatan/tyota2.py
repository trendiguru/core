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


i, j = 0

answer_dict = nfc.pd(int(sys.argv[1]))
if not answer_dict['success']:
   print 'false'
neuro_mask = answer_dict['mask']

neuro_mask = (neuro_mask > 0) * 255

cv2.imwrite('only_background.jpg', neuro_mask)
