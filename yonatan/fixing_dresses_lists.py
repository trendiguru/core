import numpy as np
# import skimage.io
from skimage import io
from scipy.ndimage import zoom
from skimage.transform import resize
import os
import caffe
# from .. import background_removal, utils, constants
# from ..utils import imutils
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
import yonatan_constants
import dlib
import requests
import grabCut
import re

images_new = []
boxes_new = []

images_array = np.load(open('/data/dress_detector/images.npy', 'rb'))
boxes_array = np.load(open('/data/dress_detector/boxes.npy', 'rb'))

images_old = images_array.tolist()
boxes_old = boxes_array.tolist()

for i in range(len(boxes_old)):

    coordinates = re.findall('\\d+', boxes_old[i])

    line_in_list_boxes = ([dlib.rectangle(coordinates[0], coordinates[1], coordinates[2], coordinates[3])])

    boxes_new.append(line_in_list_boxes)

    print i

np.array(boxes_new).dump(open('/data/dress_detector/boxes_new.npy', 'wb'))


for j in range(len(images_old)):

    image_num = re.findall('\\d+', images_old[i])

    image_file_name = 'dress-' + str(image_num) + '.jpg'

    line_in_list_images = io.imread('/data/dress_detector/images/' + image_file_name)

    images_new.append(line_in_list_images)

    print j

np.array(images_new).dump(open('/data/dress_detector/images_new.npy', 'wb'))
