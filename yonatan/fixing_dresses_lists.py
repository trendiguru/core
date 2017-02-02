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

    if i > 1000:
        break

    coordinates = re.findall('\\d+', boxes_old[i])

    line_in_list_boxes = ([dlib.rectangle(int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3]))])

    boxes_new.append(line_in_list_boxes)

    print i

np.array(boxes_new).dump(open('/data/dress_detector/boxes_small_set.npy', 'wb'))
print "Done with boxes_new!!"


for j in range(len(images_old)):

    if j > 1000:
        break

    image_num = re.findall('\\d+', images_old[j])

    image_file_name = 'dress-' + str(image_num[0]) + '.jpg'

    line_in_list_images = io.imread('/data/dress_detector/images/' + image_file_name)

    images_new.append(line_in_list_images)

    print j

np.array(images_new).dump(open('/data/dress_detector/images_small_set.npy', 'wb'))
print "Done with images_new!!"

# text_file = open("/data/irrelevant/irrelevant_db_images.txt", "w")
#
# for root, dirs, files in os.walk('/data/irrelevant/images'):
#     for file in files:
#
#         text_file.write('/data/irrelevant/images/' + file + ' 0\n')
#
# text_file.flush()

