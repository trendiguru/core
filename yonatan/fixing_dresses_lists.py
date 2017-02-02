import numpy as np
# import skimage.io
from skimage import io
from scipy.ndimage import zoom
from skimage.transform import resize
import os
import caffe
from trendi.utils import imutils
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

sum_w = 0
sum_h = 0

# for i in range(len(boxes_old)):
#
#     if i > 1000:
#         break
#
#     coordinates = re.findall('\\d+', boxes_old[i])
#
#     line_in_list_boxes = ([dlib.rectangle(int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3]))])
#
#     boxes_new.append(line_in_list_boxes)
#
#     w = int(coordinates[2]) - int(coordinates[0])
#     h = int(coordinates[3]) - int(coordinates[1])
#
#     if w < 20 or h < 20:
#         print "BB too small"
#         continue
#
#     sum_w += w
#     sum_h += h
#
#     print i
#
# average_w = sum_w / (i + 1)
# average_h = sum_h / (i + 1)
#
# print "average_w: {0}\naverage_h: {1}".format(average_w, average_h)
#
# np.array(boxes_new).dump(open('/data/dress_detector/boxes_small_set.npy', 'wb'))
# print "Done with boxes_new!!"


# for j in range(len(images_old)):
#
#     if j > 1000:
#         break
#
#     image_num = re.findall('\\d+', images_old[j])
#
#     image_file_name = 'dress-' + str(image_num[0]) + '.jpg'
#
#     line_in_list_images = io.imread('/data/dress_detector/images/' + image_file_name)
#
#     images_new.append(line_in_list_images)
#
#     full_image = cv2.imread('/data/dress_detector/images/' + image_file_name)
#
#     resized_image = imutils.resize_keep_aspect(full_image, output_size=(150, 345))
#
#     cv2.imwrite(os.path.join('/data/dress_detector/resized_images', image_file_name), resized_image)
#
#     print j
#
# np.array(images_new).dump(open('/data/dress_detector/images_small_set.npy', 'wb'))
# print "Done with images_new!!"


# text_file = open("/data/irrelevant/irrelevant_db_images.txt", "w")
#
# for root, dirs, files in os.walk('/data/irrelevant/images'):
#     for file in files:
#
#         text_file.write('/data/irrelevant/images/' + file + ' 0\n')
#
# text_file.flush()


for root, dirs, files in os.walk('/data/dress_detector/resized_images'):
    for file in files:

        line_in_list_boxes = ([dlib.rectangle(0, 0, 150, 345)])

        boxes_new.append(line_in_list_boxes)


        line_in_list_images = io.imread('/data/dress_detector/resized_images/' + file)

        images_new.append(line_in_list_images)

        print file

np.array(boxes_new).dump(open('/data/dress_detector/boxes_small_set.npy', 'wb'))

np.array(images_new).dump(open('/data/dress_detector/images_small_set.npy', 'wb'))
