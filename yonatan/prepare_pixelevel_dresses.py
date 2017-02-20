#!/usr/bin/env python

import os
import sys
import glob
import numpy as np

import dlib
from skimage import io
import cv2


path = "/data/jeremy/image_dbs/pixlevel_images/pixlevel_fullsize_labels_v3"

dress_pixel_value = 7  # 13 is the pixel value represents a dress
dress_counter = 0
dress_list = []

for root, dirs, files in os.walk(path):

    for file in files:

        if not file.endswith(".png"):
            print "not a png file"
            continue

        labeled_image = cv2.imread(os.path.join(root, file), 0)  # the zero is for

        is_dress = np.any(labeled_image == dress_pixel_value)

        if is_dress:
            dress_counter += 1
            dress_list.append(file)
            print "found dress!"

            dress_and_zeros = np.where(labeled_image != dress_pixel_value, 0, labeled_image)

        else:
            print "no dress!"

print "number of dresses: {0}\ndress_list: {1}".format(dress_counter, dress_list)


#
# In [81]: image = cv2.imread('14483.png')
#     ...:
#
# In [82]: image_0 = cv2.imread('14483.png', 0)
#
# In [83]: dress_and_zeros = np.where(image != dress_pixel_value, 0, image)
#
# In [84]: dress_and_zeros_0 = np.where(image != dress_pixel_value, 0, image)
#
# In [85]: mask = np.where(dress_and_zeros == 13).astype('uint8')
# ---------------------------------------------------------------------------
# AttributeError                            Traceback (most recent call last)
# <ipython-input-85-d68cc1ecfca6> in <module>()
# ----> 1 mask = np.where(dress_and_zeros == 13).astype('uint8')
#
# AttributeError: 'tuple' object has no attribute 'astype'
#
# In [86]: mask = np.where(dress_and_zeros == 13, 0, 1).astype('uint8')
#
# In [87]: mask_0 = np.where(dress_and_zeros_0 == 13, 0, 1).astype('uint8')
#
# In [88]: i, j = np.where(mask)
# ---------------------------------------------------------------------------
# ValueError                                Traceback (most recent call last)
# <ipython-input-88-2a390ec7fc7e> in <module>()
# ----> 1 i, j = np.where(mask)
#
# ValueError: too many values to unpack
#
# In [89]: dress_and_zeros = dress_and_zeros*255
#
# In [90]: dress_and_zeros_0 = dress_and_zeros_0*255
#
# In [91]: mask = np.zeros(dress_and_zeros.shape[:2] ,np.uint8)
#
# In [92]: mask_0 = np.zeros(dress_and_zeros_0.shape[:2] ,np.uint8)
#
# In [93]: mask2 = np.where(( mask ==2 ) |( mask ==0) ,0 ,1).astype('uint8')
#
# In [94]: mask2_0 = np.where(( mask_0 ==2 ) |( mask_0 ==0) ,0 ,1).astype('uint8')
#
# In [95]: without_bg_img = dress_and_zeros* mask2[:, :, np.newaxis]*255
#
# In [96]: without_bg_img_0 = dress_and_zeros_0* mask2[:, :, np.newaxis]*255
#
# In [97]: print cv2.imwrite("/data/yonatan/linked_to_web/dress_and_zeros.jpg", dress_and_zeros)
# True
#
# In [98]: print cv2.imwrite("/data/yonatan/linked_to_web/dress_and_zeros_0.jpg", dress_and_zeros_0)
# True
#
# In [99]: image = cv2.imread('14483.png')
#
# In [100]: image = image * 10
#
# In [101]: print cv2.imwrite("/data/yonatan/linked_to_web/testing_dress.jpg", image)
# True
#
#
