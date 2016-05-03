#!/usr/bin/env python

import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
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
from PIL import Image


def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if image.size == 0:
        print url
        return 'Fail'
    new_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return new_image


def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


def crop_face(raw_image, coordinates):
    image = url_to_image(raw_image)
    if image == 'Fail':
        return 'Fail'

    filter(lambda x: x.isdigit(), coordinates)

    print filter(lambda x: x.isdigit(), coordinates)
    print coordinates[0]
    print coordinates[0][0]
    print coordinates[0,0]

    x = coordinates[0][0]
    y = coordinates[0][1]
    w = coordinates[0][2]
    h = coordinates[0][3]

    face_image = image[y: y + h, x: x + w]

    return face_image


width = 115
height = 115



file = open('live_data_set_links.txt', 'r')
text_file = open("live_data_set_ready.txt", "w")

counter = 0

for line in file:
    counter += 1
    file_as_array_by_lines = line
    #split line to link and label
    words = file_as_array_by_lines.split()

    if words == []:
        continue

    face_image = crop_face(words[0], words[2])
    if face_image == 'Fail':
        continue
    # Resize it.
    resized_image = cv2.resize(face_image, (width, height))

    image_file_name = 'resized_face-' + str(counter) + '.jpg'

    cv2.imwrite(os.path.join('/home/yonatan/live_data_set', image_file_name), resized_image)

    text_file.write('/home/yonatan/live_data_set/' + image_file_name + ' ' + words[1] + '\n')

    print counter

text_file.flush()
