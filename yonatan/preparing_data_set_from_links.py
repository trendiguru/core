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
import requests


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


def find_face(raw_image):
    #image = url_to_image(url)
    #image = url_to_image(raw_image)
    response = requests.get(raw_image)  # download
    if not response:
        return 'Fail'
    image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    if image == 'Fail':
        return 'Fail'

    gray = cv2.cvtColor(image, constants.BGR2GRAYCONST)
    face_cascades = [
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt2.xml')),
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt.xml')),
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt_tree.xml')),
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_default.xml'))]
    cascade_ok = False
    for cascade in face_cascades:
        if not cascade.empty():
            cascade_ok = True
            break
    if cascade_ok is False:
        raise IOError("no good cascade found!")
    faces = []
    for cascade in face_cascades:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(5, 5),
            flags=constants.scale_flag
        )
        if len(faces) > 0:
            break

    if len(faces) == 0:
        print "Fail"
        return 'Fail'

    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]

    face_image = image[y:(y + h), x:(x + w)]

    return face_image


width = 224
height = 224


sets = {'train', 'cv', 'test'}

for set in sets:
    if set == 'train':
        file = open('Stan_train.txt', 'r')
        text_file = open("55k_face_train_list.txt", "w")
    elif set == 'cv':
        file = open('Stan_cv.txt', 'r')
        text_file = open("55k_face_cv_list.txt", "w")
    else:
        file = open('Stan_test.txt', 'r')
        text_file = open("55k_face_test_list.txt", "w")

    counter = 0

    for line in file:
        counter += 1
        file_as_array_by_lines = line
        #split line to link and label
        words = file_as_array_by_lines.split()

        if words == []:
            continue

        face_image = find_face(words[0])
        if face_image == 'Fail':
            continue
        # Resize it.
        resized_image = cv2.resize(face_image, (width, height))

        image_file_name = 'resized_face-' + str(counter) + '.jpg'

        cv2.imwrite(os.path.join('/home/yonatan/55k_' + set + '_set_224', image_file_name), resized_image)

        text_file.write('/home/yonatan/55k_' + set + '_set_224/' + image_file_name + ' ' + words[1] + '\n')

        print counter

    text_file.flush()