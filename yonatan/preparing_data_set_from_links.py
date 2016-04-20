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


def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	# return the image
	return image

def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


def find_face(raw_image):

    #image = url_to_image(url)
    image = url_to_image(raw_image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

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

    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]

    face_image = image[y:(y + h), x:(x + w)]

    cv2.imshow("cropped_face", face_image)
    cv2.waitKey(0)
    return face_image


width = 115
height = 115

#opens the txt file for reading
file = open('55k_train_set.txt', 'r')

counter = 0
#convert the file to an array and divide it by lines
for line in file:
    counter += 1
    file_as_array_by_lines = line
    #split line to link and label
    words = file_as_array_by_lines.split()

    face_image = find_face(words[0])

    im = Image.fromarray(face_image)
    # Resize it.
    img = im.resize((width, height), Image.BILINEAR)

    img.save(os.path.join('/home/yonatan/55k_train_set', 'resized_face-'))

    cv2.imshow("cropped_face", img)
    cv2.waitKey(0)

    break

'''
sets = {'train', 'test'}

for set in sets:
    if set == 'train':
        mypath_male = '/home/yonatan/train_set/male'
        mypath_female = '/home/yonatan/train_set/female'
    else:
        mypath_male = '/home/yonatan/test_set/male'
        mypath_female = '/home/yonatan/test_set/female'

    for root, dirs, files in os.walk(mypath_male):
        for file in files:
            if file.endswith(".jpg"):
                # Open the image file.
                img = Image.open(os.path.join(root, file))

                # Resize it.
                img = img.resize((width, height), Image.BILINEAR)

                # Save it back to disk.
                img.save(os.path.join(root, 'resized_face-' + file))
                counter += 1
                print counter


    for root, dirs, files in os.walk(mypath_female):
        for file in files:
            if file.endswith(".jpg"):
                # Open the image file.
                img = Image.open(os.path.join(root, file))

                # Resize it.
                img = img.resize((width, height), Image.BILINEAR)

                # Save it back to disk.
                img.save(os.path.join(root, 'resized_face-' + file))
                counter += 1
                print counter
'''
