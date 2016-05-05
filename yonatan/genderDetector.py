#!/usr/bin/env python

import numpy as np
import os
import caffe
import sys
import argparse
import glob
import time
from trendi import background_removal, Utils, constants
import cv2
import urllib
import skimage


MODLE_FILE = "/home/yonatan/trendi/yonatan/Alexnet_deploy.prototxt"
PRETRAINED = "/home/yonatan/alexnet_imdb_first_try/caffe_alexnet_train_faces_iter_10000.caffemodel"
caffe.set_mode_gpu()
image_dims = [115, 115]
mean, input_scale = np.array([120, 120, 120]), None
channel_swap = [2, 1, 0]
#channel_swap = None
raw_scale = 255.0

# Make classifier.
classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
                              image_dims=image_dims, mean=mean,
                              input_scale=input_scale, raw_scale=raw_scale,
                              channel_swap=channel_swap)


def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    print url

    if url.count('jpg') > 1:
        return None

    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if image.size == 0:
        print url
        return None
    new_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return new_image


#def theDetector(image):
def theDetector(url_image, face_coordinates):

    full_image = url_to_image(url_image)

    #checks if the face coordinates are inside the image
    height, width, channels = full_image.shape

    x = face_coordinates[0]
    y = face_coordinates[1]
    w = face_coordinates[2]
    h = face_coordinates[3]

    if x > width or x + w > width or y > height or y + h > height:
        return None

    face_image = full_image[y: y + h, x: x + w]
    #input_image = image
    print face_image
    print type(face_image)

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    #face_file = os.path.expanduser(face_image)

    #print face_file
    #print type(face_file)

    #inputs = Utils.get_cv2_img_array(image)
    #inputs = [cv2.imread(input_file)]

    face_for_caffe = [cv2_image_to_caffe(face_image)]
    #face_for_caffe = [caffe.io.load_image(face_image)]

    print face_for_caffe
    print type(face_for_caffe)
    print face_for_caffe[0].shape

    if face_for_caffe is None:
        return None

    # Classify.
    start = time.time()
    predictions = classifier.predict(face_for_caffe)
    print("Done in %.2f s." % (time.time() - start))

    if predictions[0][1] > 0.7:
        print predictions[0][1]
        print "it's a boy!"
        return 'male'
    else:
        print predictions[0][0]
        print "it's a girl!"
        return 'female'
