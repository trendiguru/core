#!/usr/bin/env python

__author__ = 'yonatan_guy'

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
import yonatan_classifier


MODLE_FILE = "/home/yonatan/trendi/yonatan/Alexnet_deploy.prototxt"
PRETRAINED = "/home/yonatan/alexnet_imdb_first_try/caffe_alexnet_train_faces_iter_10000.caffemodel"
caffe.set_mode_gpu()
image_dims = [115, 115]
mean, input_scale = np.array([120, 120, 120]), None
channel_swap = [2, 1, 0]
raw_scale = 255.0

# Make classifier.
classifier = yonatan_classifier.Classifier(MODLE_FILE, PRETRAINED,
                              image_dims=image_dims, mean=mean,
                              input_scale=input_scale, raw_scale=raw_scale,
                              channel_swap=channel_swap)

print "Done initializing!"

def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format

    if url.count('jpg') > 1:
        return None

    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if image.size == 0:
        return None
    new_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return new_image

#def theDetector(image):
def theDetector(url_or_np_array):

    first_start = time.time()

    print "Starting the genderism!"
    # check if i get a url (= string) or np.ndarray
    #if isinstance(url_or_np_array, basestring):
    #    full_image = url_to_image(url_or_np_array)
    #elif type(url_or_np_array) == np.ndarray:
    #    full_image = url_or_np_array
    if os.path.isdir(url_or_np_array):
        print("Loading folder: %s" % url_or_np_array)
        full_image = [caffe.io.load_image(im_f)
                  for im_f in glob.glob(url_or_np_array + '/*.jpg')]
    else:
        print("Loading file")
        full_image = caffe.io.load_image(url_or_np_array)

    #checks if the face coordinates are inside the image
    #height, width, channels = full_image.shape

    #x, y, w, h = face_coordinates

    #if x > width or x + w > width or y > height or y + h > height:
    #    return None


    # face_image = full_image[y: y + h, x: x + w]

    #face_for_caffe = [cv2_image_to_caffe(full_image)]
    #face_for_caffe = [caffe.io.load_image(face_image)]

    #if face_for_caffe is None:
    #    return None

    # Classify.
    start = time.time()
    predictions = classifier.predict(full_image)
    print("Done in %.2f s." % (time.time() - start))

    for i in range(1, len(predictions[:])):
        if predictions[i][1] > 0.7:
            print predictions[i][1]
            print 'Male'
        else:
            print predictions[i][0]
            print 'Female'

