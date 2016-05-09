#!/usr/bin/env python

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import caffe
import sys
import argparse
import glob
import time
from trendi import background_removal, Utils, constants
import cv2
import urllib
import skimage


#MODLE_FILE = "/home/yonatan/trendi/yonatan/Alexnet_deploy.prototxt"
#PRETRAINED = "/home/yonatan/alexnet_imdb_first_try/caffe_alexnet_train_faces_iter_10000.caffemodel"
MODLE_FILE = "/home/yonatan/neuro_doorman/deploy.prototxt"
PRETRAINED = "/home/yonatan/neuro_doorman/_iter_8078.caffemodel"
caffe.set_mode_gpu()
#image_dims = [115, 115]
#mean, input_scale = np.array([107,117,123]), None
image_dims = [227, 227]
mean, input_scale = None, None
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


array_success = np.array([])
array_failure = np.array([])


width = 115
height = 115


success_counter = 0
failure_counter = 0
guessed_f_instead_m = 0
guessed_m_instead_f = 0

counter = 0

path = '/home/jeremy/image_dbs/doorman/irrelevant'


#text_file = open("irrelevant.txt", "w")
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(").jpg"):
            #text_file.write(root + "/" + file)
            #text_file.flush()

            # split line to link and label
            # words = file_as_array_by_lines.split()
            '''
            if words == []:
                print 'empty string!'
                continue

            full_image = url_to_image(words[0])

            if full_image is None:
                continue
            '''
            #face_for_caffe = [full_image]

            face_for_caffe = [caffe.io.load_image(root + "/" + file)]

            if face_for_caffe is None:
                continue

            # Classify.
            start = time.time()
            predictions = classifier.predict(face_for_caffe)
            print("Done in %.2f s." % (time.time() - start))

            if (predictions[0][1] > predictions[0][0]):
                array_success = np.append(array_success, predictions[0][1])
            # if the gender_detector is wrong
            elif (predictions[0][0] > predictions[0][1]):
                array_failure = np.append(array_failure, predictions[0][0])

            counter += 1
            print counter


histogram=plt.figure(1)

bins = np.linspace(-1000, 1000, 50)

plt.hist(array_success, alpha=0.5, label='array_success')
plt.legend()

plt.hist(array_failure, alpha=0.5, label='array_failure')
plt.legend()

histogram.savefig('only_irrelevant_images.png')