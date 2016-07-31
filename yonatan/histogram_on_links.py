#!/usr/bin/env python

import matplotlib.pyplot as plt
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
import requests
import yonatan_classifier


MODLE_FILE = "/home/yonatan/trendi/yonatan/Alexnet_deploy.prototxt"
PRETRAINED = "/home/yonatan/alexnet_imdb_first_try/caffe_alexnet_train_faces_iter_10000.caffemodel"
caffe.set_mode_gpu()
image_dims = [115, 115]
mean, input_scale = np.array([120, 120, 120]), None
channel_swap = [2, 1, 0]
#channel_swap = None
raw_scale = 255.0

# Make classifier.
classifier = yonatan_classifier.Classifier(MODLE_FILE, PRETRAINED,
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



#path = '/home/yonatan/55k_test_set'
array_success = np.array([])
array_failure = np.array([])

text_file = open("live_data_set_links.txt", "r")

width = 115
height = 115

counter = 0
success_counter = 0
failure_counter = 0
guessed_f_instead_m = 0
guessed_m_instead_f = 0

for line in text_file:
    counter += 1
    file_as_array_by_lines = line
    # split line to link and label
    words = file_as_array_by_lines.split()

    if words == []:
        print 'empty string!'
        continue

    #full_image = url_to_image(words[0])

    response = requests.get(words[0])  # download
    full_image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)

    if full_image is None:
        continue

    # checks if the face coordinates are inside the image
    height, width, channels = full_image.shape

    x = int(filter(lambda x: x.isdigit(), words[2]))
    y = int(filter(lambda x: x.isdigit(), words[3]))
    w = int(filter(lambda x: x.isdigit(), words[4]))
    h = int(filter(lambda x: x.isdigit(), words[5]))

    if x > width or x + w > width or y > height or y + h > height:
        continue

    face_image = full_image[y: y + h, x: x + w]
    # input_image = image

    if face_image == 'Fail':
        print 'face_image not found!'
        continue

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    # face_file = os.path.expanduser(face_image)

    # print face_file
    # print type(face_file)

    # inputs = Utils.get_cv2_img_array(image)
    # inputs = [cv2.imread(input_file)]

    #face_for_caffe = [cv2_image_to_caffe(face_image)]
    face_for_caffe = [face_image]

    # face_for_caffe = [caffe.io.load_image(face_image)]

    if face_for_caffe is None:
        continue

    # Classify.
    start = time.time()
    predictions = classifier.predict(face_for_caffe)
    print("Done in %.2f s." % (time.time() - start))


    #if the gender_detector is right
    if (predictions[0][0] > predictions[0][1]) and (words[1] == '0'):
        array_success = np.append(array_success, predictions[0][0])
    elif (predictions[0][1] > predictions[0][0]) and (words[1] == '1'):
        array_success = np.append(array_success, predictions[0][1])
    # if the gender_detector is wrong
    elif (predictions[0][0] > predictions[0][1]) and (words[1] == '1'):
        array_failure = np.append(array_failure, predictions[0][0])
        print predictions
        guessed_f_instead_m += 1
    elif (predictions[0][1] > predictions[0][0]) and (words[1] == '0'):
        array_failure = np.append(array_failure, predictions[0][1])
        print predictions
        guessed_m_instead_f += 1

    print counter

#print guessed_f_instead_m
#print guessed_m_instead_f

success = len(array_success)
failure = len(array_failure)

if success == 0 or failure == 0:
    print "wrong!"
else:
    print 'accuracy percent: {0}'.format(float(success) / (success + failure))


histogram=plt.figure(1)

#bins = np.linspace(-1000, 1000, 50)

plt.hist(array_success, alpha=0.5, label='array_success')
plt.legend()

plt.hist(array_failure, alpha=0.5, label='array_failure')
plt.legend()

histogram.savefig('live_test_links_bgr.png')