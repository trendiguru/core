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


MODLE_FILE = "/home/yonatan/trendi/yonatan/Alexnet_deploy.prototxt"
PRETRAINED = "/home/yonatan/alexnet_imdb_first_try/caffe_alexnet_train_faces_iter_10000.caffemodel"
caffe.set_mode_gpu()
image_dims = [115, 115]
mean, input_scale = np.array([120, 120, 120]), None
channel_swap = [2, 1, 0]
raw_scale = 255.0
# ext = 'jpg'

# Make classifier.
classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
                              image_dims=image_dims, mean=mean,
                              input_scale=input_scale, raw_scale=raw_scale,
                              channel_swap=channel_swap)


def theDetector(image):
#def theDetector(image, coordinates):

    #input_image = image[coordinates[1]: coordinates[1] + coordinates[3], coordinates[0]: coordinates[0] + coordinates[2]]
    input_image = image


    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    input_file = os.path.expanduser(input_image)

    print type(input_file)

    inputs = Utils.get_cv2_img_array(image)

    #inputs = np.load(input_file)
    #inputs = [caffe.io.load_image(input_file)]

    #print type(inputs)
    #print inputs.shape

    if not len(inputs):
        return 'None'

    # Classify.
    start = time.time()
    predictions = classifier.predict(inputs)
    print("Done in %.2f s." % (time.time() - start))

    if predictions[0][1] > 0.7:
        print predictions[0][1]
        print "it's a boy!"
        return 'male'
    else:
        print predictions[0][0]
        print "it's a girl!"
        return 'female'
