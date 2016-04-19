#!/usr/bin/env python

import numpy as np
import os
import caffe
import sys
import argparse
import glob
import time


#def main(input_image):
def genderator(argv):

    input_image = sys.argv[1]
    MODLE_FILE = "/home/yonatan/trendi/yonatan/Alexnet_deploy.prototxt"
    PRETRAINED = "/home/yonatan/alexnet_first_try/caffe_alexnet_train_iter_10000.caffemodel"
    caffe.set_mode_gpu()
    image_dims = [115, 115]
    mean, input_scale = None, None
    channel_swap = [2, 1, 0]
    raw_scale = 255.0
    ext = 'jpg'

    # Make classifier.
    classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
            image_dims=image_dims, mean=mean,
            input_scale=input_scale, raw_scale=raw_scale,
            channel_swap=channel_swap)


    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    input_file = os.path.expanduser(input_image)
    if input_file.endswith('npy'):
        print("Loading file: %s" % input_file)
        inputs = np.load(input_file)
    elif os.path.isdir(input_file):
        print("Loading folder: %s" % input_file)
        inputs =[caffe.io.load_image(im_f)
                 for im_f in glob.glob(input_file + '/*.' + ext)]
    else:
        print("Loading file: %s" % input_file)
        inputs = [caffe.io.load_image(input_file)]

    print("Classifying %d inputs." % len(inputs))

    # Classify.
    start = time.time()
    predictions = classifier.predict(inputs)
    print("Done in %.2f s." % (time.time() - start))

    if predictions[0][0] > predictions[0][1]:
        print "it's a boy!"
    else:
        print "it's a girl!"

    return predictions
