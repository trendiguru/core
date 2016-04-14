#!/usr/bin/env python

import numpy as np
import os
import sys
import argparse
import glob
import time
import caffe


#def main(input_image):
#def genderator(argv):
def genderator(image):

    #input_image = sys.argv[1]
    input_image = image
    MODLE_FILE = "/home/yonatan/trendi/yonatan/deploy.prototxt"
    PRETRAINED = "/home/yonatan/network_5000_train_faces_115/intermediate_output_iter_10000.caffemodel"
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

    # making the predictions -> precentage
    sum = predictions[0][0] + predictions[0][1]
    predictions[0][0] = predictions[0][0] / sum
    predictions[0][1] = predictions[0][1] / sum

    if predictions[0][0] > predictions[0][1]:
        print "it's a boy!"
    else:
        print "it's a girl!"
    print predictions
    print np.array(inputs).shape

    text_file = open("face_testing.txt", "w")
    text_file = open("face_testing.txt", "a")
    text_file.write("predictions: %s" % (np.array2string(predictions, separator=', ')))
    text_file.flush()
#if __name__ == '__main__':
#    genderator(sys.argv)
