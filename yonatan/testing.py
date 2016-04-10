#!/usr/bin/env python

import caffe
import numpy as np
from trendi import background_removal, Utils, constants
import cv2
import os
import sys
import argparse
import glob
import time

path = "/home/yonatan/test_set/female/Juljia_Vysotskij_0001.jpg"
image = Utils.get_cv2_img_array(path)
print image

def find_face(image):
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
    return faces


face = find_face(image)
print face

if len(face) == 0:
    print "Fail"

x = face[0][0]
y = face[0][1]
w = face[0][2]
h = face[0][3]

face_image = image[y:(y+h), x:(x+w)]

print face_image
print type(face_image)
print face_image.shape
cv2.imshow("cropped", face_image)
cv2.waitKey(0)

def the_detector(image):

    input_image = image
    MODLE_FILE = "/home/yonatan/core/yonatan/deploy.prototxt"
    PRETRAINED = "/home/yonatan/network_5000_train_set/intermediate_output_iter_10000.caffemodel"
    caffe.set_mode_gpu()
    image_dims = [250, 250]
    mean, input_scale = None, None
    channel_swap = [2, 1, 0]
    raw_scale = 255.0
    ext = 'jpg'

    # Make classifier.
    classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
            image_dims=image_dims, mean=mean,
            input_scale=input_scale, raw_scale=raw_scale,
            channel_swap=channel_swap)


    inputs = image

# Classify.
    start = time.time()
    predictions = classifier.predict(inputs)
    print("Done in %.2f s." % (time.time() - start))

    # Save
    #print("Saving results into %s" % args.output_file)
    #np.save(args.output_file, predictions)
    if predictions[0][0] > predictions[0][1]:
        print "it's a boy!"
    else:
        print "it's a girl!"
    print predictions
    print np.array(inputs).shape


the_detector(face_image)
