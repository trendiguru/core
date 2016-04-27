#!/usr/bin/env python

import caffe
import numpy as np
from .. import background_removal, Utils, constants
import cv2
import os
import sys
import argparse
import glob
import time
import skimage
import urllib

#path = "/home/yonatan/test_set/female/Juljia_Vysotskij_0001.jpg"
#image = Utils.get_cv2_img_array(path)


# METHOD #1: OpenCV, NumPy, and urllib
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


#def find_face(url):
def find_face(argv):

    #image = url_to_image(url)
    image = url_to_image(sys.argv[1])
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


face = find_face(sys.argv)
print face


'''
print face_image
print type(face_image)
print face_image.shape
cv2.imshow("cropped", face_image)
cv2.waitKey(0)
'''

def the_detector(face):

    MODLE_FILE = "/home/yonatan/trendi/yonatan/Alexnet_deploy.prototxt"
    PRETRAINED = "/home/yonatan/alexnet_imdb_first_try/caffe_alexnet_train_iter_10000.caffemodel"
    caffe.set_mode_gpu()
    image_dims = (115, 115)
    mean, input_scale = None, None
    channel_swap = None
    raw_scale = 255.0
    ext = 'jpg'

    # Make classifier.
    classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
            image_dims=image_dims, mean=mean,
            input_scale=input_scale, raw_scale=raw_scale,
            channel_swap=channel_swap)

    #threshed = cv2.adaptiveThreshold(face, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    #cv2.imshow("cropped", threshed)
    #cv2.waitKey(0)

    input = [cv2_image_to_caffe(face)]
    print("Classifying %d input." % len(input))
# Classify.
    start = time.time()
    predictions = classifier.predict(input)
    print("Done in %.2f s." % (time.time() - start))

    # Save
    #print("Saving results into %s" % args.output_file)
    #np.save(args.output_file, predictions)
    if predictions[0][1] > predictions[0][0]:
        print "it's a boy!"
    else:
        print "it's a girl!"
    print predictions


the_detector(face)
