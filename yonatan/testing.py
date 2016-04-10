#!/usr/bin/env python

import caffe
import numpy as np
from trendi import background_removal, Utils, constants
from core.yonatan import gender_detector
import cv2
import os

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

face_image = image[x:(x+w), y:(y-h)]

print face_image
print face_image.type


#gender_detector()