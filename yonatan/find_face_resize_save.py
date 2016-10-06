#!/usr/bin/env python

import numpy as np
import os
from PIL import Image
import caffe
from trendi import background_removal, Utils, constants
import cv2
import sys
import argparse
import glob
import time
import dlib
from ..utils import imutils

detector = dlib.get_frontal_face_detector()


# def find_face(image):
#     gray = cv2.cvtColor(image, constants.BGR2GRAYCONST)
#     face_cascades = [
#         cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt2.xml')),
#         cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt.xml')),
#         cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt_tree.xml')),
#         cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_default.xml'))]
#     cascade_ok = False
#     for cascade in face_cascades:
#         if not cascade.empty():
#             cascade_ok = True
#             break
#     if cascade_ok is False:
#         raise IOError("no good cascade found!")
#     faces = []
#     for cascade in face_cascades:
#         faces = cascade.detectMultiScale(
#             gray,
#             scaleFactor=1.1,
#             minNeighbors=2,
#             minSize=(5, 5),
#             flags=constants.scale_flag
#         )
#         if len(faces) > 0:
#             break
#     return faces
#
#
# width = 115
# height = 115
# counter = 0
#
# sets = {'train', 'test'}
#
# for set in sets:
#     if set == 'train':
#         mypath_male = '/home/yonatan/train_set/male'
#         mypath_female = '/home/yonatan/train_set/female'
#     else:
#         mypath_male = '/home/yonatan/test_set/male'
#         mypath_female = '/home/yonatan/test_set/female'
#
#     for root, dirs, files in os.walk(mypath_male):
#         for file in files:
#             if file.endswith(".jpg"):
#                 image = Utils.get_cv2_img_array(os.path.join(root, file))
#                 face = find_face(image)
#
#                 if face != ():
#                     x = face[0][0]
#                     y = face[0][1]
#                     w = face[0][2]
#                     h = face[0][3]
#
#                     face_image = image[y:(y + h), x:(x + w)]
#
#                     resized_image = cv2.resize(face_image, (width, height))
#
#                     cv2.imwrite(os.path.join(root, 'face-' + file), resized_image)
#                     counter += 1
#                     print counter
#                 else:
#                     print "Can't detect face"
#
#
#     for root, dirs, files in os.walk(mypath_female):
#         for file in files:
#             if file.endswith(".jpg"):
#                 image = Utils.get_cv2_img_array(os.path.join(root, file))
#                 face = find_face(image)
#
#                 if face != ():
#                     x = face[0][0]
#                     y = face[0][1]
#                     w = face[0][2]
#                     h = face[0][3]
#
#                     face_image = image[y:(y + h), x:(x + w)]
#
#                     resized_image = cv2.resize(face_image, (width, height))
#
#                     cv2.imwrite(os.path.join(root, 'face-' + file), resized_image)
#                     counter += 1
#                     print counter
#                 else:
#                     print "Can't detect face"
#


def find_that_face(image, max_num_of_faces=10):
    faces = detector(image, 1)
    faces = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(faces)]
    if not len(faces):
        return {'are_faces': False, 'faces': []}
    return {'are_faces': len(faces) > 0, 'faces': faces}


sets = {'male', 'female'}

for set in sets:
    counter = 0
    path = "/home/yonatan/faces_stuff/test_dir"
    for root, dirs, files in os.walk(path):
        for file in files:
            # if file.endswith(".jpg"):
            image_array = cv2.imread(file)
            image_array = image_array.astype(np.uint8)
            face_answer = find_that_face(image_array, 1)

            if face_answer['are_faces'] > 0:
                x, y, w, h = face_answer['faces'][0]

                face_image = image_array[y:(y + h), x:(x + w)]

                resized_image = imutils.resize_keep_aspect(face_image, output_size=(224, 224))

                image_file_name = 'face-' + str(counter) + '.jpg'

                cv2.imwrite(os.path.join(root, image_file_name), resized_image)

                counter += 1
                print counter
            else:
                print "Can't detect face"
            os.remove(os.path.join(root, file))
