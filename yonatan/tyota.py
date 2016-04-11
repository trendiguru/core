#!/usr/bin/env python

import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
import os
from PIL import Image
import caffe
from trendi import background_removal, Utils, constants
import cv2
import sys
import argparse
import glob
import time
import skimage
import scipy
from resizeimage import resizeimage


def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


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


width = 115
height = 115
counter = 0

sets = {'train', 'test'}

for set in sets:
    if set == 'train':
        mypath_male = '/home/yonatan/train_set/male'
        mypath_female = '/home/yonatan/train_set/female'
    else:
        mypath_male = '/home/yonatan/test_set/male'
        mypath_female = '/home/yonatan/test_set/female'

    for root, dirs, files in os.walk(mypath_male):
        for file in files:
            if file.endswith(".jpg"):
                image = Utils.get_cv2_img_array(os.path.join(root, file))
                face = find_face(image)

                x = face[0][0]
                y = face[0][1]
                w = face[0][2]
                h = face[0][3]

                face_image = image[y:(y + h), x:(x + w)]
                #im = Image.fromarray(face_image)
                #im.save("/home/yonatan/test_set/female/DELETE-Juljia_Vysotskij_0001.jpg")
                cv2.imshow("cropped", face_image)
                cv2.waitKey(0)

                print face_image.shape
                # Open the image file.
                img = Image.fromarray(face_image)

                img = resizeimage.resize_thumbnail(img, (width, height))

                #img = img.resize((width, height), img.BILINEAR)
                print img.shape
                print type(img)
                cv2.imshow("cropped", img)
                cv2.waitKey(0)
                #final_img = Image.fromarray(img)
                # Save it back to disk.
                #final_img.save(os.path.join(root, 'resized_face-' + file))
                #scipy.misc.toimage(img, cmin=0.0, cmax=...).save(os.path.join(root, 'resized_face-' + file))
                scipy.misc.imsave(os.path.join(root, 'resized_face-' + file), img)
                counter += 1
                print counter
                print file
                exit()


    for root, dirs, files in os.walk(mypath_female):
        for file in files:
            if file.endswith(".jpg"):
                image = Utils.get_cv2_img_array(os.path.join(root, file))
                face = find_face(image)

                x = face[0][0]
                y = face[0][1]
                w = face[0][2]
                h = face[0][3]

                face_image = image[y:(y + h), x:(x + w)]
                # im = Image.fromarray(face_image)
                # im.save("/home/yonatan/test_set/female/DELETE-Juljia_Vysotskij_0001.jpg")



                # Open the image file.

                # img = Image.open(os.path.join(root, file))

                # Resize it.
                img = Image.fromarray(face_image)
                img = img.resize((width, height), face_image.BILINEAR)

                # Save it back to disk.
                img.save(os.path.join(root, 'resized_face-' + file))
                counter += 1
                print counter