#!/usr/bin/env python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil
import yonatan_constants
from .. import background_removal, utils, constants
import sys
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
from ..utils import imutils
import argparse
import glob
import time
import skimage
import urllib
from PIL import Image
import pymongo
import dlib
import requests


detector = dlib.get_frontal_face_detector()


def person_isolation(image, face):
    x, y, w, h = face
    image_copy = np.zeros(image.shape, dtype=np.uint8)
    x_back = np.max([x - 1.5 * w, 0])
    x_ahead = np.min([x + 2.5 * w, image.shape[1] - 2])
    image_copy[:, int(x_back):int(x_ahead), :] = image[:, int(x_back):int(x_ahead), :]
    return image_copy


def collar_isolation(image, face):
    x, y, w, h = face

    x_back = np.max([x - 0.75 * w, 0])
    x_ahead = np.min([x + 1.75 * w, image.shape[1] - 2])

    print image.shape

    y_top = np.max([y - 0.3 * h, 0])
    y_bottom = np.min([y + 2 * h, image.shape[0] - 2])

    image_copy = np.zeros((int(y_bottom) - int(y_top), int(x_ahead) - int(x_back), 3), dtype=np.uint8)

    image_copy[...] = image[int(y_top):int(y_bottom), int(x_back):int(x_ahead), :]
    return image_copy


def find_that_face(image, max_num_of_faces=10):
    faces = detector(image, 1)
    faces = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(faces)]
    if not len(faces):
        return {'are_faces': False, 'faces': []}
    return {'are_faces': len(faces) > 0, 'faces': faces}


def crop_person_figure_by_face(url_or_np_array):

    print "Starting the cropping!"
    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        #full_image = url_to_image(url_or_np_array)
        response = requests.get(url_or_np_array)  # download
        full_image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    elif type(url_or_np_array) == np.ndarray:
        full_image = url_or_np_array
    else:
        return None

    #checks if the face coordinates are inside the image
    if full_image is None:
        print "not a good image"
        return None

    #resized_image = imutils.resize_keep_aspect(full_image, output_size=(124, 124))

    faces = background_removal.find_face_dlib(full_image)

    if not faces["are_faces"]:
        print "didn't find any faces"
        return None

    print faces["faces"][0] # just checking if the face that found seems in the right place

    height, width, channels = full_image.shape

    person_figure = person_isolation(full_image, faces["faces"][0])
    collar_figure = collar_isolation(full_image, faces["faces"][0])

    print cv2.imwrite('person_figure.jpg', person_figure)
    print cv2.imwrite('collar_figure.jpg', collar_figure)

