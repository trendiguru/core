#!/usr/bin/env python

import numpy as np
import cv2
import requests
import skimage
import dlib

# # uncomment the line below only if you want to use dlib_face_detector
detector = dlib.get_frontal_face_detector()


def url_to_np_array(url_or_np_array):
    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        try:
            response = requests.get(url_or_np_array, timeout=10)  # download
            image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
        except:
            print "couldn't open link"
            return False
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    else:
        return False

    return image


def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


def find_face_dlib(image, max_num_of_faces=10):
    # faces, scores, idx = detector.run(image, 1, -1) - gives more results, those that add low confidence percentage
    # faces, scores, idx = detector.run(image, 1, 1) - gives less results, doesn't show the lowest confidence percentage results
    # i can get only the faces locations with: faces = detector(image, 1)

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    # can also put 0.

    # if needed: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector(image, 1)
    faces = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(faces)]
    if not len(faces):
        return {'are_faces': False, 'faces': []}

    return {'are_faces': len(faces) > 0, 'faces': faces}

