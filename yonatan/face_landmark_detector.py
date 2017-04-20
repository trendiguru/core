#!/usr/bin/env python

__author__ = 'yonatan_guy'

import numpy as np
import time
import cv2
import skimage
import requests
import dlib
from imutils import face_utils
import imutils

detector = dlib.get_frontal_face_detector()
# in allison server
predictor = dlib.shape_predictor("/data/yonatan/yonatan_files/trendi/yonatan/shape_predictor_68_face_landmarks.dat")
# locally
# predictor = dlib.shape_predictor("/home/core/yonatan/shape_predictor_68_face_landmarks.dat")

eyes_landmarks = {38, 39, 41, 42, 44, 45, 47, 48}

eyes_dict = {}


def find_face_dlib(image, max_num_of_faces=10):
    faces_orig = detector(image, 1)
    faces_list = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(faces_orig)]
    if not len(faces_list):
        return {'are_faces': False, 'faces': []}
    return {'are_faces': len(faces_list) > 0, 'faces': faces_list, 'faces_orig': faces_orig}


def detect(url_or_np_array):

    print "Starting the genderism!"
    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        try:
            response = requests.get(url_or_np_array, timeout=10)  # download
            full_image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
        except:
            print "couldn't open link"
            return None
    elif type(url_or_np_array) == np.ndarray:
        full_image = url_or_np_array
    else:
        return None

    if full_image is None:
        print "not a good image"
        return None

    # faces = find_face_dlib(full_image, 1)
    #
    # if not faces["are_faces"]:
    #     print "didn't find any faces"
    #     return None

    image = imutils.resize(full_image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    faces_list = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(rects)]
    if not len(faces_list):
        print "didn't find a face!"
        return

    print "rects: {}".format(faces_list)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for j, (x, y) in enumerate(shape):
            if j + 1 in eyes_landmarks:
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
                eyes_dict[j+1] = (x,y)
            else:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        cv2.line(image, eyes_dict[38], eyes_dict[41], (255, 0, 0), 5)
        cv2.line(image, eyes_dict[39], eyes_dict[42], (255, 0, 0), 5)
        cv2.line(image, eyes_dict[44], eyes_dict[47], (255, 0, 0), 5)
        cv2.line(image, eyes_dict[45], eyes_dict[48], (255, 0, 0), 5)

    print cv2.imwrite("/data/yonatan/linked_to_web/face_landmarks/image3.jpg", image)

    # cv2.imshow("Output", image)
    # cv2.waitKey(0)
