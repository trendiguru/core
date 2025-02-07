#!/usr/bin/env python

__author__ = 'yonatan_guy'

import numpy as np
import time
import cv2
# import skimage
import requests
import dlib
from imutils import face_utils
import imutils

detector = dlib.get_frontal_face_detector()
# in allison server
# predictor = dlib.shape_predictor("/data/yonatan/yonatan_files/trendi/yonatan/shape_predictor_68_face_landmarks.dat")
# locally
predictor = dlib.shape_predictor("/home/core/yonatan/shape_predictor_68_face_landmarks.dat")

eyes_landmarks = {38, 39, 41, 42, 44, 45, 47, 48}

eyes_dict = {}


def find_face_dlib(image, max_num_of_faces=10):
    faces_orig = detector(image, 1)
    faces_list = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(faces_orig)]
    if not len(faces_list):
        return {'are_faces': False, 'faces': []}
    return {'are_faces': len(faces_list) > 0, 'faces': faces_list, 'faces_orig': faces_orig}


def detect(url_or_np_array):

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

        # cv2.line(image, eyes_dict[38], eyes_dict[41], (255, 0, 0), 2)
        # cv2.line(image, eyes_dict[39], eyes_dict[42], (255, 0, 0), 2)
        # cv2.line(image, eyes_dict[44], eyes_dict[47], (255, 0, 0), 2)
        # cv2.line(image, eyes_dict[45], eyes_dict[48], (255, 0, 0), 2)

        left_eye_x, left_eye_y = int(np.mean([eyes_dict[38][0], eyes_dict[39][0], eyes_dict[41][0], eyes_dict[42][0]])), int(np.mean([eyes_dict[38][1], eyes_dict[39][1], eyes_dict[41][1], eyes_dict[42][1]]))
        right_eye_x, right_eye_y = int(np.mean([eyes_dict[44][0], eyes_dict[45][0], eyes_dict[47][0], eyes_dict[48][0]])), int(np.mean([eyes_dict[44][1], eyes_dict[45][1], eyes_dict[47][1], eyes_dict[48][1]]))

        cv2.circle(image, (left_eye_x, left_eye_y), 1, (0, 255, 0), -1)
        cv2.circle(image, (right_eye_x, right_eye_y), 1, (0, 255, 0), -1)

        left_eye = np.array((left_eye_x, left_eye_y))
        right_eye = np.array((right_eye_x, right_eye_y))

        print "distance between eyes: {}".format(np.linalg.norm((right_eye) - (left_eye)))
        # print "distance between eyes: {}".format(np.sqrt(np.sum(((right_eye_x, right_eye_y) - (left_eye_x, left_eye_y))**2)))

    # print cv2.imwrite("/data/yonatan/linked_to_web/face_landmarks/image3.jpg", image)

    cv2.namedWindow("Output", cv2.CV_WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    cv2.imshow("Output", image)
    cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # cap.release()
    # cv2.destroyAllWindows()

    # while 1:
    #     cv2.imshow("Output", image)
    #     k = cv2.waitKey(33)
    #     # if k == 97:
    #     if k & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()
    #         cv2.waitKey(1)
    #         break
    #     elif k == -1:  # normally -1 returned,so don't print it
    #         continue
    #     else:
    #         print k  # else print its value

    # if cv2.waitKey(33) == ord('a'):
    #     print "pressed a"
