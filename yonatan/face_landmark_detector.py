#!/usr/bin/env python

__author__ = 'yonatan_guy'

import numpy as np
import caffe
import time
import cv2
import skimage
import requests
import dlib
import yonatan_classifier
from imutils import face_utils
import imutils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor()

# MODLE_FILE = "/data/yonatan/yonatan_files/trendi/yonatan/resnet_50_gender_by_face/ResNet-50-deploy.prototxt"
# PRETRAINED = "/data/yonatan/yonatan_caffemodels/genderator_caffemodels/caffe_resnet50_snapshot_sgd_genfder_by_face_iter_10000.caffemodel"
# caffe.set_mode_gpu()
# caffe.set_device(0)  # choose GPU number
# image_dims = [224, 224]
# mean, input_scale = np.array([120, 120, 120]), None
# # channel_swap = [:, :, (2, 1, 0)]
# channel_swap = None
# raw_scale = 255.0
#
# # Make classifier.
# classifier = yonatan_classifier.Classifier(MODLE_FILE, PRETRAINED,
#                               image_dims=image_dims, mean=mean,
#                               input_scale=input_scale, raw_scale=raw_scale,
#                               channel_swap=channel_swap)
#
# print "Done initializing!"


def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


def find_face_dlib(image, max_num_of_faces=10):
    faces_orig = detector(image, 1)
    faces_list = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(faces_orig)]
    if not len(faces_list):
        return {'are_faces': False, 'faces': []}
    return {'are_faces': len(faces_list) > 0, 'faces': faces_list, 'faces_orig': faces_orig}


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    print coords
    print "shape.part(0): {0}, shape.part(1): {1}".format(shape.part(0), shape.part(1))

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        print i

    # return the list of (x, y)-coordinates
    return coords


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
        shape = predictor(gray, faces_list[i])
        print "shape.part(0): {0}, shape.part(1): {1}".format(shape.part(0), shape.part(1))

        shape = shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # gender = ""
    # score = 0
    # frame_and_text_color = (0, 0, 0)
    #
    # # for k, d in enumerate(faces["faces_orig"]):
    # #
    # #     print "k: {0}".format(k)
    # #     print faces["faces_orig"][k]  # just checking if the face that found seems in the right place
    # #
    # #     height, width, channels = full_image.shape
    # #
    # #     # x, y, w, h = faces["faces_orig"][k]
    # #     #
    # #     # # checks if the face coordinates are inside the image
    # #     # if x > width or x + w > width or y > height or y + h > height:
    # #     #     print "face coordinates are out of image boundary"
    # #     #     continue
    # #
    # #     # face_image = full_image[y: y + h, x: x + w]
    # #
    # #     # resized_face_image = imutils.resize_keep_aspect(face_image, output_size=(224, 224))
    # #
    # #     shape = predictor(full_image, d)
    #
    # # loop over the face detections
    # for (i, rect) in enumerate(faces["faces_orig"]):
    #     # determine the facial landmarks for the face region, then
    #     # convert the facial landmark (x, y)-coordinates to a NumPy
    #     # array
    #     shape = predictor(gray, rect)
    #     shape = shape_to_np(shape)
    #
    #     # convert dlib's rectangle to a OpenCV-style bounding box
    #     # [i.e., (x, y, w, h)], then draw the face bounding box
    #     (x, y, w, h) = face_utils.rect_to_bb(rect)
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    #     # show the face number
    #     cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #
    #     # loop over the (x, y)-coordinates for the facial landmarks
    #     # and draw them on the image
    #     for (x, y) in shape:
    #         cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    #
    # #     face_for_caffe = [cv2_image_to_caffe(face_image)]
    # #     #face_for_caffe = [caffe.io.load_image(face_image)]
    # #
    # #     if face_for_caffe is None:
    # #         print "image to caffe failed"
    # #         continue
    # #
    # #     # Classify.
    # #     start = time.time()
    # #     predictions = classifier.predict(face_for_caffe)
    # #     print("Done in %.2f s." % (time.time() - start))
    # #
    # #     if predictions[0][1] > predictions[0][0]:
    # #         score = predictions[0][1]
    # #         gender = "Male"
    # #         frame_and_text_color = (255, 0, 0)  # blue for men
    # #     else:
    # #         score = predictions[0][0]
    # #         gender = "Female"
    # #         frame_and_text_color = (127, 0, 225)  # pink for women
    # #
    # #     cv2.rectangle(full_image, (x, y), (x + w, y + h), frame_and_text_color, 1 + w / 50)
    # #
    # #     font = cv2.FONT_HERSHEY_SIMPLEX
    # #     cv2.putText(full_image, '{:s} {:.2f}'.format(gender, score), (int(x + 4), int(y + h - 6 - h / 130)), font, float(w) / 230,
    # #                 frame_and_text_color, 1 + w / 170, cv2.LINE_AA)
    # #
    # #     print score
    # #     print gender
    # #
    # # print cv2.imwrite("/data/yonatan/linked_to_web/gender_classifier_results/image1.jpg", full_image)
