#!/usr/bin/env python

__author__ = 'yonatan_guy'

import numpy as np
import os
import caffe
import sys
import argparse
import glob
import time
from trendi import background_removal, Utils, constants
import cv2
import urllib
import skimage
import requests


MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_50_dress_sleeve/ResNet-50-deploy.prototxt"
PRETRAINED = "/home/yonatan/resnet50_caffemodels/caffe_resnet50_snapshot_50_sgd_iter_5000.caffemodel"
caffe.set_mode_gpu()
image_dims = [224, 224]
mean, input_scale = np.array([120, 120, 120]), None
channel_swap = [2, 1, 0]
raw_scale = 255.0

# Make classifier.
classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
                              image_dims=image_dims, mean=mean,
                              input_scale=input_scale, raw_scale=raw_scale,
                              channel_swap=channel_swap)

db = constants.db

print "Done initializing!"


def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


dresses = db.yonatan_dresses_test.find({'dress_sleeve_length': {'$exists': 0}})

delete = 0
counter = 0

print "Starting the genderism!"

# text_file = open("all_dresses_" + key + "_list.txt", "w")
for doc in dresses:
    # if i > num_of_each_category:
    #   break
    if 'dress_sleeve_length' not in doc:

        counter += 1

        url_or_np_array = doc['images']['XLarge']

        # check if i get a url (= string) or np.ndarray
        if isinstance(url_or_np_array, basestring):
            #full_image = url_to_image(url_or_np_array)
            response = requests.get(url_or_np_array)  # download
            full_image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
        elif type(url_or_np_array) == np.ndarray:
            full_image = url_or_np_array
        else:
            print "bad picture"
            continue

        #checks if the face coordinates are inside the image
        if full_image is None:
            print "not a good image"
            continue

        face_for_caffe = [cv2_image_to_caffe(full_image)]
        #face_for_caffe = [caffe.io.load_image(face_image)]

        if face_for_caffe is None:
            print "bad picture"
            continue

        # Classify.
        start = time.time()
        predictions = classifier.predict(face_for_caffe)
        print("Done in %.2f s." % (time.time() - start))

        #max_result = max(predictions[0])

        max_result_index = np.argmax(predictions[0])

        predict_label = int(max_result_index)

        db.yonatan_dresses_test.update_one({"_id": doc["_id"]}, {"$set": {"dress_sleeve_length": predict_label}})

        print counter

        '''
        if predict_label == 0:
            type = 'strapless'
        elif predict_label == 1:
            type =  'spaghetti_straps'
        elif predict_label == 2:
            type =  'regular_straps'
        elif predict_label == 3:
            type = 'sleeveless'
        elif predict_label == 4:
            type = 'cap_sleeve'
        elif predict_label == 5:
            type = 'short_sleeve'
        elif predict_label == 6:
            type = 'midi_sleeve'
        elif predict_label == 7:
            type = 'long_sleeve'
        '''

        #print predictions[0][predict_label]

        #and for delete a field from doc:
        #db.yonatan_dresses_test.update({"_id": a[0]["_id"]}, {"$unset": {"dress_sleeve_length": 100}})







# how to delete the field "dress_sleeve_length" from all docs that have it:
#    if db.yonatan_dresses_test.find({"_id": dresses[i]["_id"]}, {'dress_sleeve_length': {'$exists': True}}):
#        db.yonatan_dresses_test.update({"_id": dresses[i]["_id"]}, {"$unset": {"dress_sleeve_length": ""}})
#        delete += 1
#        print delete
#print "delete_num: " + delete