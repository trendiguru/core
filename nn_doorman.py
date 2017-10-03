__author__ = 'jeremy'

import logging

import numpy as np
from rq import Queue
import cv2
import time
from trendi import constants
from trendi import Utils
#from trendi.utils_tg import imutils

redis_conn = constants.redis_conn
TTL = constants.general_ttl
# Tell RQ what Redis connection to use

#!/usr/bin/env python

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

MODEL_FILE = "/home/jyonatanneuro_doorman/deploy.prototxt"
PRETRAINED = "/home/jyonatanneuro_doorman/_iter_8078.caffemodel"
caffe.set_mode_gpu()
image_dims = [227, 227]
#m ean = np.array([107,117,123])
# the training was without mean subtraction
mean = None
input_scale = None
channel_swap = [2, 1, 0]
raw_scale = 255.0

# Make classifier.
classifier = caffe.Classifier(MODEL_FILE, PRETRAINED,
                              image_dims=image_dims, mean=mean,
                              input_scale=input_scale, raw_scale=raw_scale,
                              channel_swap=channel_swap)


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format

    if url.count('jpg') > 1:
        return None

    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if image.size == 0:
        return None
    new_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return new_image


def theDetector(url_or_np_array):

    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    else:
        return None

    print('shape: '+str(image.shape))
    if not len(image):
        return 'None'

    # Classify.
    start = time.time()
    predictions = classifier.predict(image)

    print("predictions %s Done in %.2f s." % (str(predictions), (time.time() - start)))

    if predictions[0][1] > predictions[0][0]:
        print predictions[0][1]
        # relevant
        return True
    else:
        print predictions[0][0]
        # irrelevant
        return False



def nn_doorman_enqueue(img_url_or_cv2_array, async=False):
    """
    The 'doorman queue' which starts engines and keeps them warm is 'neurodoor'.  This worker should be running somewhere (ideally in a screen named something like 'doorman').
    :param img_url_or_cv2_array: the image/url
    """
    queue_name = constants.neurodoorman_queuename
    queue = Queue(queue_name, connection=redis_conn)
    job1 = queue.enqueue_call(func='trendi.nn_doorman.none_may_pass',
                              args=(img_url_or_cv2_array),
                              ttl=TTL, result_ttl=TTL, timeout=TTL)
    if isinstance(img_url_or_cv2_array, basestring):
        url = img_url_or_cv2_array
    else:
        url = None
    print('started doorman job on queue:'+str(queue)+' url:'+str(url))
    start = time.time()
    if not async:
        print('running synchronously (waiting for result)'),
        elapsed_time=0.0
        while job1.result is None:
            time.sleep(0.1)
            print('.'),
            elapsed_time = time.time()-start
            if elapsed_time>constants.paperdoll_ttl :
                if isinstance(img_url_or_cv2_array,basestring):
                    print('timeout waiting for pd.get_parse_mask, url='+img_url_or_cv2_array)
                    raise Exception('paperdoll timed out on this file:',img_url_or_cv2_array)
                else:
                    print('timeout waiting for pd.get_parse_mask, img_arr given')
                    raise Exception('paperdoll timed out with an img arr')
            return
        print('')
        print('elapsed time in paperdoll_enqueue:'+str(elapsed_time))
        #todo - see if I can return the parse, pose etc here without breaking anything (aysnc version gets job1 back so there may be expectation of getting a job instead of tuple
    else:
        print('running asynchronously (not waiting for result)')
    return job1
