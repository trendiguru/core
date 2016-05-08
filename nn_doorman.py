__author__ = 'jeremy'

import logging

import numpy as np
from rq import Queue
import cv2
import time
from trendi import constants
from trendi import Utils
#from trendi.utils import imutils

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


MODEL_FILE = "/home/jeremy/caffenets/neuro_doorman/deploy.prototxt"
PRETRAINED = "/home/jeremy/caffenets/neuro_doorman/_iter_8078.caffemodel"
caffe.set_mode_gpu()
image_dims = [227, 227]
#mean = np.array([107,117,123])
# the training was without mean subtraction
mean = None
input_scale = None
channel_swap = [2, 1, 0]
raw_scale = 255.0
# ext = 'jpg'

# Make classifier.
classifier = caffe.Classifier(MODEL_FILE, PRETRAINED,
                              image_dims=image_dims, mean=mean,
                              input_scale=input_scale, raw_scale=raw_scale,
                              channel_swap=channel_swap)


#def genderator(argv):
def theDetector(image):
#def theDetector(image, coordinates):

    #input_image = sys.argv[1]
    #input_image = image[coordinates[1]: coordinates[1] + coordinates[3], coordinates[0]: coordinates[0] + coordinates[2]]
    input_file = os.path.expanduser(image)
    #print("Loading file: %s" % input_file)
    #inputs = Utils.get_cv2_img_array(input_file)
    inputs = [caffe.io.load_image(input_file)]

    print('shape: '+str(inputs.shape))
    if not len(inputs):
        return 'None'

    # Classify.
    start = time.time()
    predictions = classifier.predict(inputs)

    print("predictions %s Done in %.2f s." % (str(predictions),(time.time() - start)))

    if predictions[0][1] > 0.7:
        print predictions[0][1]
        print "relevant!"
        return 'relevant'
    else:
        print predictions[0][0]
        print "it's irrelevant!"
        return 'irrelevant'



def nn_doorman_enqueue(img_url_or_cv2_array,async=False):
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
