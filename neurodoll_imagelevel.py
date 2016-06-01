#!/usr/bin/env python
__author__ = 'jeremy'

from PIL import Image
import cv2
import caffe
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import os
import time
import urllib

from trendi import background_removal, Utils, constants
from trendi.utils import imutils

caffe_protos_models = [["/home/jeremy/caffenets/binary_imagelevel/caf/dressdeploy.prototxt","/home/jeremy/caffenets/binary_imagelevel/caf/caffe_binary_dress_iter_30000.caffemodel"],
               ["/home/jeremy/caffenets/binary_imagelevel/caf/pantsdeploy.prototxt","/home/jeremy/caffenets/binary_imagelevel/caf/caffe_binary_pants_iter_11638.caffemodel"],
               ["/home/jeremy/caffenets/binary_imagelevel/caf/skirtdeploy.prototxt", "/home/jeremy/caffenets/binary_imagelevel/caf/caffe_binary_skirt_iter_12395.caffemodel"],
               ["/home/jeremy/caffenets/binary_imagelevel/caf/topdeploy.prototxt", "/home/jeremy/caffenets/binary_imagelevel/caf/caffe_binary_top_iter_21891.caffemodel"]]

caffe.set_mode_gpu()
caffe.set_device(1);


def cv2_image_to_caffe(image):
    return skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)


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

def url_or_np_array_or_filename_to_np_array(url_or_np_array_or_filename):

    if isinstance(url_or_np_array_or_filename, basestring):
        if 'http://' in url_or_np_array_or_filename: #its a url
            logging.debug('getting image from url:'+url_or_np_array_or_filename)
            image = url_to_image(url_or_np_array_or_filename)
        else:
            logging.debug('getting image from file:'+url_or_np_array_or_filename)
            image = cv2.imread(url_or_np_array_or_filename)
        return image
    elif type(url_or_np_array_or_filename) == np.ndarray:
        logging.debug('getting image from array')
        image = url_or_np_array_or_filename
        return image
    else:
        return None


def theDetector(url_or_np_array,classifier):

    image = url_or_np_array_or_filename_to_np_array(url_or_np_array)
    if image is None:
        logging.warning('couldnt get image')
    #image_for_caffe = [caffe.io.load_image(image)]
#    image_for_caffe = [cv2_image_to_caffe(image)]   #skip the double channelswap
    image_for_caffe = [image]

    if image_for_caffe is None:
        return None

    # Classify.
    start = time.time()
    predictions = classifier.predict(image_for_caffe)

    print("predictions %s Done in %.2f s." % (str(predictions), (time.time() - start)))
    print('pred 0,1:'+str(predictions[0][0]),str(predictions[0][1]))

    if predictions[0][1] > predictions[0][0]:
        # relevant
        return True, [predictions[0][0],predictions[0][1]]
    else:
        # irrelevant
        return False,  [predictions[0][0],predictions[0][1]]

def detect_many(imgdir):
    image_dims = [227, 227]
    mean = np.array([107, 117, 123])
    image_mean = np.array([107.0,117.0,123.0])
    input_scale = None
    channel_swap = [2, 1, 0]
    channel_swap = None
    raw_scale = 255.0

    filelist = [os.path.join(dir,f) for f in os.listdir(imgdir) if 'jpg' in f]

    print('loading caffemodel for neurodoll')
    for cpm in caffe_protos_models:
#        net = caffe.Net(cpm[0],cpm[1], caffe.TEST)
        classifier = caffe.Classifier(cpm[0],cpm[1],
                                  image_dims=image_dims, mean=mean,
                                  input_scale=input_scale, raw_scale=raw_scale,
                                  channel_swap=channel_swap)

        for f in filelist:
            res = theDetector(f,classifier)
            print res

if __name__ == "__main__":

    url = 'http://diamondfilms.com.au/wp-content/uploads/2014/08/Fashion-Photography-Sydney-1.jpg'
    dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/test_256x256_novariations'
    detect_many(dir)
#    result = infer_one(url,required_image_size=required_image_size)
#    cv2.imwrite('output.png',result)
#    labels=constants.ultimate_21
#    imutils.show_mask_with_labels('output.png',labels,visual_output=True)


