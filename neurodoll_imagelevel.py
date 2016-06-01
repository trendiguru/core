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


def theOldDetector(url_or_np_array,classifier):

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


def detect_many(image_dir,prototxt,caffemodel,dims=(224,224)):
#    image = url_or_np_array_or_filename_to_np_array(imagename)
#    if image is None:
#        logging.warning('couldnt get image')
    net = caffe.Net(prototxt,caffemodel, caffe.TEST)
    filelist = [os.path.join(dir,f) for f in os.listdir(image_dir) if 'jpg' in f]
    results = []
    start_time = time.time()
    for imagename in filelist:
        print('working on:'+imagename)
            # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
        im = Image.open(imagename)
        im = im.resize(dims,Image.ANTIALIAS)
        in_ = np.array(im, dtype=np.float32)
        if len(in_.shape) != 3:
            print('got 1-chan image, skipping')
            return
        elif in_.shape[2] != 3:
            print('got n-chan image, skipping - shape:'+str(in_.shape))
            return
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.0,116.7,122.7))
        in_ = in_.transpose((2,0,1))
        print('shape after:'+str(in_.shape))
        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        out = net.blobs['prob'].data[0]
        print('shape of prob:'+str(net.blobs['prob'].shape))
        print('shape:'+str(out.shape))
        print('out:'+str(out))
        results.append(out[0])
    print('elapsed time:'+str(time.time()-start_time))
    return results

def old_detect_many(imgdir):
    image_dims = [227, 227]
    mean = np.array([107, 117, 123])
    image_mean = np.array([107.0,117.0,123.0])
    input_scale = None
    channel_swap = [2, 1, 0]
    channel_swap = None
    raw_scale = 255.0

    filelist = [os.path.join(dir,f) for f in os.listdir(imgdir) if 'jpg' in f]
    filelist.sort()
    print('loading caffemodel for neurodoll')
    for cpm in caffe_protos_models:
        print('proto {} model {}'.format(cpm[0],cpm[1]))
#        net = caffe.Net(cpm[0],cpm[1], caffe.TEST)
        classifier = caffe.Classifier(cpm[0],cpm[1],
                                  image_dims=image_dims, mean=mean,
                                  input_scale=input_scale, raw_scale=raw_scale,
                                  channel_swap=channel_swap)

        for f in filelist:
            res,probs = theDetector(f,classifier)
            print res

if __name__ == "__main__":

    url = 'http://diamondfilms.com.au/wp-content/uploads/2014/08/Fashion-Photography-Sydney-1.jpg'
    dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/test_256x256_novariations'
    filelist = [os.path.join(dir,f) for f in os.listdir(dir) if 'jpg' in f]
    filelist.sort()
    n = len(filelist)
    allresults = np.zeros([n,4])
    agg = []
    i=0
    for cpm in caffe_protos_models:

        results = detect_many(dir,cpm[0],cpm[1])
        allresults[:,i] = results
#        agg.append(results)
        i=i+1

    print allresults
    for i in range(n):
        print(filelist[i],' ',allresults[i,:])
#    print agg

#    result = infer_one(url,required_image_size=required_image_size)
#    cv2.imwrite('output.png',result)
#    labels=constants.ultimate_21
#    imutils.show_mask_with_labels('output.png',labels,visual_output=True)


