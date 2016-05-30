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



def infer_many(images,prototxt,caffemodel,out_dir='./'):
    net = caffe.Net(prototxt,caffemodel, caffe.TEST)
    dims = [150,100]
    start_time = time.time()
    masks=[]
    for imagename in images:
        print('working on:'+imagename)
            # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
        im = Image.open(imagename)
        im = im.resize(dims,Image.ANTIALIAS)
        in_ = np.array(im, dtype=np.float32)
        if len(in_.shape) != 3:
            print('got 1-chan image, skipping')
            continue
        elif in_.shape[2] != 3:
            print('got n-chan image, skipping - shape:'+str(in_.shape))
            continue
        print('size:'+str(in_.shape))
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.0,116.7,122.7))
        in_ = in_.transpose((2,0,1))
        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        out = net.blobs['score'].data[0].argmax(axis=0)
        result = Image.fromarray(out.astype(np.uint8))
    #        outname = im.strip('.png')[0]+'out.bmp'
        outname = os.path.basename(imagename)
        outname = outname.split('.jpg')[0]+'.bmp'
        outname = os.path.join(out_dir,outname)
        print('outname:'+outname)
        result.save(outname)
        masks.append(out.astype(np.uint8))
    elapsed_time=time.time()-start_time
    print('elapsed time:'+str(elapsed_time)+' tpi:'+str(elapsed_time/len(images)))
    return masks
    #fullout = net.blobs['score'].data[0]


def infer_one(url_or_np_array,net,required_imagesize=(256,256)):
    start_time = time.time()
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    print('working on:'+imagename)
        # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(imagename)
    im = im.resize(required_imagesize,Image.ANTIALIAS)
    in_ = np.array(im, dtype=np.float32)
    if len(in_.shape) != 3:
        print('got 1-chan image, skipping')
        return
    elif in_.shape[2] != 3:
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
    print('shape before:'+str(in_.shape))
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.0,116.7,122.7))
    in_ = in_.transpose((2,0,1))
    print('shape after:'+str(in_.shape))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    result = Image.fromarray(out.astype(np.uint8))
#        outname = im.strip('.png')[0]+'out.bmp'
    outname = os.path.basename(imagename)
    outname = outname.split('.jpg')[0]+'.bmp'
    outname = os.path.join(out_dir,outname)
    print('outname:'+outname)
    result.save(outname)
    #        fullout = net.blobs['score'].data[0]
    elapsed_time=time.time()-start_time
    print('elapsed time:'+str(elapsed_time))
    return out.astype(np.uint8)



MODEL_FILE = "/home/jeremy/voc8_15_pixlevel_deploy.prototxt"
PRETRAINED = "/home/jeremy/voc8_15_pixlevel_iter120000.caffemodel"
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(MODEL_FILE,PRETRAINED, caffe.TEST)

required_image_size = (256, 256)
image_mean = np.array([107.0,117.0,123.0])
input_scale = None
channel_swap = [2, 1, 0]
raw_scale = 255.0

print('loading caffemodel for neurodoll')


# Make classifier.
#classifier = caffe.Classifier(MODEL_FILE, PRETRAINED,
#                              image_dims=image_dims, mean=mean,
##                              input_scale=input_scale, raw_scale=raw_scale,
 #                             channel_swap=channel_swap)



if __name__ == "__main__":

    url = 'http://diamondfilms.com.au/wp-content/uploads/2014/08/Fashion-Photography-Sydney-1.jpg'
    infer_one(url,net,required_imagesize=required_image_size)