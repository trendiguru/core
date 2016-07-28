#!/usr/bin/env python

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
from trendi import pipeline

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



def infer_one(url_or_np_array,required_image_size=(256,256)):
    start_time = time.time()
    if isinstance(url_or_np_array, basestring):
        print('infer_one working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
        # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#    im = Image.open(imagename)
#    im = im.resize(required_imagesize,Image.ANTIALIAS)

#    in_ = in_.astype(float)
    in_ = imutils.resize_keep_aspect(image,output_size=required_image_size,output_file=None)
    in_ = np.array(in_, dtype=np.float32)   #.astype(float)
    if len(in_.shape) != 3:
        print('got 1-chan image, turning into 3 channel')
        #DEBUG THIS , ORDER MAY BE WRONG
        in_ = np.array([in_,in_,in_])
    elif in_.shape[2] != 3:
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
#    in_ = in_[:,:,::-1]  for doing RGB -> BGR
#    cv2.imshow('test',in_)
    in_ -= np.array((104,116,122.0))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    result = Image.fromarray(out.astype(np.uint8))
#        outname = im.strip('.png')[0]+'out.bmp'
#    outname = os.path.basename(imagename)
#    outname = outname.split('.jpg')[0]+'.bmp'
#    outname = os.path.join(out_dir,outname)
#    print('outname:'+outname)
#    result.save(outname)
    #        fullout = net.blobs['score'].data[0]
    elapsed_time=time.time()-start_time
    print('infer_one elapsed time:'+str(elapsed_time))
 #   cv2.imshow('out',out.astype(np.uint8))
 #   cv2.waitKey(0)
    return out.astype(np.uint8)

def get_category_graylevel(url_or_np_array,category_index,required_image_size=(256,256)):
    start_time = time.time()
    if isinstance(url_or_np_array, basestring):
        print('infer_one working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#    im = Image.open(imagename)
#    im = im.resize(required_imagesize,Image.ANTIALIAS)
#    in_ = in_.astype(float)
    in_ = imutils.resize_keep_aspect(image,output_size=required_image_size,output_file=None)
    in_ = np.array(in_, dtype=np.float32)   #.astype(float)
    if len(in_.shape) != 3:  #h x w x channels, will be 2 if only h x w
        print('got 1-chan image, turning into 3 channel')
        #DEBUG THIS , ORDER MAY BE WRONG [what order? what was i thinking???]
        in_ = np.array([in_,in_,in_])
    elif in_.shape[2] != 3:  #for rgb/bgr, some imgages have 4 chan for alpha i guess
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
#    in_ = in_[:,:,::-1]  for doing RGB -> BGR
#    cv2.imshow('test',in_)
    in_ -= np.array((104,116,122.0))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
#    out = net.blobs['score'].data[0].argmax(axis=0) #for a parse with per-pixel max
    out = net.blobs['siggy'].data[0][category_index] #for the nth class layer #siggy is after sigmoid
    min = np.min(out)
    max = np.max(out)
    print('min {} max {} out shape {}'.format(min,max,out.shape))
    out = out*255
    min = np.min(out)
    max = np.max(out)
    print('min {} max {} out after scaling  {}'.format(min,max,out.shape))
    result = Image.fromarray(out.astype(np.uint8))
#        outname = im.strip('.png')[0]+'out.bmp'
#    outname = os.path.basename(imagename)
#    outname = outname.split('.jpg')[0]+'.bmp'
#    outname = os.path.join(out_dir,outname)
#    print('outname:'+outname)
#    result.save(outname)
    #        fullout = net.blobs['score'].data[0]
    elapsed_time=time.time()-start_time
    print('infer_one elapsed time:'+str(elapsed_time))
 #   cv2.imshow('out',out.astype(np.uint8))
 #   cv2.waitKey(0)
    return out.astype(np.uint8)

def get_final_activation(url_or_np_array,category_index,required_image_size=(256,256)):
    #NOTE THIS IS NOT FINISHED , get correct layer if you want to continue - moving to multilabel instead
    start_time = time.time()
    if isinstance(url_or_np_array, basestring):
        print('infer_one working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#    im = Image.open(imagename)
#    im = im.resize(required_imagesize,Image.ANTIALIAS)
#    in_ = in_.astype(float)
    in_ = imutils.resize_keep_aspect(image,output_size=required_image_size,output_file=None)
    in_ = np.array(in_, dtype=np.float32)   #.astype(float)
    if len(in_.shape) != 3:  #h x w x channels, will be 2 if only h x w
        print('got 1-chan image, turning into 3 channel')
        #DEBUG THIS , ORDER MAY BE WRONG [what order? what was i thinking???]
        in_ = np.array([in_,in_,in_])
    elif in_.shape[2] != 3:  #for rgb/bgr, some imgages have 4 chan for alpha i guess
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
#    in_ = in_[:,:,::-1]  for doing RGB -> BGR
#    cv2.imshow('test',in_)
    in_ -= np.array((104,116,122.0))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
#    out = net.blobs['score'].data[0].argmax(axis=0) #for a parse with per-pixel max
    out = net.blobs['siggy'].data[0][category_index] #for the nth class layer #siggy is after sigmoid
    min = np.min(out)
    max = np.max(out)
    print('min {} max {} out shape {}'.format(min,max,out.shape))
    out = out*255
    min = np.min(out)
    max = np.max(out)
    print('min {} max {} out after scaling  {}'.format(min,max,out.shape))
    result = Image.fromarray(out.astype(np.uint8))
#        outname = im.strip('.png')[0]+'out.bmp'
#    outname = os.path.basename(imagename)
#    outname = outname.split('.jpg')[0]+'.bmp'
#    outname = os.path.join(out_dir,outname)
#    print('outname:'+outname)
#    result.save(outname)
    #        fullout = net.blobs['score'].data[0]
    elapsed_time=time.time()-start_time
    print('infer_one elapsed time:'+str(elapsed_time))
 #   cv2.imshow('out',out.astype(np.uint8))
 #   cv2.waitKey(0)
    return out.astype(np.uint8)


print('starting multilabel.py')
protopath = os.path.join(os.path.dirname(os.path.abspath( __file__ )), 'classifier_stuff/caffe_nns/protos')
modelpath = '/home/jeremy/caffenets/production'
solverproto = os.path.join(modelpath,'ResNet-101-test.prototxt'
deployproto = os.path.join(modelpath,'ResNet-101-deploy.prototxt'
caffemodel = os.path.join(modelpath,'multilabel_resnet101_sgd_iter_120000.caffemodel'


MODEL_FILE = "/home/jeremy/voc8_15_pixlevel_deploy.prototxt"
SINGLE_CLASS_LAYER_DEPLOY = "/home/jeremy/voc8_15_pixlevel_deploy_with_sigmoid.prototxt"
PRETRAINED = "/home/jeremy/voc8_15_pixlevel_iter120000.caffemodel"
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(SINGLE_CLASS_LAYER_DEPLOY,PRETRAINED, caffe.TEST)

required_image_size = (256, 256)
image_mean = np.array([107.0,117.0,123.0])
input_scale = None
channel_swap = [2, 1, 0]
raw_scale = 255.0

print('loading caffemodel for neurodoll (single class layers)')


# Make classifier.
#classifier = caffe.Classifier(MODEL_FILE, PRETRAINED,
#                              image_dims=image_dims, mean=mean,
##                              input_scale=input_scale, raw_scale=raw_scale,
 #                             channel_swap=channel_swap)



if __name__ == "__main__":

    do_category = True
    if(do_category):
        outmat = np.zeros([256*4,256*21],dtype=np.uint8)
        url = 'http://diamondfilms.com.au/wp-content/uploads/2014/08/Fashion-Photography-Sydney-1.jpg'
        url = 'http://pinmakeuptips.com/wp-content/uploads/2015/02/1.4.jpg'

        for index_to_show in range(0,21):
            result = get_category_graylevel(url,index_to_show)

            t1,im1 = cv2.threshold(result,60,255,cv2.THRESH_BINARY)
            t2,im2 = cv2.threshold(result,127,255,cv2.THRESH_BINARY)
            t3,im3 = cv2.threshold(result,180,255,cv2.THRESH_BINARY)

            t4 = cv2.adaptiveThreshold(result,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            t5,im5 = cv2.threshold(result,128,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

            outmat[0:256,256*index_to_show:256*(index_to_show+1)] = result
            outmat[256:256*2,256*index_to_show:256*(index_to_show+1)] = im1
            outmat[256*2:256*3,256*index_to_show:256*(index_to_show+1)] = t4
            outmat[256*3:256*4,256*index_to_show:256*(index_to_show+1)] = im5

            cv2.imwrite('output.png',outmat)
#            cv2.imshow('output layer'+str(index_to_show),result)
#            cv2.waitKey(0)
        cv2.imshow('output layers',outmat)
        cv2.waitKey(0)
#            labels=constants.ultimate_21
#            imutils.show_mask_with_labels('output.png',labels,visual_output=True)


    else:
        url = 'http://diamondfilms.com.au/wp-content/uploads/2014/08/Fashion-Photography-Sydney-1.jpg'
        url = 'http://pinmakeuptips.com/wp-content/uploads/2015/02/1.4.jpg'
        result = infer_one(url,required_image_size=required_image_size)
        cv2.imwrite('output.png',result)
        labels=constants.ultimate_21
        imutils.show_mask_with_labels('output.png',labels,visual_output=True)

#    after_nn_result = pipeline.after_nn_conclusions(result,constants.ultimate_21_dict)
#    cv2.imwrite('output_afternn.png',after_nn_result)
#   labels=constants.ultimate_21
