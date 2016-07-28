__author__ = 'jeremy'

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
#from trendi.paperdoll import neurodoll_falcon_client as nfc

from trendi.paperdoll import neurodoll_falcon_client as nfc


#


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



def get_multilabel_output(url_or_np_array,required_image_size=(227,227)):
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    print('multilabel working on image of shape:'+str(image.shape))
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
#    in_ = in_[:,:,::-1]  for doing RGB -> BGR : since this is loaded nby cv2 its unecessary
#    cv2.imshow('test',in_)
    in_ -= np.array((104,116,122.0))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    multilabel_net.blobs['data'].reshape(1, *in_.shape)
    multilabel_net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    multilabel_net.forward()
#    out = net.blobs['score'].data[0].argmax(axis=0) #for a parse with per-pixel max
    out = multilabel_net.blobs['score'].data[0] #for the nth class layer #siggy is after sigmoid
    min = np.min(out)
    max = np.max(out)
    print('multilabel:  {}'.format(out))
    return out

def grabcut_using_neurodoll_output(url_or_np_array,category_index):
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    print('grabcut working on image of shape:'+str(image.shape))

        #def neurodoll(image, category_idx):
    dic = nfc.pd(image, category_index)
    if not dic['success']:
        return False, []
    neuro_mask = dic['mask']

    nm_size = neuro_mask.shape[0:2]
    image_size = image.shape[0:2]
    if image_size != nm_size:
        image = cv2.resize(image,(nm_size[1],nm_size[0]))
    # rect = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    mask = np.zeros(image.shape[:2], np.uint8)
    #TODO - maybe find something better than median as the threshold
    med = np.median(neuro_mask)
    mask[neuro_mask > med] = 3
    mask[neuro_mask < med] = 2
    try:
        #TODO - try more than 1 grabcut call in itr
        itr = 1
        cv2.grabCut(image, mask, None, bgdmodel, fgdmodel, itr, cv2.GC_INIT_WITH_MASK)
    except:
        print('grabcut exception')
        return False, []
    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype(np.uint8)
    return mask2

def get_multilabel_output_using_nfc(url_or_np_array):
    multilabel_dict = nfc.pd(url, get_multilabel_results=True)
    if not multilabel_dict['success']:
        print('did not get nfc pd result succesfully')
        return
    multilabel_output = multilabel_dict['multilabel_output']
    print('multilabel output:'+str(multilabel_output))
    return multilabel_output

def combine_neurodoll_and_multilabel(url_or_np_array,multilabel_threshold=0.7):
#    multilabel = get_multilabel_output(url_or_np_array)
    multilabel = get_multilabel_output_using_nfc(url_or_np_array)
    #take only labels above a threshold on the multilabel result
    #possible other way to do this: multiply the neurodoll mask by the multilabel result and threshold that product
    thresholded_multilabel = [ml>multilabel_threshold for ml in multilabel]
#    print('orig label:'+str(multilabel))
    print('thrs label:'+str(thresholded_multilabel))
    for i in range(len(thresholded_multilabel)):
        if thresholded_multilabel[i]:
            neurodoll_index = constants.web_tool_categories_v1_to_ultimate_21[i]
            print('index {} webtoollabel {} newindex {} neurodoll_label {} was above threshold {}'.format(
                i,constants.web_tool_categories[i],neurodoll_index,constants.ultimate_21[neurodoll_index], multilabel_threshold))
            item_mask = grabcut_using_neurodoll_output(url_or_np_array,neurodoll_index)
            gc_mask = grabcut_using_neurodoll_output(url_or_np_array,category_index)

            cv2.imshow('mask '+str(i),item_mask)
            cv2.imshow('gc_mask '+str(i),gc_mask)
            cv2.waitKey(0)

# Make classifier.
#classifier = caffe.Classifier(MODEL_FILE, PRETRAINED,
#                              image_dims=image_dims, mean=mean,
##                              input_scale=input_scale, raw_scale=raw_scale,
 #                             channel_swap=channel_swap)



if __name__ == "__main__":
    outmat = np.zeros([256*4,256*21],dtype=np.uint8)
    url = 'http://diamondfilms.com.au/wp-content/uploads/2014/08/Fashion-Photography-Sydney-1.jpg'
    url = 'http://pinmakeuptips.com/wp-content/uploads/2015/02/1.4.jpg'
#    get_nd_and_multilabel_output_using_nfc(url_or_np_array)
    combine_neurodoll_and_multilabel(url)

    #LOAD NEURODOLL
''' #
MODEL_FILE = "/home/jeremy/voc8_15_pixlevel_deploy.prototxt"
SINGLE_CLASS_LAYER_DEPLOY = "/home/jeremy/voc8_15_pixlevel_deploy_with_sigmoid.prototxt"
PRETRAINED = "/home/jeremy/voc8_15_pixlevel_iter120000.caffemodel"
caffe.set_mode_gpu()
caffe.set_device(1)
print('loading caffemodel for neurodoll (single class layers)')
neurodoll_per_class_net = caffe.Net(SINGLE_CLASS_LAYER_DEPLOY,PRETRAINED, caffe.TEST)
neurodoll_required_image_size = (256, 256)
image_mean = np.array([107.0,117.0,123.0])
input_scale = None
channel_swap = [2, 1, 0]
raw_scale = 255.0

###########LOAD MULTILABELLER
caffemodel =  '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/snapshot/train_iter_340000.caffemodel'
deployproto = '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/deploy.prototxt'
caffe.set_mode_gpu()
caffe.set_device(0)
multilabel_net = caffe.Net(deployproto,caffemodel, caffe.TEST)
multilabel_required_image_size = (227,227)
'''