#!/usr/bin/env python
# addr in constants (http://52.174.49.180:8080/nd)

__author__ = 'jeremy'

import caffe
import copy
from PIL import Image
import cv2
import caffe
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
import os
import urllib
import sys
import hashlib
import time
import pdb
import requests
import json
import operator
import gevent
from gevent import socket


from trendi import constants
from trendi.utils import imutils
from trendi import Utils
from trendi.paperdoll import binary_multilabel_falcon_client as bmfc
from trendi.paperdoll import binary_multilabel_falcon_client2 as bmfc2
from trendi.paperdoll import binary_multilabel_falcon_client3 as bmfc3
from trendi.paperdoll import hydra_tg_falcon_client
from trendi.paperdoll import neurodoll_falcon_client as nfc
from trendi.downloaders import label_conversions
from trendi.classifier_stuff import index_htmls
#REQUIREMENTS FOR THIS TO RUN
#These files have to exist
#modelpath = '/home/jeremy/caffenets/production/ResNet-101-test.prototxt'
#solverproto = '/home/jeremy/caffenets/production/ResNet-101-test.prototxt')
#deployproto = '/home/jeremy/caffenets/production/ResNet-101-deploy.prototxt')
#caffemodel = '/home/jeremy/caffenets/production/multilabel_resnet101_sgd_iter_200000.caffemodel')  #10Aug2016

###########################
#load model into net object
###########################
#MODEL_FILE = "/home/jeremy/voc8_15_pixlevel_deploy.prototxt"
#SINGLE_CLASS_LAYER_DEPLOY = "/home/jeremy/voc8_15_pixlevel_deploy_with_sigmoid.prototxt"
#PRETRAINED = "/home/jeremy/voc8_15_pixlevel_iter120000.caffemodel"
print('loading caffemodel for neurodoll')
#protopath = os.path.join(os.path.dirname(os.path.abspath( __file__ )), 'classifier_stuff/caffe_nns/protos')
#modelpath = '/data/production/caffemodels_and_protos/neurodoll/'
modelpath = os.path.basename(constants.pixlevel_v3_caffemodel_info['caffemodel'])
#modelpath = '/home/jeremy/caffenets/production'
#MODEL_FILE = os.path.join(modelpath,'voc8_15_pixlevel_deploy.prototxt')
#MODEL_FILE = os.path.join(modelpath,'voc8_15_pixlevel_deploy_with_sigmoid.prototxt')
#MODEL_FILE = os.path.join(modelpath,'sharp5_pixlevel_deploy_with_sigmoid.prototxt')
MODEL_FILE = constants.pixlevel_v3_caffemodel_info['prototxt']
#PRETRAINED = os.path.join(modelpath,'voc8_15_pixlevel_iter120000.caffemodel')
#PRETRAINED = os.path.join(modelpath,'voc8_15_0816_iter10000_pixlevel_deploy.caffemodel')
#PRETRAINED = os.path.join(modelpath,'sharp5_all_bn_iter_32000.caffemodel')
PRETRAINED = constants.pixlevel_v3_caffemodel_info['caffemodel']
LABELS = constants.pixlevel_v3_caffemodel_info['labels']
OUTPUT_LAYER = constants.pixlevel_v3_caffemodel_info['output_layer']
#OUTPUT_LAYER = 'output2'

test_on = False
test_gpu=1
production_gpu=1
if test_on:
    if len(sys.argv)>1:
        try:
            gpu = int(sys.argv[1])
        except:
            gpu=test_gpu
    else:
        gpu = test_gpu
    print('using gpu '+str(gpu))
else:
    gpu = production_gpu
caffe.set_mode_gpu()
caffe.set_device(gpu)
net = caffe.Net(MODEL_FILE,caffe.TEST, weights = PRETRAINED)
image_mean = np.array([107.0,117.0,123.0])
input_scale = None
channel_swap = [2, 1, 0]
raw_scale = 255.0
print('done loading caffemodel for neurodoll, output layer '+str(OUTPUT_LAYER))

#best multilabel as of 260716, see http://extremeli.trendi.guru/demo/results/ for updates
multilabel_from_binaries = False
multilabel_from_hydra = True
if not (multilabel_from_binaries or multilabel_from_hydra): #dont need this if answers are coming from multilabel_from_binaries. otherwise get the multilabel net
    print('starting up multilabel net')
#    protopath = os.path.join(os.path.dirname(os.path.abspath( __file__ )), 'classifier_stuff/caffe_nns/protos')
    modelpath = '/home/jeremy/caffenets/production'
    solverproto = os.path.join(modelpath,'ResNet-101-test.prototxt')
    deployproto = os.path.join(modelpath,'ResNet-101-deploy.prototxt')
    #caffemodel = os.path.join(modelpath,'multilabel_resnet101_sgd_iter_120000.caffemodel')
    caffemodel = os.path.join(modelpath,'multilabel_resnet101_sgd_iter_200000.caffemodel')  #10Aug2016
    print('solver proto {} deployproto {} caffemodel {}'.format(solverproto,deployproto,caffemodel))
    print('set_mode_gpu()')
    caffe.set_mode_gpu()
    print('device 0')
    caffe.set_device(gpu)
    multilabel_net = caffe.Net(deployproto,caffemodel, caffe.TEST)
    ###########OLD MULTILABELLER
    #caffemodel =  '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/snapshot/train_iter_340000.caffemodel'
    #deployproto = '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/deploy.prototxt'
    #multilabel_net = caffe.Net(deployproto,caffemodel, caffe.TEST)
    #multilabel_required_image_size = (227,227)

else:
    print('using multilabel thru falcon')

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

def get_layer_output(url_or_np_array,required_image_size=(224,224),layer='myfc7'):
    if isinstance(url_or_np_array, basestring):
        print('get_layer_output working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    if image is None:
        logging.debug('got None for image')
        return
    if required_image_size is not None:
        original_h,original_w = image.shape[0:2]
        logging.debug('resizing nd input to '+str(required_image_size)+' from '+str(original_h)+'x'+str(original_w))
      #  image,r = background_removal.standard_resize(image,max_side = 256)
        image = imutils.resize_keep_aspect(image,output_size=required_image_size,output_file=None)

    in_ = np.array(image, dtype=np.float32)   #.astype(float)
    if in_ is None:
        logging.debug('got none image in neurodoll.get_layer_output()')
        return None
    if len(in_.shape) != 3:
        if len(in_.shape) != 2:
            print('got something weird with shape '+str(in_.shape)+' , giving up')
            return None
        else:
            print('got  image with shape '+str(in_.shape)+' , turning into 3 channel')
            in_ = np.array([copy.deepcopy(in_),copy.deepcopy(in_),copy.deepcopy(in_)])
            print('now image has shape '+str(in_.shape))
    elif in_.shape[2] != 3:
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return None
    in_ -= np.array([104,116,122.0])  #was not used in training!!
    in_ = in_.transpose((2,0,1))   #wxhxc -> cxwxh
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()

    layer_data = net.blobs[layer].data
    return layer_data

def infer_one(url_or_np_array,required_image_size=(224,224),output_layer=OUTPUT_LAYER,mean=(104.0, 116.7, 122.7),save_results = True):
    start_time = time.time()
    image = Utils.get_cv2_img_array(url_or_np_array)
    thedir = './images'
    Utils.ensure_dir(thedir)
    if isinstance(url_or_np_array, basestring):
        print('infer_one working on url:'+url_or_np_array)
        orig_filename = os.path.join(thedir,url_or_np_array.split('/')[-1])
    elif type(url_or_np_array) == np.ndarray:
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        name_base = 'orig'+hash.hexdigest()[:10]+'.jpg'
        orig_filename = os.path.join(thedir,name_base)
    if image is None:
        logging.debug('got None in grabcut_using_neurodoll_output')

        # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#    im = Image.open(imagename)
#    im = im.resize(required_imagesize,Image.ANTIALIAS)
#    in_ = in_.astype(float)
    if image is None:
        logging.debug('got None for image')
        return
    if required_image_size is not None:
        original_h,original_w = image.shape[0:2]
        logging.debug('resizing nd input to '+str(required_image_size)+' from '+str(original_h)+'x'+str(original_w))
      #  image,r = background_removal.standard_resize(image,max_side = 256)
        image = imutils.resize_keep_aspect(image,output_size=required_image_size,output_file=None)
#        image = cv2.resize(image,dsize=(required_image_size[1],required_image_size[0]))

    in_ = np.array(image, dtype=np.float32)   #.astype(float)
    if in_ is None:
        logging.debug('got none image in neurodoll.infer_one()')
        return None
    if len(in_.shape) != 3:
        if len(in_.shape) != 2:
            print('got something weird with shape '+str(in_.shape)+' , giving up')
            return None
        else:
            print('got image with shape '+str(in_.shape)+' , turning into 3 channel')
            in_ = np.array([copy.deepcopy(in_),copy.deepcopy(in_),copy.deepcopy(in_)])
            print('now image has shape '+str(in_.shape))
    elif in_.shape[2] != 3:
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return None
#    in_ = in_[:,:,::-1]  for doing RGB -> BGR, not necessary when using cv2.imread
#    cv2.imwrite('test1234.jpg',in_) #verify that images are coming in as rgb

    in_ -= np.array(mean)  #make sure this fits whatever was used in training!!
    in_ = in_.transpose((2,0,1))   #wxhxc -> cxwxh
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs[output_layer].data[0].argmax(axis=0)
    out = np.array(out,dtype=np.uint16)
    if out is None:
        logging.debug('out image is None')

    #since original image may have been resized, reshape the output back to orig size
    if required_image_size is not None:
        logging.debug('resizing nd input back to '+str(original_h)+'x'+str(original_w))
    #    out = [out,out,out]
        #cv2 resize uses wxh
#        out = cv2.resize(out,(original_w,original_h))
        #my resize uses hxw (thats actually more consistent w. cv2 and numpy )
#        out = imutils.resize_keep_aspect(out,output_size=(original_h,original_w),output_file=None)
        out = imutils.undo_resize_keep_aspect(out,output_size=(original_h,original_w),careful_with_the_labels=True)
#        out = out[:,:,0]

    out = threshold_pixlevel(out)

    elapsed_time=time.time()-start_time
    print('infer_one elapsed time:'+str(elapsed_time))
 #   cv2.imshow('out',out.astype(np.uint8))
 #   cv2.waitKey(0)
#    return out.astype(np.uint8)
    out = np.array(out,dtype=np.uint8)

    if save_results:
        print('writing orig to '+orig_filename)
        cv2.imwrite(orig_filename,image)
        pngname = orig_filename[:-4]+'.png'
        cv2.imwrite(filename=pngname,img=out)
        imutils.show_mask_with_labels(pngname,labels=LABELS,visual_output=False,save_images=True,original_image=orig_filename)
    uniques = np.unique(out)
    logging.debug('final uniques:'+str(uniques))
    count_values(out,labels=constants.ultimate_21)
    return out,LABELS

def threshold_pixlevel(out,item_area_thresholds = constants.pixlevel_v3_min_area_thresholds,labels=constants.pixlevel_categories_v3):
#TODO - make the threshold per item ,e.g. small shoes are ok and should be left in
#done
    logging.debug('thresholding pixlevel areas using thresholds:'+str(item_area_thresholds))
    image_size = out.shape[0]*out.shape[1]
    uniques = np.unique(out)
    for unique in uniques:
        pixelcount = len(out[out==unique])
        ratio = float(pixelcount)/image_size
#        logging.debug('i {} pixels {} tot {} ratio {} threshold {} ratio<thresh {}'.format(unique,pixelcount,image_size,ratio,threshold,ratio<threshold))
        threshold = item_area_thresholds[unique]
        print('index {} {}  ratio {} threshold {}'.format(unique,labels[unique],  ratio,threshold))
        if ratio < threshold:
#            logging.debug('kicking out index '+str(unique)+' with ratio '+str(ratio))
            out[out==unique] = 0  #set label with small number of pixels to 0 (background)
            pixelcount = len(out[out==unique])
            logging.debug(str(unique) + ' is under area threshold, new ratio '+str(ratio))
    return(out)


def threshold_graylevels(graylevels,item_area_thresholds = constants.pixlevel_v3_min_area_thresholds,labels=constants.pixlevel_categories_v3):
    '''
    check which graylevels have  pixels >< threshold claimed, kill those graylevels
    '''
    out=np.argmax(graylevels,axis=2)
    logging.debug('thresholding gray levels using thresholds:'+str(item_area_thresholds))
    image_size = out.shape[0]*out.shape[1]
    uniques = np.unique(out)
    for unique in uniques:
        pixelcount = len(out[out==unique])
        ratio = float(pixelcount)/image_size
#        logging.debug('i {} pixels {} tot {} ratio {} threshold {} ratio<thresh {}'.format(unique,pixelcount,image_size,ratio,threshold,ratio<threshold))
        threshold = item_area_thresholds[unique]
        print('index {} {} n {} ratio {} threshold {}'.format(unique,labels[unique],pixelcount,  ratio,threshold))
        if ratio < threshold:
#            logging.debug('kicking out index '+str(unique)+' with ratio '+str(ratio))
            out[out==unique] = 0  #set label with small number of pixels to 0 (background)
            graylevels[:,:,unique]=0
            logging.debug(str(unique) + ' is under area threshold')
    return(graylevels)


def get_multilabel_output(url_or_np_array,required_image_size=(224,224),multilabel_source='hydra'):
#################################
#todo - parallelize the first if#
#################################

    if multilabel_from_binaries:
        print('getting ml from binaries')
        dic1 = bmfc.mlb(url_or_np_array)
        if not dic1['success']:
            logging.debug('nfc mlb not a success')
            return False
        output1 = dic1['output']
        dic2 = bmfc2.mlb(url_or_np_array)
        if not dic2['success']:
            logging.debug('nfc mlb2 not a success')
            return False
        output2 = dic2['output']
        dic3 = bmfc3.mlb(url_or_np_array)
        if not dic3['success']:
            logging.debug('nfc mlb3 not a success')
            return False
        output3 = dic3['output']
        output = output1+output2+output3
        return output

    elif multilabel_source == 'hydra':
        print('getting ml from hydra')
        dict = hydra_tg_falcon_client.hydra_tg(url_or_np_array)
        print('returning {}'.format(dict))
        return dict

    else:
        if isinstance(url_or_np_array, basestring):
            image = url_to_image(url_or_np_array)
        elif type(url_or_np_array) == np.ndarray:
            image = url_or_np_array
        if image is None:
            logging.debug('got None image')
            return None
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
        out = multilabel_net.blobs['prob'].data[0] #for the nth class layer #siggy is after sigmoid
        print('multilabel:  {}'.format(out))
        return out

def get_neurodoll_output_using_falcon(url_or_np_array):
    '''
    example of how you would get nd output using the falcon client
    :param url_or_np_array:
    :return:
    '''
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    dic = nfc.pd(image)
    if not dic['success']:
        logging.debug('nfc pd not a success')
        return False
    neuro_mask = dic['mask']
    return neuro_mask

def get_all_category_graylevels(url_or_np_array,resize=(256,256),required_image_size=(224,224),output_layer=OUTPUT_LAYER):
    start_time = time.time()
    print('get_all_category_graylevels using outputlayer '+output_layer)

    if isinstance(url_or_np_array, basestring):
        print('get_all_category_graylevels working on url:'+url_or_np_array+' req imsize:'+str(required_image_size)+' using outputlayer '+output_layer)
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    else:
        logging.debug('got something other than string and np array in get_all_categry_graylevels, returning')
        return
    if image is None:
        logging.debug('got None for image in get_all_categry_graylevels, returning')
        return

#todo - do a resize then crop to required_image_size, then undo the crop /resize (currently just resize/unresize) .
#this will avoid slight difference between train and deploy - train is on resize+crop, deploy is just on resize - so
# deploys are 13% smaller than train on average if train starts at 256x256 and is cropped to 224x224
    if required_image_size is not None:
        original_h, original_w = image.shape[0:2]
        logging.debug('get_all_cat_gl requesting resize from {} to {}'.format(image.shape,required_image_size))
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
    in_ = in_.transpose((2,0,1))  #change row,col,chan to chan,row,col as caffe wants
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
#    out = net.blobs['score'].data[0].argmax(axis=0) #for a parse with per-pixel max
 #   print('blobs:'+str(net.blobs))


#    out = net.blobs['score'].data[0] #for layer score, all outputs, no softmax#
    out = net.blobs[output_layer].data[0] #for layer score, all outputs after softmax
    min = np.min(out)
    max = np.max(out)
    print('get_all_category_graylevels output shape {} min {} max {}'.format(out.shape,min,max))
    out = out*255
    min = np.min(out)
    max = np.max(out)
    logging.debug('min {} max {} out after scaling'.format(min,max))
#    result = Image.fromarray(out.astype(np.uint8))
#        outname = im.strip('.png')[0]+'out.bmp'
#    outname = os.path.basename(imagename)
#    outname = outname.split('.jpg')[0]+'.bmp'
#    outname = os.path.join(out_dir,outname)
#    print('outname:'+outname)
#    result.save(outname)
    #        fullout = net.blobs['score'].data[0]
    elapsed_time=time.time()-start_time
 #   cv2.imshow('out',out.astype(np.uint8))
 #   cv2.waitKey(0)
    out = np.array(out,dtype=np.uint8)
    logging.debug('get_all_categorygraylevels:outshape '+str(out.shape))
#    out = out.transpose((2,0,1))  #change row,col,chan to chan,row,col as caffe wants
    out = out.transpose((1,2,0))  #change chan,row,col to row,col,chan  as the rest of world wants
    logging.debug('get_all_categorygraylevels:outshape '+str(out.shape))
    if required_image_size is not None:
        logging.debug('resizing nd input back to '+str(original_h)+'x'+str(original_w))
        out = imutils.undo_resize_keep_aspect(out,output_size=(original_h,original_w),careful_with_the_labels=True)
        print('get_all_categorygraylevels after reshape: '+str(out.shape))
    logging.debug('get_all_category_graylevels elapsed time:'+str(elapsed_time))
    return out

def analyze_graylevels(url_or_np_array,labels=constants.ultimate_21):
    if isinstance(url_or_np_array, basestring):
        print('analyze_graylevels working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
        name = url_or_np_array.replace('/','').replace('.jpg','').replace('.','').replace('http:','')
    elif type(url_or_np_array) == np.ndarray:
        print('starting to analyze graylevel on img')
        image = url_or_np_array
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        name = hash.hexdigest()[:10]

    gl = get_all_category_graylevels(url_or_np_array)
    if gl is None:
        logging.debug('got none from get_all_cateogry_graylevels, returning')
        return
    mask = gl.argmax(axis=2)
    background = np.array((mask==0)*1,dtype=np.uint8)
    foreground = np.array((mask>0)*1,dtype=np.uint8)
    cv2.imwrite(name+'bg.jpg',background*255)
    cv2.imwrite(name+'fg.jpg',foreground*255)
    tmin = np.min(foreground)
    tmax = np.max(foreground)
#    astype(np.uint8))

    print('masktype: '+str(type(background))+' shape:'+str(foreground.shape)+' min '+str(tmin)+' max '+str(tmax))
    h,w = gl.shape[0:2]
    window_size = 1700
    n_rows=5
    compress_factor = max(float(h*n_rows)/window_size,float(w*n_rows)/window_size)
    compressed_h = int(h/compress_factor)
    compressed_w = int(w/compress_factor)
    print('gl shape {} type {}'.format(gl.shape,type(gl)))
    compressed_gl = cv2.resize(gl,(compressed_w,compressed_h))
    print('fg shape {} type {}'.format(foreground.shape,type(foreground)))
    compressed_foreground = cv2.resize(foreground,(compressed_w,compressed_h))
    cv2.imwrite(name+'fg_comp.jpg',compressed_foreground*255)
    print('compressed hw {} {}'.format(compressed_h,compressed_w))
    compressed_image = cv2.resize(image,(compressed_w,compressed_h))
    big_out = np.zeros([compressed_h*n_rows,compressed_w*n_rows,3])
    print('bigsize:'+str(big_out.shape))


    for i in range(5):
        for j in range(5):
            n = i*n_rows+j
#            print('n {} i{} j {}:'.format(n,i,j))
            if n>=gl.shape[2]:
                print('finished blocks i {} j {} n {}'.format(i,j,n))
                big_out[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,0] = compressed_image[:,:,0]
                big_out[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,1] = compressed_image[:,:,1]
                big_out[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,2] = compressed_image[:,:,2]

                j = j+1
                big_out[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,0] = compressed_foreground*255
                big_out[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,1] = compressed_foreground*255
                big_out[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,2] = compressed_foreground*255


                break
#            print('y0 {} y1 {} x0 {} x1 {}'.format(i*h,(i+1)*h,j*w,(j+1)*w))
            big_out[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,0] = compressed_gl[:,:,n]
            big_out[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,1] = compressed_gl[:,:,n]
            big_out[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,2] = compressed_gl[:,:,n]
            cv2.putText(big_out,labels[n],(int((j+0.3)*compressed_w),int((i+1)*compressed_h-10)),cv2.FONT_HERSHEY_PLAIN,2,(250,200,255),thickness=3)
            cv2.imwrite(name+'bigout.jpg',big_out)
    time.sleep(0.1)
#            cv2.imshow('bigout',big_out)

    big_out2 = np.zeros([compressed_h*n_rows,compressed_w*n_rows,3])
    print('bigsize:'+str(big_out2.shape))

    for thresh in [0.5,0.7,0.9,0.95,0.98]:
        for i in range(5):
            for j in range(5):
                n = i*n_rows+j
               # print('n:'+str(n))
                if n>=gl.shape[2]:
                    print('finished blocks')
                    big_out2[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,:] = compressed_image
                    j = j+1
                    big_out2[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,0] = compressed_foreground*255
                    big_out2[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,1] = compressed_foreground*255
                    big_out2[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,2] = compressed_foreground*255

                    break
         #       print('y0 {} y1 {} x0 {} x1 {}'.format(i*h,(i+1)*h,j*w,(j+1)*w))
                big_out2[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,0] = (compressed_gl[:,:,n] > thresh*255)*255 * compressed_foreground
                big_out2[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,1] = (compressed_gl[:,:,n] > thresh*255)*255 * compressed_foreground
                big_out2[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,2] = (compressed_gl[:,:,n] > thresh*255)*255 * compressed_foreground

          #      print('tx {} ty {}'.format(int((j+0.5)*w),int((i+1)*h-10)))
                cv2.putText(big_out2,labels[n],(int((j+0.3)*compressed_w),int((i+1)*compressed_h-10)),cv2.FONT_HERSHEY_PLAIN,2,(250,200,255),thickness=3)
                cv2.imwrite(name+'bigout_thresh'+str(thresh)+'.jpg',big_out2)

def get_category_graylevel(url_or_np_array,category_index,required_image_size=(224,224)):
    all_layers = get_all_category_graylevels(url_or_np_array,required_image_size=required_image_size)
    requested_layer = all_layers[:,:,category_index]
    return requested_layer

def get_category_graylevel_masked_thresholded(url_or_np_array,category_index,required_image_size=(224,224),threshold=0.95):
    '''
    This takes a given layer, thresholds it, but keeps original backgound strictly
    :param url_or_np_array:
    :param category_index:
    :param required_image_size:
    :param threshold:
    :return:
    '''
    print('get_category_gl_masked_thresholded is working on {} trehshold {}'.format(url_or_np_array,threshold))
    if isinstance(url_or_np_array, basestring):
        name = url_or_np_array.replace('/','').replace('.jpg','').replace('.','').replace('http:','')
    else :
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        name = hash.hexdigest()[:10]

    all_layers = get_all_category_graylevels(url_or_np_array,required_image_size=required_image_size)
    if all_layers is None:
        logging.debug('got nothing back from get_all_category_graylevels in get_category_graylevel_masked_thresholded, returning')
        return
    requested_layer = all_layers[:,:,category_index]
    mask = all_layers.argmax(axis=2)
    basename = 'get_gl_thresh_'+str(name)+'_'+str(category_index)+'_'
    cv2.imwrite(basename+'mask.jpg',mask*255/21)
    background = mask==0
    cv2.imwrite(basename+'bgnd.jpg',background*255)
    foreground = np.array((mask>0)*1)  #*1 turns T/F into 1/0
    cv2.imwrite(basename+'fgnd.jpg',foreground*255)
    thresholded_layer = np.array((requested_layer>(threshold*255))*1)
    print('size of thresh layer '+str(thresholded_layer.shape))
    cv2.imwrite(basename+'thresh.jpg',thresholded_layer*255)
    new_mask = foreground * thresholded_layer #  * 1  multiplying by one turns True/False into 1/0 but seems to mess something?
    cv2.imwrite(basename+'out.jpg',new_mask*255)
    n_pixels = np.sum(new_mask)
    print('n nonzero pixels in get_cat_gl_masked_trhesholded:'+str(n_pixels))
    return new_mask

def grabcut_using_neurodoll_output(url_or_np_array,category_index,median_factor=1.6):
    '''
    takes an image (or url) and category.
    gets the neurodoll mask for that category.
    find the median value of the neurodoll mask.
    anything above becomes probable foreground (3) and anything less prob. background (2) (check this)
    then does a grabcut with these regions of fg, bg
    returned mask is 1 for fg and 0 for bg
    '''
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    if image is None:
        logging.debug('got None in grabcut_using_neurodoll_output')
        return

        #def neurodoll(image, category_idx):
    dic = nfc.pd(image, category_index=category_index)
    if not dic['success']:
        logging.debug('nfc pd not a success')
        return False, []
    neuro_mask = dic['mask']

    print('grabcut working on image of shape:'+str(image.shape)+' and mask of shape:'+str(neuro_mask.shape))

    nm_size = neuro_mask.shape[0:2]
    image_size = image.shape[0:2]
    if image_size != nm_size:
#        logging.debug('size mismatch')
        image = cv2.resize(image,(nm_size[1],nm_size[0]))
    # rect = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    mask = np.zeros(image.shape[:2], np.uint8)
    #TODO - maybe find something better than median as the threshold
    med = np.median(neuro_mask)*median_factor
    mask[neuro_mask > med] = 3
    mask[neuro_mask < med] = 2
    try:
        #TODO - try more than 1 grabcut call in itr
        itr = 1
        cv2.grabCut(image, mask, None, bgdmodel, fgdmodel, itr, cv2.GC_INIT_WITH_MASK)
    except:
        print('grabcut exception')
        return None
    mask2 = np.where((mask == 1) + (mask == 3), 1, 0).astype(np.uint8)
    return mask2

def grabcut_using_neurodoll_graylevel(url_or_np_array,neuro_mask,median_factor=1.6):
    '''
    takes an image (or url) and category.
    gets the neurodoll mask for that category.
    find the median value of the neurodoll mask.
    anything above becomes probable foreground (3) and anything less prob. background (2) (check this)
    then does a grabcut with these regions of fg, bg
    returned mask is 1 for fg and 0 for bg
    '''
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    if image is None:
        logging.debug('got None in grabcut_using_neurodoll_output')
        return
    print('grabcut working on image of shape:'+str(image.shape)+' and mask of shape:'+str(neuro_mask.shape))
        #def neurodoll(image, category_idx):
#    neuro_mask = dic['mask']

    nm_size = neuro_mask.shape[0:2]
    image_size = image.shape[0:2]
    if image_size != nm_size:
#        logging.debug('size mismatch')
        logging.warning('SHAPE MISMATCH IN GC USING ND GRAYLEVEL')
        image = cv2.resize(image,(nm_size[1],nm_size[0]))
    # rect = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    mask = np.zeros(image.shape[:2], np.uint8)
    #TODO - maybe find something better than median as the threshold
    med = np.median(neuro_mask)*median_factor
    mask[neuro_mask > med] = cv2.GC_PR_FGD  #(=3, prob foreground)
    mask[neuro_mask < med] = cv2.GC_PR_BGD #(=2, prob. background)
    print('gc pr fg {} pr bgnd {} '.format(cv2.GC_PR_FGD,cv2.GC_PR_BGD))

    try:
        #TODO - try more than 1 grabcut call in itr
        itr = 1
        cv2.grabCut(image, mask, None, bgdmodel, fgdmodel, itr, cv2.GC_INIT_WITH_MASK)
    except:
        print('grabcut exception')
        return None
    mask2 = np.where((mask == 1) + (mask == 3), 1, 0).astype(np.uint8)
    return mask2

def grabcut_using_neurodoll_graylevel(url_or_np_array,neuro_mask,median_factor=1.6):
    '''
    takes an image (or url) and category.
    gets the neurodoll mask for that category.
    find the median value of the neurodoll mask.
    anything above becomes probable foreground (3) and anything less prob. background (2) (check this)
    then does a grabcut with these regions of fg, bg
    returned mask is 1 for fg and 0 for bg
    '''
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    if image is None:
        logging.debug('got None in grabcut_using_neurodoll_output')
        return
    print('grabcut working on image of shape:'+str(image.shape)+' and mask of shape:'+str(neuro_mask.shape))
        #def neurodoll(image, category_idx):
#    neuro_mask = dic['mask']

    nm_size = neuro_mask.shape[0:2]
    image_size = image.shape[0:2]
    if image_size != nm_size:
#        logging.debug('size mismatch')
        logging.warning('SHAPE MISMATCH IN GC USING ND GRAYLEVEL')
        image = cv2.resize(image,(nm_size[1],nm_size[0]))
    # rect = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    mask = np.zeros(image.shape[:2], np.uint8)
    #TODO - maybe find something better than median as the threshold
    med = np.median(neuro_mask)*median_factor
    mask[neuro_mask > med] = cv2.GC_PR_FGD  #(=3, prob foreground)
    mask[neuro_mask < med] = cv2.GC_PR_BGD #(=2, prob. background)
    print('gc pr fg {} pr bgnd {} '.format(cv2.GC_PR_FGD,cv2.GC_PR_BGD))

    try:
        #TODO - try more than 1 grabcut call in itr
        itr = 1
        cv2.grabCut(image, mask, None, bgdmodel, fgdmodel, itr, cv2.GC_INIT_WITH_MASK)
    except:
        print('grabcut exception')
        return None
    mask2 = np.where((mask == 1) + (mask == 3), 1, 0).astype(np.uint8)
    return mask2

#this is confusing : this is how you would call falcon which calls get_multilabel_output (above...)
def get_multilabel_output_using_nfc(url_or_np_array):
    print('starting get_multilabel_output_using_nfc')
    multilabel_dict = nfc.pd(url, get_multilabel_results=True)
    print('get_multi_output:dict from falcon dict:'+str(multilabel_dict))
    if not multilabel_dict['success']:
        print('did not get nfc pd result succesfully')
        return
    multilabel_output = multilabel_dict['multilabel_output']
    print('multilabel output:'+str(multilabel_output))
    return multilabel_output #

def get_multilabel_output_using_post(url_or_arr):
    '''
    get hydra details on an image
    :param subimage: np array , e..g a crop of the original which fcrnn has found
    :return:
    '''
    HYDRA_CLASSIFIER_ADDRESS = constants.HYDRA_TG_CLASSIFIER_ADDRESS
    img_arr = Utils.get_cv2_img_array(url_or_arr)
    data = json.dumps({"image": img_arr })
    print('defense falcon is attempting to get response from hydra at ' + str(HYDRA_CLASSIFIER_ADDRESS))
    try:
        resp = requests.post(HYDRA_CLASSIFIER_ADDRESS, data=data)
        dict = json.loads(resp.content)
        return dict['output']
    except:
        print('couldnt get hydra output')
        return None

def zero_graylevels_not_in_ml(graylevels,ml_values,threshold=0.7,ml_to_nd_conversion=constants.binary_classifier_categories_to_ultimate_21):
    #has some bug and i dont wanna use it anyway
    for i in range(len(ml_values)):
        if ml_values[i] < threshold:
            nd_index = ml_to_nd_conversion[i]
            if nd_index is None:
                logging.debug('in zero_graylevels, no conversion from ml {} to nd'.format(i))
            else:
                graylevels[:,:,nd_index] = 0
    return graylevels

def count_values(mask,labels=None):
    image_size = mask.shape[0]*mask.shape[1]
    uniques = np.unique(mask)
    pixelcounts = {}
    for unique in uniques:
        pixelcount = len(mask[mask==unique])
        ratio = float(pixelcount)/image_size
        if labels is not None:
            print('class {} {} count {} ratio {}'.format(unique,labels[unique],pixelcount,ratio))
        else:
            print('class {} count {} ratio {}'.format(unique,pixelcount,ratio))
        pixelcounts[unique]=pixelcount
    return pixelcounts

def combine_neurodoll_and_multilabel(url_or_np_array,multilabel_threshold=0.7,median_factor=1.0,
                                     multilabel_to_ultimate21_conversion=constants.binary_classifier_categories_to_ultimate_21,
                                     multilabel_labels=constants.binary_classifier_categories, face=None,
                                     output_layer = OUTPUT_LAYER,required_image_size=(224,224),
                                     do_graylevel_zeroing=False):

    graylevel_nd_output = get_all_category_graylevels(url_or_np_array,output_layer=output_layer,required_image_size=required_image_size)
    multilabel_output = get_multilabel_output(url_or_np_array)

    retval = combine_neurodoll_and_multilabel_using_graylevel(url_or_np_array,graylevel_nd_output,multilabel_output,multilabel_threshold=multilabel_threshold,
                                     median_factor=median_factor,
                                     multilabel_to_pixlevel_conversion=multilabel_to_ultimate21_conversion,
                                     multilabel_labels=multilabel_labels, face=face,
                                     output_layer = output_layer,required_image_size=required_image_size,
                                     do_graylevel_zeroing=do_graylevel_zeroing)
    return retval


def combine_neurodoll_and_multilabel_using_graylevel(url_or_np_array,graylevel_nd_output,multilabel,multilabel_threshold=0.7,median_factor=1.0,
                                     multilabel_to_pixlevel_conversion=constants.binary_classifier_categories_to_ultimate_21,
                                     multilabel_labels=constants.binary_classifier_categories, face=None,
                                     output_layer = 'pixlevel_sigmoid_output',required_image_size=(224,224),
                                     do_graylevel_zeroing=False,thresholds=constants.pixlevel_v3_min_area_thresholds,labels=constants.pixlevel_categories_v3):
    '''
    try product of multilabel and nd output and taking argmax
    multilabel_to_ultimate21_conversion=constants.web_tool_categories_v1_to_ultimate_21 , or
    multilabel_to_ultimate21_conversion=constants.binary_classifier_categories_to_ultimate_21
    multilabel_labels=constants.web_tool_categories    , or
    multilabel_labels=constants.binary_classifier_categories

    multilabel - this should be in a form that the converter can deal with

    '''
    print('combining multilabel w. neurodoll, watch out, required imsize:'+str(required_image_size))
#
    thedir = './images'
    Utils.ensure_dir(thedir)
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
        orig_filename = os.path.join(thedir,url_or_np_array.split('/')[-1]).replace('.jpg','')
    elif type(url_or_np_array) == np.ndarray:
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        name_base = 'orig'+hash.hexdigest()[:10]
        orig_filename = os.path.join(thedir,name_base)
        image = url_or_np_array
    if image is None:
        logging.debug('got None in grabcut_using_neurodoll_output')
    print('writing orig to '+orig_filename+'.jpg')
    cv2.imwrite(orig_filename+'.jpg',image)


#    multilabel = get_multilabel_output(url_or_np_array)  this is now in calling function to allow use with other multilabel sources (eg hydra)
#    multilabel = get_multilabel_output_using_nfc(url_or_np_array)
    #take only labels above a threshold on the multilabel result
    #possible other way to do this: multiply the neurodoll mask by the multilabel result and threshold that product
    if multilabel is None:
        logging.debug('None result from multilabel')
        return None
    thresholded_multilabel = [ml>multilabel_threshold for ml in multilabel] #
    logging.info('orig label:'+str(multilabel)+' len:'+str(len(multilabel)))
#    print('incoming label:'+str(multilabel))
#    logging.info('thresholded label:'+str(thresholded_multilabel))
    for i in range(len(thresholded_multilabel)):
        if thresholded_multilabel[i]:
            logging.info(multilabel_labels[i]+' is over threshold')
#    print('multilabel to u21 conversion:'+str(multilabel_to_ultimate21_conversion))
#    print('multilabel labels:'+str(multilabel_labels))

    #todo - this may be wrong later if we start taking both nd and multilabel into acct. Maybe ml thinks theres nothing there but nd thinks there is...
    if np.equal(thresholded_multilabel,0).all():  #all labels 0 - nothing found
        logging.debug('no items found')
        return #


    pixlevel_categorical_output = graylevel_nd_output.argmax(axis=2) #the returned mask is HxWxC so take max along C
    pixlevel_categorical_output = threshold_pixlevel(pixlevel_categorical_output,item_area_thresholds=thresholds) #threshold out the small areas
    print('before graylevel zeroing:')
    count_values(pixlevel_categorical_output,labels=labels)

    if do_graylevel_zeroing: #if want to do this at least make multiple thresholds one for each cat. and instead of this donate pixels to more likely cats
        graylevel_nd_output = zero_graylevels_not_in_ml(graylevel_nd_output,multilabel,threshold=0.7)


    pixlevel_categorical_output = graylevel_nd_output.argmax(axis=2) #the returned mask is HxWxC so take max along C
    pixlevel_categorical_output = threshold_pixlevel(pixlevel_categorical_output) #threshold out the small areas
    print('after graylevel zeroing:')
    count_values(pixlevel_categorical_output,labels=labels)
    foreground = np.array((pixlevel_categorical_output>0) * 1)
    background = np.array((pixlevel_categorical_output==0) * 1)
    #    item_masks =  nfc.pd(image, get_all_graylevels=True)
    logging.debug('shape of pixlevel categorical output:'+str(pixlevel_categorical_output.shape))
    logging.debug('n_fg {} n_bg {} tot {} w*h {}'.format(np.sum(foreground),np.sum(background),np.sum(foreground)+np.sum(background),pixlevel_categorical_output.shape[0]*pixlevel_categorical_output.shape[1]))

    first_time_thru = True  #hack to dtermine image size coming back from neurodoll

 #   final_mask = np.zeros([224,224])
    final_mask = np.zeros(pixlevel_categorical_output.shape[:])
    print('final_mask shape '+str(final_mask.shape))

    if face:
        y_split = face[1] + 3 * face[3]
    else:
        # BETTER TO SEND A FACE
        y_split = np.round(0.4 * final_mask.shape[0])
    print('the y split {} face {}'.format(y_split,face))

    #the grabcut results dont seem too hot so i am moving to a 'nadav style' from-nd-and-ml-to-results system
    #namely : for top , decide if its a top or dress or jacket
    # for bottom, decide if dress/pants/skirt
    #decide on one bottom
 #   for i in range(len(thresholded_multilabel)):
 #       if multilabel_labels[i] in ['dress', 'jeans','shorts','pants','skirt','suit','overalls'] #missing from list is various swimwear which arent getting returned from nd now anyway

#############################################################################################
#Make some conclusions nadav style.
#Currently the decisions are based only on ml results without taking into acct the nd results.
#In future possibly inorporate nd as well, first do a head-to-head test of nd vs ml
#############################################################################################

    #1. take winning upper cover,  donate losers to winner
    #2. take winning upper under, donate losers to winner
    #3. take winning lower cover, donate losers to winner.
    #4. take winning lower under, donate losers to winner
    #5. decide on whole body item (dress, suit, overall) vs. non-whole body (two part e.g. skirt+top) items.
    #6. if wholebody beats two-part - donate all non-whole-body pixels to whole body (except upper-cover (jacket/blazer etc)  and lower under-stockings)
    #?  if no upper cover and no upper under and no whole-body: take max of all those and donate losers to winner

    #upper_cover: jacket, coat, blazer etc
    #upper under: shirt, top, blouse etc
    #lower cover: skirt, pants, shorts
    #lower under: tights, leggings

    whole_body_indexlist = [multilabel_labels.index(s) for s in  ['dress', 'suit','overalls']] #swimsuits could be added here
    upper_cover_indexlist = [multilabel_labels.index(s) for s in  ['cardigan', 'coat','jacket','sweater','sweatshirt']]
    upper_under_indexlist = [multilabel_labels.index(s) for s in  ['top']]
    lower_cover_indexlist = [multilabel_labels.index(s) for s in  ['jeans','pants','shorts','skirt']]
    lower_under_indexlist = [multilabel_labels.index(s) for s in  ['stocking']]

    final_mask = np.copy(pixlevel_categorical_output)
    logging.info('size of final mask '+str(final_mask.shape))

    print('wholebody indices:'+str(whole_body_indexlist))
    for i in whole_body_indexlist:
        print multilabel_labels[i]
    whole_body_ml_values = np.array([multilabel[i] for i in whole_body_indexlist])
    print('wholebody ml_values:'+str(whole_body_ml_values))
    whole_body_winner = whole_body_ml_values.argmax()
    whole_body_winner_value=whole_body_ml_values[whole_body_winner]
    whole_body_winner_index=whole_body_indexlist[whole_body_winner]
    print('winning index:'+str(whole_body_winner)+' mlindex:'+str(whole_body_winner_index)+' value:'+str(whole_body_winner_value))

    print('uppercover indices:'+str(upper_cover_indexlist))
    for i in upper_cover_indexlist:
        print multilabel_labels[i]
    upper_cover_ml_values = np.array([multilabel[i] for i in  upper_cover_indexlist])
    print('upper_cover ml_values:'+str(upper_cover_ml_values))
    upper_cover_winner = upper_cover_ml_values.argmax()
    upper_cover_winner_value=upper_cover_ml_values[upper_cover_winner]
    upper_cover_winner_index=upper_cover_indexlist[upper_cover_winner]
    print('winning upper_cover:'+str(upper_cover_winner)+' mlindex:'+str(upper_cover_winner_index)+' value:'+str(upper_cover_winner_value))

    print('upperunder indices:'+str(upper_under_indexlist))
    for i in upper_under_indexlist:
        print multilabel_labels[i]
    upper_under_ml_values = np.array([multilabel[i] for i in  upper_under_indexlist])
    print('upper_under ml_values:'+str(upper_under_ml_values))
    upper_under_winner = upper_under_ml_values.argmax()
    upper_under_winner_value=upper_under_ml_values[upper_under_winner]
    upper_under_winner_index=upper_under_indexlist[upper_under_winner]
    print('winning upper_under:'+str(upper_under_winner)+' mlindex:'+str(upper_under_winner_index)+' value:'+str(upper_under_winner_value))

    print('lowercover indices:'+str(lower_cover_indexlist))
    for i in lower_cover_indexlist:
        print multilabel_labels[i]
    lower_cover_ml_values = np.array([multilabel[i] for i in lower_cover_indexlist])
    print('lower_cover ml_values:'+str(lower_cover_ml_values))
    lower_cover_winner = lower_cover_ml_values.argmax()
    lower_cover_winner_value=lower_cover_ml_values[lower_cover_winner]
    lower_cover_winner_index=lower_cover_indexlist[lower_cover_winner]
    print('winning lower_cover:'+str(lower_cover_winner)+' mlindex:'+str(lower_cover_winner_index)+' value:'+str(lower_cover_winner_value))

    print('lowerunder indices:'+str(lower_under_indexlist))
    for i in lower_under_indexlist:
        print multilabel_labels[i]
    lower_under_ml_values = np.array([multilabel[i] for i in  lower_under_indexlist])
    print('lower_under ml_values:'+str(lower_under_ml_values))
    lower_under_winner = lower_under_ml_values.argmax()
    lower_under_winner_value=lower_under_ml_values[lower_under_winner]
    lower_under_winner_index=lower_under_indexlist[lower_under_winner]
    print('winning lower_under:'+str(lower_under_winner)+' mlindex:'+str(lower_under_winner_index)+' value:'+str(lower_under_winner_value))

    #for use later, decide on a winner between upper cover and upper under
    if upper_under_winner_value > upper_cover_winner_value:
        upper_winner_value = upper_under_winner_value
        upper_winner_index = upper_under_winner_index
    else:
        upper_winner_value = upper_cover_winner_value
        upper_winner_index = upper_cover_winner_index
    #for use later, decide on a winner between lower cover and lower under
    if lower_under_winner_value > lower_cover_winner_value:
        lower_winner_value = lower_under_winner_value
        lower_winner_index = lower_under_winner_index
    else:
        lower_winner_value = lower_cover_winner_value
        lower_winner_index = lower_cover_winner_index
    upper_winner_nd_index = multilabel_to_pixlevel_conversion[upper_winner_index]
    lower_winner_nd_index = multilabel_to_pixlevel_conversion[lower_winner_index]
    print('upper winner {} nd {} val {} lower winner {} nd {} val {}'.format(upper_winner_index,upper_winner_nd_index,upper_winner_value,
                                                                             lower_winner_index,lower_winner_nd_index,lower_winner_value))
#1. take max upper cover , donate losers to winner
#this actually might not be always right, e.g. jacket+ sweater
#todo  - #1 - 4 can be put into a function since they are nearly identical
    neurodoll_upper_cover_index = multilabel_to_pixlevel_conversion[upper_cover_winner_index]
    if neurodoll_upper_cover_index is None:
        logging.warning('nd upper cover index {}  has no conversion '.format(upper_cover_winner_index))
    else:
        n = len(final_mask[final_mask==neurodoll_upper_cover_index])
        logging.debug('donating to upper cover winner, initial n :'+str(n)+' for ndindex '+str(neurodoll_upper_cover_index)+' ml index '+str(upper_cover_winner_index)+ ', checking mls '+str(upper_cover_indexlist))
        for i in upper_cover_indexlist: #whole_body donated to upper_under
            nd_index = multilabel_to_pixlevel_conversion[i]
            if nd_index is None:
                logging.warning('ml index {} has no conversion '.format(i))
                continue
            x = final_mask[final_mask==nd_index]
            final_mask[final_mask==nd_index] = neurodoll_upper_cover_index
            n = len(final_mask[final_mask==neurodoll_upper_cover_index])
            logging.info('upper cover ndindex {} {} donated to upper cover winner nd {} , now {} pixels, lenx {} '.format(nd_index,constants.ultimate_21[nd_index],neurodoll_upper_cover_index, n,len(x)))

#2. take max upper under, donate losers to winner
    neurodoll_upper_under_index = multilabel_to_pixlevel_conversion[upper_under_winner_index]
    if neurodoll_upper_under_index is None:
        logging.warning('nd upper cover index {}  has no conversion '.format(upper_under_winner_index))
    else:
        n = len(final_mask[final_mask==neurodoll_upper_under_index])
        logging.debug('donating to upper under winner, initial n :'+str(n)+' for ndindex '+str(neurodoll_upper_under_index)+' ml index '+str(upper_under_winner_index)+ ', checking mls '+str(upper_under_indexlist))
        for i in upper_under_indexlist: #upper under losers donated to upper under winner
            nd_index = multilabel_to_pixlevel_conversion[i]
            print('nd index {} ml index {}'.format(nd_index,i))
            if nd_index is None:
                logging.warning('ml index {} has no conversion '.format(i))
                continue
            final_mask[final_mask==nd_index] = neurodoll_upper_under_index
            n = len(final_mask[final_mask==neurodoll_upper_under_index])
            logging.info('upper under ndindex {} {} donated to upper under winner nd {}, now {} pixels'.format(nd_index,constants.ultimate_21[nd_index],neurodoll_upper_under_index,n))

#3. take max lower cover, donate losers to winner.
    neurodoll_lower_cover_index = multilabel_to_pixlevel_conversion[lower_cover_winner_index]
    if neurodoll_lower_cover_index is None:
        logging.warning('nd lower cover index {}  has no conversion '.format(lower_cover_winner_index))
    else:
        n = len(final_mask[final_mask==neurodoll_lower_cover_index])
        logging.debug('donating to lower cover winner, initial n :'+str(n)+' for ndindex '+str(neurodoll_lower_cover_index)+' ml index '+str(lower_cover_winner_index)+ ', checking mls '+str(lower_cover_indexlist))
        for i in lower_cover_indexlist: #lower cover losers donated to lower cover winner
            nd_index = multilabel_to_pixlevel_conversion[i]
            if nd_index is None:
                logging.warning('ml index {} has no conversion '.format(i))
                continue
            final_mask[final_mask==nd_index] = neurodoll_lower_cover_index
            n = len(final_mask[final_mask==neurodoll_lower_cover_index])
            logging.info('lower cover ndindex {} {} donated to lower cover winner nd {}, now {} pixels'.format(nd_index,constants.ultimate_21[nd_index],neurodoll_lower_cover_index,n))

#4. take max lower under, donate losers to winner.
    neurodoll_lower_under_index = multilabel_to_pixlevel_conversion[lower_under_winner_index]
    if neurodoll_lower_under_index is None:
        logging.warning('nd lower under index {}  has no conversion '.format(lower_under_winner_index))
    else:
        n = len(final_mask[final_mask==neurodoll_lower_under_index])
        logging.debug('donating to lower under winner, initial n :'+str(n)+' for ndindex '+str(neurodoll_lower_under_index)+' ml index '+str(lower_under_winner_index)+ ', checking mls '+str(lower_under_indexlist))
        for i in lower_under_indexlist: #lower under losers donated to lower under winner
            nd_index = multilabel_to_pixlevel_conversion[i]
            if nd_index is None:
                logging.warning('ml index {} has no conversion '.format(i))
                continue
            final_mask[final_mask==nd_index] = neurodoll_lower_under_index
            n = len(final_mask[final_mask==neurodoll_lower_under_index])
            logging.info('lower under ndindex {} {} donated to lower under winner nd {}, now {} pixels'.format(nd_index,constants.ultimate_21[nd_index],neurodoll_lower_under_index,n))

    logging.debug('after step 4, pixelcounts look like:')
    count_values(final_mask,labels=constants.ultimate_21)

#########################
# 5. WHOLEBODY VS TWO-PART
# decide on whole body item (dress, suit, overall) vs. non-whole body items.
# case 1 wholebody>upper_under>lower_cover
# case 2 upper_under>wholebody>lower_cover
# case 3 lower_cover>wholebody>upper-under
# case 4 lower_cover,upper_under > wholebody
#consider reducing this to nadav's method:
#    whole_sum = np.sum([item.values()[0] for item in mask_sizes['whole_body']])
#    partly_sum = np.sum([item.values()[0] for item in mask_sizes['upper_under']]) +\
#                 np.sum([item.values()[0] for item in mask_sizes['lower_cover']])
#                if whole_sum > partly_sum:
#    donate partly to whole
# its a little different tho since in multilabel you cant compare directly two items to one , e.g. if part1 = 0.6, part2 = 0.6, and whole=0.99, you
# should prob go with whole even tho part1+part2>whole
#########################
    neurodoll_wholebody_index = multilabel_to_pixlevel_conversion[whole_body_winner_index]
    if neurodoll_wholebody_index is None:
        logging.warning('nd wholebody index {} ml index {} has no conversion '.format(neurodoll_wholebody_index,whole_body_winner_index))

#first case - wholebody > upper_under > lowercover
#donate all non-whole-body pixels to whole body (except upper-cover (jacket/blazer etc)  and lower under-stockings)
    elif (whole_body_winner_value>upper_under_winner_value) and (whole_body_winner_value>lower_cover_winner_value) and whole_body_winner_value>multilabel_threshold:
        logging.info('case 1. one part {} wins over upper cover {} and lower cover {}'.format(whole_body_winner_value,upper_cover_winner_value,lower_cover_winner_value))
        n = len(final_mask[final_mask==neurodoll_wholebody_index])
        logging.info('n in final mask from wholebody alone:'+str(n))
        for i in upper_cover_indexlist:
            #jackets etc can occur with dress/overall so dont donate these
            pass
        for i in upper_under_indexlist:  #donate upper_under to whole_body
    #todo fix the case of suit (which can have upper_under)
            #ideally, do this for dress - suit and overalls can have upper_under
            nd_index = multilabel_to_pixlevel_conversion[i]
            if nd_index is None:
                logging.debug('upper cover nd index for {} has no conversion '.format(i))
                continue
            #add upper cover item to wholebody mask
            final_mask[final_mask==nd_index] = neurodoll_wholebody_index
            logging.info('adding upperunder nd index {} '.format(nd_index))
            n = final_mask[final_mask==neurodoll_wholebody_index]
            logging.info('n in final mask from wholebody:'+str(n))
        for i in lower_cover_indexlist: #donate lower_cover to whole_body
            nd_index = multilabel_to_pixlevel_conversion[i]
            final_mask[final_mask==nd_index] = neurodoll_wholebody_index
            logging.info('adding lowercover nd index {} '.format(nd_index))
            n = final_mask[final_mask==neurodoll_wholebody_index]
            logging.info('n in final mask from wholebody alone:'+str(n))
        for i in lower_under_indexlist:
            #not doing this for stockings which is currently the only lower under
            pass
        logging.debug('after case one pixel values look like')
        count_values(final_mask,labels=constants.ultimate_21)


# second case - upper_under > wholebody > lowercover
# here its not clear who to sack - the wholebody or the upper_under
# so i arbitrarily decided to sack th whole_body in favor of the upper_under since upper_under is higher
# EXCEPT if the wholebody is overalls , in which case keep overalls, upper_under and upper_cover, donate  lower_cover/under to overalls
# otherwise  if wholebody is e.g. dress then add dress to upper_under and lower_cover

    elif (whole_body_winner_value<upper_under_winner_value) and (whole_body_winner_value>lower_cover_winner_value) and (whole_body_winner_value>multilabel_threshold):
        logging.info('case 2. one part {} < upper under {} but > lower cover {}'.format(whole_body_winner_value,upper_under_winner_value,lower_cover_winner_value))
#if overalls, donate loewr_cover and lower_under to overalls
        if whole_body_winner_index == multilabel_labels.index('overalls'):
            neurodoll_whole_body_index = multilabel_to_pixlevel_conversion[whole_body_winner_index]
            n = len(final_mask[final_mask==neurodoll_wholebody_index])
            logging.info('n in final mask from wholebody (overall) alone:'+str(n))
            for i in upper_cover_indexlist:
                pass  #upper cover ok with overalls
            for i in upper_under_indexlist:
                pass #upper under ok with overalls
            for i in lower_cover_indexlist: #lower cover donated to overalls
                nd_index = multilabel_to_pixlevel_conversion[i]
                if nd_index is None:
                    logging.warning('ml index {} has no conversion '.format(i))
                    continue
                final_mask[final_mask==nd_index] = neurodoll_wholebody_index
                logging.info('uppercover nd index {} donated to overalls'.format(nd_index))
                n = len(final_mask[final_mask==neurodoll_wholebody_index])
                logging.info('n in final mask from wholebody alone:'+str(n))
            for i in lower_under_indexlist: #lower under donated to overalls - this can conceivably go wrong e.g. with short overalls and stockings
                nd_index = multilabel_to_pixlevel_conversion[i]
                if nd_index is None:
                    logging.warning('ml index {} has no conversion '.format(i))
                    continue
                final_mask[final_mask==nd_index] = neurodoll_wholebody_index
                logging.info('uppercover nd index {} donated to overalls'.format(nd_index))
                n = len(final_mask[final_mask==neurodoll_wholebody_index])
                logging.info('n in final mask from wholebody alone:'+str(n))
#not overalls, so donate  whole_body to upper_under - maybe not to  lower_under . Not clear what to do actually.
        else: #not overalls
            if upper_winner_nd_index is None:
                logging.warning('nd wholebody index {} ml index {} has no conversion '.format(upper_winner_nd_index,upper_winner_index))
            else:
                n1 = len(final_mask[final_mask==upper_winner_nd_index])
                n2 = len(final_mask[final_mask==lower_winner_nd_index])
                logging.info('n in final mask from wholebody before donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
                #todo - actually only wholebody pixels in the upper half of the image should be donated
                for i in whole_body_indexlist: #whole_body donated to upper_under
                    nd_index = multilabel_to_pixlevel_conversion[i]
                    if nd_index is None:
                        logging.warning('ml index {} has no conversion (4upper)'.format(i))
                        continue            #donate upper pixels to upper_winner
                    logging.debug('3. donating nd {} in top of wholebody to upper_under and bottom to lower_under'.format(nd_index))
                    logging.debug('first adding from upper part of nd {}  to nd {}, ysplit {}'.format(nd_index,upper_winner_nd_index,y_split))
                    for y in range(0, final_mask.shape[0]):
                        if y <= y_split:
                            for x in range(0, final_mask.shape[1]):
                                if final_mask[y][x] == nd_index:
                                    final_mask[y][x] = upper_winner_nd_index
                n1 = len(final_mask[final_mask==upper_winner_nd_index])
                n2 = len(final_mask[final_mask==lower_winner_nd_index])
                logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))

    # donate whole-body pixels to lower winner
            if lower_winner_nd_index is None:
                logging.warning('nd wholebody index {} ml index {} has no conversion '.format(upper_winner_nd_index,upper_winner_index))
            else:
                n1 = len(final_mask[final_mask==upper_winner_nd_index])
                n2 = len(final_mask[final_mask==lower_winner_nd_index])
                logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} , lower winner {} px, nd {} '.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
                #todo - actually only wholebody pixels in the upper half of the image should be donated
                for i in whole_body_indexlist: #whole_body donated to upper_under
                    nd_index = multilabel_to_pixlevel_conversion[i]
                    if nd_index is None:
                        logging.warning('ml index {} has no conversion (4lower)'.format(i))
                        continue
            #donate upper pixels to upper_winner
                    logging.debug('3. donating nd {} in botto of wholebody to upper_under and bottom to lower_under'.format(nd_index))
                    logging.debug('second, adding from lower part of nd {}  to nd {}, ysplit {}'.format(nd_index,lower_winner_nd_index,y_split))
                    for y in range(0, final_mask.shape[0]):
                        if y > y_split:
                            for x in range(0, final_mask.shape[1]):
                                if final_mask[y][x] == nd_index:
                                    final_mask[y][x] = lower_winner_nd_index
            #donate upper pixels to lower_winner
                n1 = len(final_mask[final_mask==upper_winner_nd_index])
                n2 = len(final_mask[final_mask==lower_winner_nd_index])
                logging.info('n in final mask from wholebody donation to upper {} and lower {}:'.format(n1,n2))



        logging.debug('after case two pixel values look like')
        count_values(final_mask,labels=constants.ultimate_21)

# third case - lowercover > wholebody > upper_under
# here its not clear who to sack - the lowercover or the wholebody
# so i arbitrarily decided to sack the whole_body in favor of the lowercover since lowercover is higher
# donate lower part of wholebody to lowerwinner and upper part to upper winner
# this can be combined with second case I guess as there is nothing different - whole body gets added to lower/upper winners

    elif (whole_body_winner_value<lower_cover_winner_value) and (whole_body_winner_value>upper_under_winner_value) and whole_body_winner_value>multilabel_threshold:
        logging.info('case 3. one part {} > upper under {} and < lower cover {}'.format(whole_body_winner_value,upper_under_winner_value,lower_cover_winner_value))
        if upper_winner_nd_index is None:
            logging.warning('nd wholebody index {} ml index {} has no conversion '.format(upper_winner_nd_index,upper_winner_index))
        else:
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody before donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
            #todo - actually only wholebody pixels in the upper half of the image should be donated
            for i in whole_body_indexlist: #whole_body donated to upper_under
                nd_index = multilabel_to_pixlevel_conversion[i]
                if nd_index is None:
                    logging.warning('ml index {} has no conversion (4upper)'.format(i))
                    continue            #donate upper pixels to upper_winner
                logging.debug('3. donating nd {} in top of wholebody to upper_under and bottom to lower_under'.format(nd_index))
                logging.debug('first adding from upper part of nd {}  to nd {}, ysplit {}'.format(nd_index,upper_winner_nd_index,y_split))
                for y in range(0, final_mask.shape[0]):
                    if y <= y_split:
                        for x in range(0, final_mask.shape[1]):
                            if final_mask[y][x] == nd_index:
                                final_mask[y][x] = upper_winner_nd_index
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))

# donate whole-body pixels to lower winner
        if lower_winner_nd_index is None:
            logging.warning('nd wholebody index {} ml index {} has no conversion '.format(upper_winner_nd_index,upper_winner_index))
        else:
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} , lower winner {} px, nd {} '.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
            #todo - actually only wholebody pixels in the upper half of the image should be donated
            for i in whole_body_indexlist: #whole_body donated to upper_under
                nd_index = multilabel_to_pixlevel_conversion[i]
                if nd_index is None:
                    logging.warning('ml index {} has no conversion (4lower)'.format(i))
                    continue
        #donate upper pixels to upper_winner
                logging.debug('3. donating nd {} in botto of wholebody to upper_under and bottom to lower_under'.format(nd_index))
                logging.debug('second, adding from lower part of nd {}  to nd {}, ysplit {}'.format(nd_index,lower_winner_nd_index,y_split))
                for y in range(0, final_mask.shape[0]):
                    if y > y_split:
                        for x in range(0, final_mask.shape[1]):
                            if final_mask[y][x] == nd_index:
                                final_mask[y][x] = lower_winner_nd_index
        #donate upper pixels to lower_winner
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody donation to upper {} and lower {}:'.format(n1,n2))

# fourth case - lowercover , upper_under > wholebody
# sack wholebody in favor of upper and lower
# donate top of wholebody to greater of upper cover/upper under (yes this is arbitrary and possibly wrong)
# donate bottom pixels of wholebody to greater of lower cover/lower under (again somewhat arbitrary)
# this also could get combined with #2,3 I suppose
# neurodoll_upper_cover_index = multilabel_to_pixlevel_conversion[upper_cover_winner_index] #
        logging.debug('after case three pixel values look like')
        count_values(final_mask,labels=constants.ultimate_21)


    elif (whole_body_winner_value<lower_cover_winner_value) and (whole_body_winner_value<upper_under_winner_value):
        logging.info('case 4.one part {} < upper under {} and < lower cover {}'.format(whole_body_winner_value,upper_under_winner_value,lower_cover_winner_value))
        neurodoll_lower_cover_index = multilabel_to_pixlevel_conversion[lower_cover_winner_index]
# donate whole-body pixels to upper winner
        if upper_winner_nd_index is None:
            logging.warning('nd wholebody index {} ml index {} has no conversion '.format(upper_winner_nd_index,upper_winner_index))
        else:
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody before donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
            #todo - actually only wholebody pixels in the upper half of the image should be donated
            for i in whole_body_indexlist: #whole_body donated to upper_under
                nd_index = multilabel_to_pixlevel_conversion[i]
                if nd_index is None:
                    logging.warning('ml index {} has no conversion (4upper)'.format(i))
                    continue            #donate upper pixels to upper_winner
                logging.debug('4. donating nd {} in top of wholebody to upper_under and bottom to lower_under'.format(nd_index))
                logging.debug('first adding from upper part of nd {}  to nd {}, ysplit {}'.format(nd_index,upper_winner_nd_index,y_split))
                for y in range(0, final_mask.shape[0]):
                    if y <= y_split:
                        for x in range(0, final_mask.shape[1]):
                            if final_mask[y][x] == nd_index:
                                final_mask[y][x] = upper_winner_nd_index
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))

# donate whole-body pixels to lower winner
        if lower_winner_nd_index is None:
            logging.warning('nd wholebody index {} ml index {} has no conversion '.format(upper_winner_nd_index,upper_winner_index))
        else:
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} , lower winner {} px, nd {} '.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
            #todo - actually only wholebody pixels in the upper half of the image should be donated
            for i in whole_body_indexlist: #whole_body donated to upper_under
                nd_index = multilabel_to_pixlevel_conversion[i]
                if nd_index is None:
                    logging.warning('ml index {} has no conversion (4lower)'.format(i))
                    continue
        #donate upper pixels to upper_winner
                logging.debug('4. donating nd {} in botto of wholebody to upper_under and bottom to lower_under'.format(nd_index))
                logging.debug('second, adding from lower part of nd {}  to nd {}, ysplit {}'.format(nd_index,lower_winner_nd_index,y_split))
                for y in range(0, final_mask.shape[0]):
                    if y > y_split:
                        for x in range(0, final_mask.shape[1]):
                            if final_mask[y][x] == nd_index:
                                final_mask[y][x] = lower_winner_nd_index
        #donate upper pixels to lower_winner
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody donation to upper {} and lower {}:'.format(n1,n2))

        logging.debug('after case four pixel values look like')
        count_values(final_mask,labels=constants.ultimate_21)

    foreground = np.array((pixlevel_categorical_output>0)*1)  #*1 turns T/F into 1/0
    final_mask = final_mask * foreground # only keep stuff that was part of original fg - this is already  true
    # unless we start adding pixvalues that didn't 'win'

    #7. if no lower cover and no whole-body was decided upon above: take max of lowercover items , donate losers to winner
    #8. take at most one lower under, donate losers to winner

    if(0):
        for i in range(len(thresholded_multilabel)):
            if thresholded_multilabel[i]:
                neurodoll_index = multilabel_to_pixlevel_conversion[i]
                if neurodoll_index is None:
                    print('no mapping from index {} (label {}) to neurodoll'.format(i,multilabel_labels[i]))
                    continue
                nd_pixels = len(pixlevel_categorical_output[pixlevel_categorical_output==neurodoll_index])
                print('index {} webtoollabel {} newindex {} neurodoll_label {} was above threshold {} (ml value {}) nd_pixels {}'.format(
                    i,multilabel_labels[i],neurodoll_index,constants.ultimate_21[neurodoll_index], multilabel_threshold,multilabel[i],nd_pixels))
                gray_layer = graylevel_nd_output[:,:,neurodoll_index]
                print('gray layer size:'+str(gray_layer.shape))
    #            item_mask = grabcut_using_neurodoll_output(url_or_np_array,neurodoll_index,median_factor=median_factor)
                if nd_pixels>0:  #possibly put a threshold here, too few pixels and forget about it
                    item_mask = grabcut_using_neurodoll_graylevel(url_or_np_array,gray_layer,median_factor=median_factor)
                    #the grabcut results dont seem too hot so i am moving to a 'nadav style' from-nd-and-ml-to-results system
                #namely : for top , decide if its a top or dress or jacket
                # for bottom, decide if dress/pants/skirt
                    pass
                else:
                    print('no pixels in mask, skipping')
                if item_mask is None:
                    continue
                item_mask = np.multiply(item_mask,neurodoll_index)
                if first_time_thru:
                    final_mask = np.zeros_like(item_mask)
                    first_time_thru = False
                unique_to_new_mask = np.logical_and(item_mask != 0,final_mask == 0)   #dealing with same pixel claimed by two masks. if two masks include same pixel take first, don't add the pixel vals together
                unique_to_new_mask = np.multiply(unique_to_new_mask,neurodoll_index)
                final_mask = final_mask + unique_to_new_mask
    #            cv2.imshow('mask '+str(i),item_mask)
    #            cv2.waitKey(0)
    timestamp = int(10*time.time())

    #write file (for debugging)
    name = orig_filename+'_combinedoutput.png'

    print('combined png name:'+name+' orig filename '+orig_filename)
    cv2.imwrite(name,final_mask)
    nice_output = imutils.show_mask_with_labels(name,constants.ultimate_21,save_images=True,original_image=orig_filename+'.jpg',visual_output=False)
#    nice_output = imutils.show_mask_with_labels(name,constants.ultimate_21,save_images=True)

    #save graymask, this should be identical to nd except no threshold on low amt of pixels
    graymask_filename = orig_filename+'_origmask.png'
    print('original mask file:'+graymask_filename)
    cv2.imwrite(graymask_filename,pixlevel_categorical_output)
    nice_output = imutils.show_mask_with_labels(graymask_filename,constants.ultimate_21,save_images=True,original_image=orig_filename+'.jpg',visual_output=False)
    count_values(final_mask,labels=constants.ultimate_21)

    return final_mask






def combine_neurodoll_v3labels_and_multilabel(url_or_np_array):
    #next two lines can be paralleled
    print('starting combine_neurodoll_v3labels_and_multilabel')

    # print('testing gevent')
    # urls = ['www.google.com', 'www.example.com', 'www.python.org']
    # jobs = [gevent.spawn(socket.gethostbyname, url) for url in urls]
    # gevent.joinall(jobs, timeout=2)
    # gevent_results = [job.value for job in jobs]
    # print('gevent results '+str(gevent_results))

    print('getting graylevel output')
    graylevel_nd_output = get_all_category_graylevels(url_or_np_array)
    print('GOT GRAYLEVEL output')
    if graylevel_nd_output is None:
        if isinstance(url_or_np_array,basestring):
            logging.error('didnt get graylevel nd output or '+str(url_or_np_array))
        else:
            logging.error('didnt get graylevel nd output')
        return None

    print('getting multilabel output')
    multilabel_output = get_multilabel_output(url_or_np_array)
    print('got multilabel output')
    multilabel_as_u21 = multilabel_output  #maybe some conversion needed here
    thedir='/data/production/caffemodels_and_protos/neurodoll/images'
    Utils.ensure_dir(thedir)
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
        filename = os.path.join(thedir,url_or_np_array.split('/')[-1]).replace('.jpg','')
    elif type(url_or_np_array) == np.ndarray:
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        name_base = 'orig'+hash.hexdigest()[:10]
        filename = os.path.join(thedir,name_base)
        image = url_or_np_array
 #   filename = os.path.join(thedir,url_or_np_array.split('/')[-1]).replace('.jpg','')
    print('using name {} for saving results'.format(filename))
    cv2.imwrite(filename+'.jpg',image)
    final_mask = combine_neurodoll_v3labels_and_multilabel_using_graylevel(graylevel_nd_output,multilabel_as_u21,face=None,
                                                              required_image_size=(224,224),orig_filename=filename)

###########saving results for www
    save_for_www = True
    if save_for_www:
        try:
            wwwpath = '/data/www'
            wwwname = os.path.join(wwwpath,os.path.basename(filename))
            pngname = wwwname+'.png'
            jpgname = wwwname+'.jpg'
            multilabelname = wwwname+'.txt'
            print('WWW - saving png to '+pngname)
            cv2.imwrite(pngname,final_mask)
            print('WWW - saving jpg to '+jpgname)
            cv2.imwrite(jpgname,image)
            nice_output = imutils.show_mask_with_labels(final_mask,constants.ultimate_21,save_images=True,original_image=jpgname,visual_output=False)
            legendname = jpgname.replace('.jpg','_legend.jpg')
            htmlname = os.path.join('/data/www','results.html')

            print('WWW - saving json to '+multilabelname)
            print('WWW - ml output '+str(multilabel_output))
            Utils.ensure_file(multilabelname)
            with open(multilabelname,'w') as fp:
                print('WWW - dumping')
                json.dump(multilabel_output,fp,indent=4)
                fp.close()
            print('WWW -reading '+htmlname)
            Utils.ensure_file(htmlname)
            with open(htmlname,'r') as fp2:
                lines = fp2.readlines()
                fp2.close()

            print('WWW - writing')

            newlines = [os.path.basename(legendname)+'\n',os.path.basename(pngname)+'\n',
                        os.path.basename(jpgname)+'\n',os.path.basename(multilabelname)+'\n']+lines
            with open(htmlname,'w') as fp3:
                fp3.writelines(newlines)
                fp3.close()

            index_htmls.write_index_html_with_images(wwwpath,newlines,indexname='/data/www/index.html')

        except:
            print(sys.exc_info()[0])

    return final_mask
#    return {'mask':final_mask,'multilabel':multilabel_output}







def combine_neurodoll_v3labels_and_multilabel_using_graylevel(graylevel_nd_output,hydra_multilabel,multilabel_threshold=0.5,
                                     median_factor=1.0,multilabel_labels=constants.ultimate_21,
                                     face=None,required_image_size=(224,224),do_graylevel_zeroing=False,orig_filename=None,
                                     labels=constants.pixlevel_categories_v3):
    '''
    1. decide on wholebody vs 2part using multilabel(hydra)  results
    2. donate losing pixels to winning
    3. return paperdoll-style result using hydra categories
    hydra_multilabel is a dictionary not a list  - {netname:conf,....}
    '''

    print('\n\ncombining multilabel w. neurodoll_v3. required imsize:'+str(required_image_size))
    multilabel = label_conversions.hydra_to_pixlevel_v3(hydra_multilabel) #gives list of lists one list for each pixlevel v3 group
#    logging.info('multi label:'+str(multilabel)+' len:'+str(len(multilabel)))
    print('hydra multilabel:'+str(hydra_multilabel))
    print('converted multilabel:'+str(multilabel))
    print('graylevel shape'+str(graylevel_nd_output.shape))
   # pdb.set_trace()
    #todo take out this extra call when sure abot do_graylevel_zeroing
    pixlevel_categorical_output = np.argmax(graylevel_nd_output,axis=2) #the returned mask is HxWxC so take max along C
    final_mask = np.copy(pixlevel_categorical_output)
    modified_graylevels = np.copy(graylevel_nd_output)
    logging.info('size of final mask '+str(final_mask.shape))

######saving incoming mask for debug
    name = orig_filename+'_stage0output.png'
    print('combined png name:'+name+' orig filename '+orig_filename)
    final_mask = modified_graylevels.argmax(axis=2)
    cv2.imwrite(name,final_mask)
    nice_output = imutils.show_mask_with_labels(name,constants.pixlevel_categories_v3,save_images=True,original_image=orig_filename+'.jpg',visual_output=False)


    #donate instead of remove small areas so dont do next line (thresold_pixlevel)
#    pixlevel_categorical_output = threshold_pixlevel(pixlevel_categorical_output) #threshold out the small areas
    if do_graylevel_zeroing:  #kill the graylevels not in the ml list
        print('counts before graylevel zeroing:')
        count_values(pixlevel_categorical_output,labels=constants.ultimate_21)
        graylevel_nd_output = zero_graylevels_not_in_ml(graylevel_nd_output,multilabel,threshold=0.7)
        pixlevel_categorical_output = graylevel_nd_output.argmax(axis=2) #the returned mask is HxWxC so take max along C
        pixlevel_categorical_output = threshold_pixlevel(pixlevel_categorical_output) #threshold out the small areas
    print('counts (after posible graylevel zeroing):')
    count_values(pixlevel_categorical_output,labels=labels)
 #   count_values(final_mask,labels=labels)
    foreground = np.array((pixlevel_categorical_output>0) * 1)
    background = np.array((pixlevel_categorical_output==0) * 1)
    #    item_masks =  nfc.pd(image, get_all_graylevels=True)
    logging.debug('shape of pixlevel categorical output:'+str(pixlevel_categorical_output.shape))
    logging.debug('n_fg {} n_bg {} tot {} w*h {}'.format(np.sum(foreground),np.sum(background),np.sum(foreground)+np.sum(background),pixlevel_categorical_output.shape[0]*pixlevel_categorical_output.shape[1]))

    first_time_thru = True  #hack to dtermine image size coming back from neurodoll

    print('final_mask shape '+str(final_mask.shape))

    if face:
        y_split = face[1] + 3 * face[3]
    else:
        # BETTER TO SEND A FACE
        y_split = np.round(0.4 * final_mask.shape[0])
    print('a y split {} face {}'.format(y_split,face))

    #1. using ml results decide on whole body item (dress, suit, overall) vs. non-whole body (two part e.g. skirt+top) items.
            #max(whole body confidences) vs max(upper_cover), max(upper_UNDER), max(lower_cover), etc
    #2. if wholebody beats two-part - donate all non-whole-body pixels to whole body (except upper-cover (jacket/blazer etc)  and lower under-stockings)
    #3. take winning upper cover,  donate losers to winner
    #4. take winning upper under, donate losers to winner
    #5. take winning lower cover, donate losers to winner.
    #6. take winning lower under, donate losers to winner
    #?  if no upper cover and no upper under and no whole-body: take max of all those and donate losers to winner

# pixlevel_categories_v3 = ['bgnd','whole_body_items', 'whole_body_tight_items','undie_items','upper_under_items',
#                           'upper_cover_items','lower_cover_long_items','lower_cover_short_items','footwear_items','wraparound_items',
#                           'bag','belt','eyewear','hat','tie','skin']

# pixlevel3_whole_body = ['dress','suit','overalls','tracksuit','sarong','robe','pajamas' ]
# pixlevel3_whole_body_tight = ['womens_swimwear_nonbikini','womens_swimwear_bikini','lingerie','bra']
# pixlevel3_level_undies = ['mens_swimwear','mens_underwear','panties']
# pixlevel3_upper_under = ['shirt']  #nite this is intead of top
# pixlevel3_upper_cover = ['cardigan','coat','jacket','sweatshirt','sweater','blazer','vest','poncho']
# pixlevel3_lower_cover_long = ['jeans','pants','stocking','legging','socks']
# pixlevel3_lower_cover_short = ['shorts','skirt']
# pixlevel3_wraparwounds = ['shawl','scarf']
# pixlevel3__pixlevel_footwear = ['boots','shoes','sandals']

#possible improvement - compare nd and multilabel results for 'combined confidence' e.g. based on # pixels


    whole_body_ml_values = np.array([v for k,v in multilabel[1].iteritems()])  #does not necessadily preserve order
    print('ml1:'+str(multilabel[1]))
    print('wholebody ml_values:'+str(whole_body_ml_values))
    whole_body_winner = whole_body_ml_values.argmax()
    whole_body_winner_value=whole_body_ml_values[whole_body_winner]
    print('winning index:'+str(whole_body_winner)+' value:'+str(whole_body_winner_value))



    upper_cover_ml_values = np.array([v for k,v in multilabel[5].iteritems()])
    print('upper_cover ml_values:'+str(upper_cover_ml_values))
    if upper_cover_ml_values != []:
        upper_cover_winner = upper_cover_ml_values.argmax()
        upper_cover_winner_value=upper_cover_ml_values[upper_cover_winner]
        print('winning upper_cover index :'+str(upper_cover_winner)+' value:'+str(upper_cover_winner_value))
    else:
        upper_cover_winner_value=0

    upper_under_ml_values = np.array([v for k,v in multilabel[4].iteritems()])
    print('upper_under ml_values:'+str(upper_under_ml_values))
    if upper_under_ml_values != []:
        upper_under_winner = upper_under_ml_values.argmax()
        upper_under_winner_value=upper_under_ml_values[upper_under_winner]
        print('winning upper_under index :'+str(upper_under_winner)+' value:'+str(upper_under_winner_value))
    else:
        upper_under_winner_value=0

    lower_cover_long_ml_values = np.array([v for k,v in multilabel[6].iteritems()])
    print('lower_cover_long ml_values:'+str(lower_cover_long_ml_values))
    if lower_cover_long_ml_values!=[]:
        lower_cover_long_winner = lower_cover_long_ml_values.argmax()
        lower_cover_long_winner_value=lower_cover_long_ml_values[lower_cover_long_winner]
        print('winning lower_long  index :'+str(lower_cover_long_winner)+' value:'+str(lower_cover_long_winner_value))
    else:
        lower_cover_long_winner=0

    lower_cover_short_ml_values = np.array([v for k,v in multilabel[7].iteritems()])
    print('lower_cover_short ml_values:'+str(lower_cover_short_ml_values))
    if lower_cover_short_ml_values !=[]:
        lower_cover_short_winner = lower_cover_short_ml_values.argmax()
        lower_cover_short_winner_value=lower_cover_short_ml_values[lower_cover_short_winner]
        print('winning lower_short index :'+str(lower_cover_short_winner)+' value:'+str(lower_cover_short_winner_value))
    else:
        lower_cover_short_winner = 0

    #for use later, decide on a winner between upper cover and upper under
    # WHY do this when both can be there?
    if upper_under_winner_value > upper_cover_winner_value:
        upper_winner_value = upper_under_winner_value
        logging.debug('upper under {} > upper cover {}'.format(upper_under_winner_value,upper_cover_winner_value))
    else:
        upper_winner_value = upper_cover_winner_value
        logging.debug('upper under {} <= upper cover {}'.format(upper_under_winner_value,upper_cover_winner_value))

    #for use later, decide on a winner between lower cover long and short
    if lower_cover_long_winner_value > lower_cover_short_winner_value:
        lower_winner_value = lower_cover_long_winner_value
        lower_winner_index = lower_cover_long_winner
        logging.debug('lower cover long {} > lower cover short {}'.format(lower_cover_long_winner_value,lower_cover_short_winner_value))
    else:
        lower_winner_value = lower_cover_short_winner_value
        lower_winner_index = lower_cover_short_winner
        logging.debug('lower cover long {} <= lower cover short {}'.format(lower_cover_long_winner_value,lower_cover_short_winner_value))

#########################
# 1. WHOLEBODY VS TWO-PART
# decide on whole body item (dress, suit, overall) vs. non-whole body items.
    #do this using the multilabel results. maybe take stats on occurences for improvement
# case 1 - wholebody wins  - ( wholebody>max(upper_under/cover>lower_coverlong/short)
# case 2 - two piece wins  - donate wholebody to
#            a. upper_cover if upper_cover>upper_under (ml results)
#            b. upper_under if upper_cover<upper_cover
#            c. winner of lower_cover_short, lower_cover_long

#forget about this approach:
# case 1 wholebody>max(upper_under/cover>lower_coverlong/short)
# case 2 upper_under/cover>wholebody>lower_cover
# case 3 lower_cover>wholebody>upper-under
# case 4 lower_cover,upper_under > wholebody


#consider reducing this to nadav's method:
#    whole_sum = np.sum([item.values()[0] for item in mask_sizes['whole_body']])
#    partly_sum = np.sum([item.values()[0] for item in mask_sizes['upper_under']]) +\
#                 np.sum([item.values()[0] for item in mask_sizes['lower_cover']])
#                if whole_sum > partly_sum:
#    donate partly to whole
# its a little different tho since in multilabel you cant compare directly two items to one , e.g. if part1 = 0.6, part2 = 0.6, and whole=0.99, you
# should prob go with whole even tho part1+part2>whole
#########################
    #note the below case allows for whole_body and upper_cover to coexist which is unlikely but possible, eg dress+jacket
#    pdb.set_trace()

    non_whole_body_max = max(lower_cover_short_winner_value,lower_cover_long_winner_value,upper_under_winner_value)
    ############  WHOLE BODAAAY
    print('\nDB 50')
    if whole_body_winner_value>non_whole_body_max:
        print('whole body wins according to ml ({} vs {}'.format(whole_body_winner_value,non_whole_body_max))
        donor_cat_indices = []
        donor_cat_indices.append(constants.pixlevel_categories_v3.index('lower_cover_short_items'))
        donor_cat_indices.append(constants.pixlevel_categories_v3.index('lower_cover_long_items'))
        donor_cat_indices.append(constants.pixlevel_categories_v3.index('upper_under_items'))
        #maybe check if upper_cover has a high value ...
#        donor_cat_indices.append(constants.pixlevel_categories_v3.index('upper_cover_items')) #not sure whether to add these since they may coexist
        whole_body_index=constants.pixlevel_categories_v3.index('whole_body_items')
        for id in donor_cat_indices:
            print('donating layer {} {} to {} {}'.format(id,constants.pixlevel_categories_v3[id],
                                                         whole_body_index,constants.pixlevel_categories_v3[whole_body_index]))
        modified_graylevels = donate_graylevels(modified_graylevels,donor_cat_indices,whole_body_index) #donate nonwholebody to wholebody

######saving interim for debug
        name = orig_filename+'_stage1output.png'
        print('combined png name:'+name+' orig filename '+orig_filename)
        final_mask = modified_graylevels.argmax(axis=2)
        cv2.imwrite(name,final_mask)
        nice_output = imutils.show_mask_with_labels(name,constants.pixlevel_categories_v3,save_images=True,original_image=orig_filename+'.jpg',visual_output=False)

    ############  TWO PART
    else:  #donate lower loser to winner - upper can stay since uppercover and upperunder can coexist
        print('nonwhole body wins according to ml ({} vs {}'.format(whole_body_winner_value,non_whole_body_max))
        if upper_under_winner_value > upper_cover_winner_value:
            print('upper under wins according to ml (cover {} vs under {}'.format(upper_cover_winner_value,upper_under_winner_value))
            upper_winner_value = upper_under_winner_value
            upper_winner_index = constants.pixlevel_categories_v3.index('upper_under_items')
            upper_loser_index = constants.pixlevel_categories_v3.index('upper_cover_items')
        else:
            print('upper cover wins according to ml (cover {} vs under {}'.format(upper_cover_winner_value,upper_under_winner_value))
            upper_winner_value = upper_cover_winner_value
            upper_winner_index = constants.pixlevel_categories_v3.index('upper_cover_items')
            upper_loser_index = constants.pixlevel_categories_v3.index('upper_under_items')

        if lower_cover_short_winner_value>lower_cover_long_winner_value:
            print('lower cover short wins according to ml (s {} vs l {}'.format(lower_cover_short_winner_value,lower_cover_long_winner_value))
            lower_winner_value = lower_cover_short_winner_value
            lower_winner_index = constants.pixlevel_categories_v3.index('lower_cover_short_items')
            lower_loser_index = constants.pixlevel_categories_v3.index('lower_cover_long_items')
        else :
            print('lower cover long wins according to ml (s {} vs l {}'.format(lower_cover_short_winner_value,lower_cover_long_winner_value))
            lower_winner_value = lower_cover_long_winner_value
            lower_winner_index = constants.pixlevel_categories_v3.index('lower_cover_long_items')
            lower_loser_index = constants.pixlevel_categories_v3.index('lower_cover_long_items')

        print('upper winner {} {} loser {} {}'.format(upper_winner_index,constants.pixlevel_categories_v3[upper_winner_index],upper_loser_index,constants.pixlevel_categories_v3[upper_loser_index]))
        print('lower winner {} {} loser {} {}'.format(lower_winner_index,constants.pixlevel_categories_v3[lower_winner_index],lower_loser_index,constants.pixlevel_categories_v3[lower_loser_index]))

    #donate lower loser to winner
        donor_cat_indices = []
        donor_cat_indices.append(lower_loser_index)
        recipient_index = lower_winner_index
        modified_graylevels = donate_graylevels(modified_graylevels,donor_cat_indices,recipient_index)
        whole_body_index=constants.pixlevel_categories_v3.index('whole_body_items')
        upper_under_index = constants.pixlevel_categories_v3.index('upper_under_items')

        #ARBITRARY addition of wholebody to upper_under instead of uppercover, maybe compare those or do half-half or acc. to hydra...
        modified_graylevels = donate_graylevels_upper_and_lower(modified_graylevels,whole_body_index,upper_under_index,lower_winner_index,y_split)

# donate_graylevels_upper_and_lower(graylevels,donor_index,upper_winner_index,lower_winner_index,y_split):

######saving interim for debug
    name = orig_filename+'_stage3output.png'
    print('combined png name:'+name+' orig filename '+orig_filename)
    final_mask = modified_graylevels.argmax(axis=2)
    cv2.imwrite(name,final_mask)
    nice_output = imutils.show_mask_with_labels(name,constants.pixlevel_categories_v3,save_images=True,original_image=orig_filename+'.jpg',visual_output=False)

    modified_graylevels = threshold_graylevels(modified_graylevels)
    name = orig_filename+'_stage4output.png'
    print('combined png name:'+name+' orig filename '+orig_filename)
    final_mask = modified_graylevels.argmax(axis=2)
    cv2.imwrite(name,final_mask)
    nice_output = imutils.show_mask_with_labels(name,constants.pixlevel_categories_v3,save_images=True,original_image=orig_filename+'.jpg',visual_output=False)

    print('\nDB 99')

    #note recalc final mask after donation
    print('counting after stage4')
    count_values(final_mask,labels=labels)
#    final_mask = modified_graylevels.argmax(axis=2)
    print('final mask size '+str(final_mask.shape))
    final_mask = v3_graylevels_to_u21_cats(final_mask,multilabel)
    print('final mask pixcount')
    count_values(final_mask,labels=constants.ultimate_21)

    #write file (for debugging)
    name = orig_filename+'_finaloutput.png'
    print('combined png name:'+name+' orig filename '+orig_filename)
    cv2.imwrite(name,final_mask)
    nice_output = imutils.show_mask_with_labels(name,constants.ultimate_21,save_images=True,original_image=orig_filename+'.jpg',visual_output=False)
    #save graymask, this should be identical to nd except no threshold on low amt of pixels

##    foreground = np.array((pixlevel_categorical_output>0)*1)  #*1 turns T/F into 1/0
 #   final_mask = final_mask * foreground # only keep stuff that was part of original fg - this is already  true
    # unless we start adding pixvalues that didn't 'win'
    print('\nDB END')
    return final_mask



def donate_to_upper_and_lower(final_mask,donor_indexlist,upper_winner_nd_index,lower_winner_nd_index,multilabel_to_ultimate21_conversion,y_split):
    logging.info('donating from whole_body  {} to  upper under {} and lower cover {}'.format(donor_indexlist,upper_winner_nd_index,lower_winner_nd_index))
    if upper_winner_nd_index is None:
        logging.warning('nd wholebody index {} '.format(upper_winner_nd_index))
    else:
        n1 = len(final_mask[final_mask==upper_winner_nd_index])
        n2 = len(final_mask[final_mask==lower_winner_nd_index])
        logging.info('n in final mask from wholebody before donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
        #todo - actually only wholebody pixels in the upper half of the image should be donated
        for i in donor_indexlist: #whole_body donated to upper_under
            nd_index = multilabel_to_ultimate21_conversion[i]
            if nd_index is None:
                logging.warning('ml index {} has no conversion (4upper)'.format(i))
                continue            #donate upper pixels to upper_winner
            logging.debug('3. donating nd {} in top of wholebody to upper_under and bottom to lower_under'.format(nd_index))
            logging.debug('first adding from upper part of nd {}  to nd {}, ysplit {}'.format(nd_index,upper_winner_nd_index,y_split))
            for y in range(0, final_mask.shape[0]):
                if y <= y_split:
                    for x in range(0, final_mask.shape[1]):
                        if final_mask[y][x] == nd_index:
                            final_mask[y][x] = upper_winner_nd_index
        n1 = len(final_mask[final_mask==upper_winner_nd_index])
        n2 = len(final_mask[final_mask==lower_winner_nd_index])
        logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))

# donate whole-body pixels to lower winner
    if lower_winner_nd_index is None:
        logging.warning('nd wholebody index {} has no conversion '.format(upper_winner_nd_index))
    else:
        n1 = len(final_mask[final_mask==upper_winner_nd_index])
        n2 = len(final_mask[final_mask==lower_winner_nd_index])
        logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} , lower winner {} px, nd {} '.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
        #todo - actually only wholebody pixels in the upper half of the image should be donated
        for i in donor_indexlist: #whole_body donated to upper_under
            nd_index = multilabel_to_ultimate21_conversion[i]
            if nd_index is None:
                logging.warning('ml index {} has no conversion (4lower)'.format(i))
                continue
    #donate upper pixels to upper_winner
            logging.debug('3. donating nd {} in botto of wholebody to upper_under and bottom to lower_under'.format(nd_index))
            logging.debug('second, adding from lower part of nd {}  to nd {}, ysplit {}'.format(nd_index,lower_winner_nd_index,y_split))
            for y in range(0, final_mask.shape[0]):
                if y > y_split:
                    for x in range(0, final_mask.shape[1]):
                        if final_mask[y][x] == nd_index:
                            final_mask[y][x] = lower_winner_nd_index
    #donate upper pixels to lower_winner
        n1 = len(final_mask[final_mask==upper_winner_nd_index])
        n2 = len(final_mask[final_mask==lower_winner_nd_index])
        logging.info('n in final mask from wholebody donation to upper {} and lower {}:'.format(n1,n2))

def v3_graylevels_to_u21_cats(pixlevel_v3_categorical,multilabel,two_part=True):
    '''
    take v3 categorical (category per pixel) output and convert to ultimate_21 using multilabel
    to decide which ultamte21 label to give to each ultimate21 cat
    :param graylevels:
    :param multilabel:raw multilabel results
    :return:
    '''
    #ultimate_21 = ['bgnd','bag','belt','blazer','coat','dress','eyewear','face','hair','hat',
#               'jeans','leggings','pants','shoe','shorts','skin','skirt','stockings','suit','sweater',
#               'top']

#    pixlevel_categories_v3 = ['bgnd','whole_body_items', 'whole_body_tight_items','undie_items','upper_under_items',
#                          'upper_cover_items','lower_cover_long_items','lower_cover_short_items','footwear_items','wraparound_items',
#                          'bag','belt','eyewear','hat','tie','skin']


    v3_whole_body_to_u21=[5,18,12,12,5,5,12,18]
#    v3_swimwear_to_u21
#    v3_undies_to_u21
    v3_upper_cover_to_u21=[4,4]
    u21_results = np.zeros_like(pixlevel_v3_categorical)
    u21_results[:,:]=0 #np.zeros initializes with float not int
#    converted = label_conversions.hydra_to_u21(multilabel)
    print('incoming ml:'+str(multilabel))
#    print('test conversion hydra-u21:'+str(converted))
    #convert whole_body to the winning whole_body
    for u in np.unique(pixlevel_v3_categorical):
        print('working on nd index {} {} from nd, ml {}'.format(u,constants.pixlevel_categories_v3[u],multilabel[u]))
        if u==0:
            continue
        if multilabel[u] == {}:
      #      pdb.set_trace()
            print('empty ml for index {} {}'.format(u,constants.pixlevel_categories_v3[u]))
            pixlevel_label = constants.pixlevel_categories_v3[u]
            if pixlevel_label in constants.ultimate_21:
                u21_cat = constants.ultimate_21.index(pixlevel_label)
            # reverse_lookup = {v:k for k,v in constants.ultimate_21_dict.iteritems()}
            # if not u in reverse_lookup:
            #     logging.debug('could not fnd {} in reversedict {}'.format(u,reverse_lookup))
            #     continue
            # u21_cat = reverse_lookup[u]
            else:
                logging.warning('got no u21 category in inner check for '+str(constants.pixlevel_categories_v3[u])+' '+pixlevel_label)
                continue
            print('converting hydra with no opinion {} [] to u21 {}'.format(u,constants.pixlevel_categories_v3[u],u21_cat,constants.ultimate_21[u21_cat]))
            u21_results=u21_results+(pixlevel_v3_categorical==u)*u21_cat
            continue
        maxkey= max(multilabel[u].iteritems(), key=operator.itemgetter(1))[0]
        u21_cat = label_conversions.multilabels_from_hydra_to_u21_cat(maxkey)
        if not u21_cat:
            logging.warning('got no u21 category for '+str(constants.pixlevel_categories_v3[u]))
            continue
        print('u21 index {} cat {} maxkey {}'.format(u21_cat,constants.ultimate_21[u21_cat],maxkey))
   #     values = np.array([v for k,v in multilabel[u].iteritems()])  #does not necessadily preserve order
        u21_results=u21_results+(pixlevel_v3_categorical==u)*u21_cat
        nonzero_count=np.count_nonzero(u21_results)
        print('nonzero count after adding: {}'.format(nonzero_count))
    return u21_results



def donate_graylevels(mask_layers,donor_layers,recipient_layer,labels=constants.pixlevel_categories_v3):
    '''
    donate pixel values - mask layers are the n grayscale ayers
    :param mask_layers:
    :param donor_layers: list of which layers are donating
    :param recipient_layer: which layer is receiving
    :return:
    '''
    print('donating graylevels from {} to {}, shape {}'.format(donor_layers,recipient_layer,mask_layers.shape))
    mask = np.argmax(mask_layers,axis=2)
    print('pixel count before donation')
    count_values(mask,labels)
    initial_sum=np.sum(mask_layers)
    for d in donor_layers:
        mask_layers[:,:,recipient_layer] += mask_layers[:,:,d]
        mask_layers[:,:,d] = 0
    final_sum=np.sum(mask_layers)
    print('donate graylevels: initial sum {} final sum {}'.format(initial_sum,final_sum))
    mask = np.argmax(mask_layers,axis=2)
    print('pixel count after donation')
    count_values(mask,labels)
    return mask_layers


def donate_graylevels_upper_and_lower(graylevels,donor_index,upper_winner_index,lower_winner_index,y_split,labels=constants.pixlevel_categories_v3):
    logging.info('donating from {} {} to  {} {} and  {} {}'.format(donor_index,labels[donor_index],upper_winner_index,labels[upper_winner_index],lower_winner_index,labels[lower_winner_index]))
    logging.debug('first adding from upper part of nd {}  to nd {}, ysplit {}'.format(donor_index,upper_winner_index,y_split))
    mask = np.argmax(graylevels,axis=2)
    print('pixel count before 2part donation')
    count_values(mask,labels)
    #add upper part to upper winner index
    graylevels[0:y_split,:,upper_winner_index] += graylevels[0:y_split,:,donor_index]
    #add lower part to lower winner index
    graylevels[y_split:,:,lower_winner_index] += graylevels[y_split:,:,donor_index]
    graylevels[:,:,donor_index]=0
    mask = np.argmax(graylevels,axis=2)
    print('pixel count after 2part donation')
    count_values(mask,labels)
    return graylevels


if __name__ == "__main__":
    outmat = np.zeros([256*4,256*21],dtype=np.uint8)
    url = 'http://pinmakeuptips.com/wp-content/uploads/2015/02/1.4.jpg'
    urls = ['http://healthyceleb.com/wp-content/uploads/2014/03/Nargis-Fakhri-Main-Tera-Hero-Trailer-Launch.jpg',
            'http://pinmakeuptips.com/wp-content/uploads/2015/02/1.4.jpg',
            'http://diamondfilms.com.au/wp-content/uploads/2014/08/Fashion-Photography-Sydney-1.jpg',
            'http://pinmakeuptips.com/wp-content/uploads/2016/02/main-1.jpg',
            'http://pinmakeuptips.com/wp-content/uploads/2016/02/1.-Strategic-Skin-Showing.jpg',
            'http://pinmakeuptips.com/wp-content/uploads/2016/02/3.jpg',
            'http://pinmakeuptips.com/wp-content/uploads/2016/02/4.jpg',
            'http://pinmakeuptips.com/wp-content/uploads/2016/03/Adding-Color-to-Your-Face.jpg',
            'http://images5.fanpop.com/image/photos/26400000/Cool-fashion-pics-fashion-pics-26422922-493-700.jpg',
            'http://allforfashiondesign.com/wp-content/uploads/2013/05/style-39.jpg',
            'http://s6.favim.com/orig/65/cool-fashion-girl-hair-Favim.com-569888.jpg',
            'http://s4.favim.com/orig/49/cool-fashion-girl-glasses-jeans-Favim.com-440515.jpg',
            'http://s5.favim.com/orig/54/america-blue-cool-fashion-Favim.com-525532.jpg',
            'http://favim.com/orig/201108/25/cool-fashion-girl-happiness-high-Favim.com-130013.jpg'
    ] #
    test_nd_alone = False
    if test_nd_alone:
        raw_input('start test_nd_alone')
        for url in urls:
            #infer-one saves results depending on switch at end
            print('testing nd alone')
            result = infer_one(url)
            print result

#    after_nn_result = pipeline.after_nn_conclusions(result,constants.ultimate_21_dict)
#    cv2.imwrite('output_afternn.png',after_nn_result)
#   labels=constants.ultimate_21
#    get_nd_and_multilabel_output_using_nfc(url_or_np_array)
#    out = get_multilabel_output(url)
#    print('ml output:'+str(out))
    #get neurdoll output alone

    test_nfc_nd = False
    if test_nfc_nd:
        raw_input('start test_nfc_nd_alone')
        for url in urls:
            print('testing nfc_nd')
            nd_out = get_neurodoll_output_using_falcon(url)
            orig_filename = '/home/jeremy/'+url.split('/')[-1]
            urllib.urlretrieve(url, orig_filename)
            name = orig_filename[:-4]+'_nd_output.png'
            cv2.imwrite(name,nd_out)
            nice_output = imutils.show_mask_with_labels(name,constants.ultimate_21,save_images=True,original_image=orig_filename)


    #get_category_graylevel(urls[0],category_index = 3)
    test_graylevels = False
    if test_graylevels:
        for i in range(21):
            get_category_graylevel(url,category_index = i)

    test_gcgl = False
    if test_gcgl:
        print('start test_combined_nd')
        for url in urls:
            print('doing url:'+url)
#            for i in range(len(constants.ultimate_21)):
            i = 5 #dress
            get_category_graylevel_masked_thresholded(url,i)
            i = 16 #skirt
            get_category_graylevel_masked_thresholded(url,i)
#
#    analyze_graylevels(urls[0])
#            analyze_graylevels(url)
#    get_category_graylevel(urls[0],4)

    #get output of combine_nd_and_ml
    test_combine = False
    if test_combine:
        print('start test_combined_nd')
        for url in urls: #
            print('doing url:'+url) #
            out = combine_neurodoll_and_multilabel(url,output_layer='pixlevel_sigmoid_output',required_image_size=(224,224))

#            for median_factor in [0.75]:
#            for median_factor in [0.5,0.75,1,1.25,1.5]:
#                print('testing combined ml nd, median factor:'+str(median_factor))
#                print('combined output:'+str(out))


