#!/usr/bin/env python
__author__ = 'jeremy'

import caffe
import logging
import copy
logging.basicConfig(level=logging.DEBUG)
from PIL import Image
import cv2
import caffe
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
import os
import time
import hashlib
import urllib
import sys

from trendi import constants
from trendi.utils import imutils
from trendi import Utils
from trendi.paperdoll import binary_multilabel_falcon_client as bmfc
from trendi.paperdoll import binary_multilabel_falcon_client2 as bmfc2
from trendi.paperdoll import binary_multilabel_falcon_client3 as bmfc3
from trendi.paperdoll import neurodoll_falcon_client as nfc


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
protopath = os.path.join(os.path.dirname(os.path.abspath( __file__ )), 'classifier_stuff/caffe_nns/protos')
modelpath = '/home/jeremy/caffenets/production'
#MODEL_FILE = os.path.join(modelpath,'voc8_15_pixlevel_deploy.prototxt')
MODEL_FILE = os.path.join(modelpath,'voc8_15_pixlevel_deploy_with_sigmoid.prototxt')
#PRETRAINED = os.path.join(modelpath,'voc8_15_pixlevel_iter120000.caffemodel')
PRETRAINED = os.path.join(modelpath,'voc8_15_0816_iter10000_pixlevel_deploy.caffemodel')

test_on = True #
if test_on:
    gpu = int(sys.argv[1])
    print('using gpu '+str(gpu))
else:
    gpu = 0
caffe.set_mode_gpu()
caffe.set_device(gpu)
net = caffe.Net(MODEL_FILE,PRETRAINED, caffe.TEST)
#required_image_size = (256, 256)
required_image_size = None
image_mean = np.array([107.0,117.0,123.0])
input_scale = None
channel_swap = [2, 1, 0]
raw_scale = 255.0
print('done loading caffemodel for neurodoll')

#best multilabel as of 260716, see http://extremeli.trendi.guru/demo/results/ for updates
multilabel_from_binaries = True
if not multilabel_from_binaries: #dont need this if answers are coming from multilabel_from_binaries. otherwise get the multilabel net
    print('starting up multilabel net')
    protopath = os.path.join(os.path.dirname(os.path.abspath( __file__ )), 'classifier_stuff/caffe_nns/protos')
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
    print('using multilabel from binaries (thru falcon) ')
multilabel_required_image_size = (224,224)


# Make classifier.
#classifier = caffe.Classifier(MODEL_FILE, PRETRAINED,
#                              image_dims=image_dims, mean=mean,
##                              input_scale=input_scale, raw_scale=raw_scale,
 #                             channel_swap=channel_swap)



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

def get_layer_output(url_or_np_array,required_image_size=(256,256),layer='myfc7'):
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

def infer_one(url_or_np_array,required_image_size=(256,256),item_area_thresholds = constants.ultimate_21_area_thresholds):
    start_time = time.time()
    thedir = './images'
    Utils.ensure_dir(thedir)
    if isinstance(url_or_np_array, basestring):
        print('infer_one working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
        orig_filename = os.path.join(thedir,url_or_np_array.split('/')[-1])
    elif type(url_or_np_array) == np.ndarray:
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        name_base = 'orig'+hash.hexdigest()[:10]+'.jpg'
        orig_filename = os.path.join(thedir,name_base)
        image = url_or_np_array
    if image is None:
        logging.debug('got None in grabcut_using_neurodoll_output')
    print('writing orig to '+orig_filename)
    cv2.imwrite(orig_filename,image)

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
            print('got  image with shape '+str(in_.shape)+' , turning into 3 channel')
            in_ = np.array([copy.deepcopy(in_),copy.deepcopy(in_),copy.deepcopy(in_)])
            print('now image has shape '+str(in_.shape))
    elif in_.shape[2] != 3:
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return None
#    in_ = in_[:,:,::-1]  for doing RGB -> BGR
#    cv2.imwrite('test1234.jpg',in_) #verify that images are coming in as rgb

    in_ -= np.array([104,116,122.0])  #was not used in training!!
    in_ = in_.transpose((2,0,1))   #wxhxc -> cxwxh
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
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
    image_size = out.shape[0]*out.shape[1]
    uniques = np.unique(out)


#TODO - make the threshold per item ,e.g. small shoes are ok and should be left in
    for unique in uniques:
        pixelcount = len(out[out==unique])
        ratio = float(pixelcount)/image_size
#        logging.debug('i {} pixels {} tot {} ratio {} threshold {} ratio<thresh {}'.format(unique,pixelcount,image_size,ratio,threshold,ratio<threshold))
        threshold = item_area_thresholds[unique]
        print('index {}  ratio {} threshold {}'.format(unique,ratio,threshold))
        if ratio < threshold:
#            logging.debug('kicking out index '+str(unique)+' with ratio '+str(ratio))
            out[out==unique] = 0  #set label with small number of pixels to 0 (background)
            pixelcount = len(out[out==unique])
#            logging.debug('new ratio '+str(ratio))


   # cv2.countNonZero(item_mask)


#    result = Image.fromarray(out.astype(np.uint8))
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
#    return out.astype(np.uint8)
    out = np.array(out,dtype=np.uint8)
    save_results = True
    if save_results:
        pngname = orig_filename[:-4]+'.png'
        cv2.imwrite(filename=pngname,img=out)
        imutils.show_mask_with_labels(pngname,labels=constants.ultimate_21,visual_output=True,save_images=True,original_image=orig_filename)
    uniques = np.unique(out)
    logging.debug('final uniques:'+str(uniques))
    count_values(out,labels=constants.ultimate_21)
    return out

def get_multilabel_output(url_or_np_array,required_image_size=(224,224)):
#################################
#todo - parallelize the first if#
#################################

    if multilabel_from_binaries:
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

def get_neurodoll_output(url_or_np_array):
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

def get_all_category_graylevels(url_or_np_array,required_image_size=(256,256)):
    start_time = time.time()
    if isinstance(url_or_np_array, basestring):
        print('get_all_category_graylevels working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    if required_image_size is not None:
        original_h, original_w = image.shape[0:2]
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
    out = net.blobs['siggy'].data[0] #for layer score, all outputs after softmax
    min = np.min(out)
    max = np.max(out)
    print('get_all_category_graylevels output shape {} min {} max {}'.format(out.shape,min,max))
    out = out*255
    min = np.min(out)
    max = np.max(out)
    print('min {} max {} out after scaling'.format(min,max))
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
    print('get_all_categorygraylevels:outshape '+str(out.shape))
#    out = out.transpose((2,0,1))  #change row,col,chan to chan,row,col as caffe wants
    out = out.transpose((1,2,0))  #change chan,row,col to row,col,chan  as the rest of world wants
    print('get_all_categorygraylevels:outshape '+str(out.shape))
    if required_image_size is not None:
        logging.debug('resizing nd input back to '+str(original_h)+'x'+str(original_w))
        out = imutils.undo_resize_keep_aspect(out,output_size=(original_h,original_w),careful_with_the_labels=True)
        print('get_all_categorygraylevels after reshape: '+str(out.shape))
    print('get_all_category_graylevels elapsed time:'+str(elapsed_time))
    return out

def analyze_graylevels(url_or_np_array,labels=constants.ultimate_21):
    if isinstance(url_or_np_array, basestring):
        print('get_all_category_graylevels working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array

    gl = get_all_category_graylevels(url_or_np_array)
    h,w = gl.shape[0:2]
    window_size = 1700
    n_rows=5
    compress_factor = max(float(h*n_rows)/window_size,float(w*n_rows)/window_size)
    compressed_h = int(h/compress_factor)
    compressed_w = int(w/compress_factor)
    compressed_gl = cv2.resize(gl,(compressed_w,compressed_h))
    print('compressed hw {} {}'.format(compressed_h,compressed_w))
    compressed_image = cv2.resize(image,(compressed_w,compressed_h))
    big_out = np.zeros([compressed_h*n_rows,compressed_w*n_rows,3])
    print('bigsize:'+str(big_out.shape))

    for i in range(5):
        for j in range(5):
            n = i*n_rows+j
            print('n:'+str(n))
            if n>=gl.shape[2]:
                big_out[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,:] = compressed_image

                break
            print('y0 {} y1 {} x0 {} x1 {}'.format(i*h,(i+1)*h,j*w,(j+1)*w))
            big_out[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,0] = compressed_gl[:,:,n]
            big_out[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,1] = compressed_gl[:,:,n]
            big_out[i*compressed_h:(i+1)*compressed_h,j*compressed_w:(j+1)*compressed_w,2] = compressed_gl[:,:,n]
            print('tx {} ty {}'.format(int((j+0.5)*w),int((i+1)*h-10)))
            cv2.putText(big_out,labels[n],(int((j+0.5)*compressed_w),int((i+1)*compressed_h-10)),cv2.FONT_HERSHEY_PLAIN,2,(150,100,255),thickness=2)
            cv2.imwrite('bigout.jpg',big_out)
#            cv2.imshow('bigout',big_out)

def get_all_category_graylevels_ineff(url_or_np_array,required_image_size=(256,256)):
    start_time = time.time()
    if isinstance(url_or_np_array, basestring):
        print('get_all_category_graylevels working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#    im = Image.open(imagename)
#    im = im.resize(required_imagesize,Image.ANTIALIAS)
#    in_ = in_.astype(float)
    #todo allow first resize then crop (e.g. resize to 250x250 then crop to 224x224)
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
    out = net.blobs['score'].data[0] #for all output layers
#    out = net.blobs['score'].data[0].argmax(axis=0) #for a parse with per-pixel max
#    out = net.blobs['siggy'].data[0] #for layer siggy, all outputs

    min = np.min(out)
    max = np.max(out)
    print('get_all_category_graylevels output shape {} min {} max {}'.format(out.shape,min,max))
    out = out*255
    min = np.min(out)
    max = np.max(out)
    print('min {} max {} out after scaling  {}'.format(min,max,out.shape))
#    result = Image.fromarray(out.astype(np.uint8))
#        outname = im.strip('.png')[0]+'out.bmp'
#    outname = os.path.basename(imagename)
#    outname = outname.split('.jpg')[0]+'.bmp'
#    outname = os.path.join(out_dir,outname)
#    print('outname:'+outname)
#    result.save(outname)
    #        fullout = net.blobs['score'].data[0]
    elapsed_time=time.time()-start_time
    print('get all category graylevels ineff elapsed time:'+str(elapsed_time))
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

#this is confusing : this is how you would call falcon which calls get_multilabel_output (above...)
def get_multilabel_output_using_nfc(url_or_np_array):
    print('starting get_multilabel_output_using_nfc')
    multilabel_dict = nfc.pd(url, get_multilabel_results=True)
    print('gmoun:dict from falcon dict:'+str(multilabel_dict))
    if not multilabel_dict['success']:
        print('did not get nfc pd result succesfully')
        return
    multilabel_output = multilabel_dict['multilabel_output']
    print('multilabel output:'+str(multilabel_output))
    return multilabel_output #

def test_conversions():
    multilabel_to_ultimate21_conversion=constants.binary_classifier_categories_to_ultimate_21
    multilabel_labels=constants.binary_classifier_categories
    print('testing binary classifier to u21 cats')
    print('ml2u21 conversion:'+str(multilabel_to_ultimate21_conversion))
    print('ml labels:'+str(multilabel_labels))
    for i in range(len(multilabel_labels)):
        neurodoll_index = multilabel_to_ultimate21_conversion[i]
        #print('nd index:'+str(neurodoll_index))
        if neurodoll_index is None:
            print('no mapping from index {} (label {}) to neurodoll'.format(i,multilabel_labels[i]))
            continue
        print('index {} webtoollabel {} newindex {} neurodoll_label {}'.format(i,
            multilabel_labels[i],neurodoll_index,constants.ultimate_21[neurodoll_index]))

    multilabel_to_ultimate21_conversion=constants.web_tool_categories_v1_to_ultimate_21
    multilabel_labels=constants.web_tool_categories
    print('testing webtool v2 to u21 cats')
    print('ml2u21 conversion:'+str(multilabel_to_ultimate21_conversion))
    print('ml labels:'+str(multilabel_labels))
    for i in range(len(multilabel_labels)):
        neurodoll_index = multilabel_to_ultimate21_conversion[i]
        if neurodoll_index is None:
            print('no mapping from index {} (label {}) to neurodoll'.format(i,multilabel_labels[i]))
            continue
        print('index {} webtoollabel {} newindex {} neurodoll_label {}'.format(i,
            multilabel_labels[i],neurodoll_index,constants.ultimate_21[neurodoll_index]))

    multilabel_to_ultimate21_conversion=constants.web_tool_categories_v2_to_ultimate_21
    multilabel_labels=constants.web_tool_categories_v2
    print('testing webtool v1 to u21 cats')
    print('ml2u21 conversion:'+str(multilabel_to_ultimate21_conversion))
    print('ml labels:'+str(multilabel_labels))
    for i in range(len(multilabel_labels)):
        neurodoll_index = multilabel_to_ultimate21_conversion[i]
        if neurodoll_index is None:
            print('no mapping from index {} (label {}) to neurodoll'.format(i,multilabel_labels[i]))
            continue
        print('index {} webtoollabel {} newindex {} neurodoll_label {}'.format(i,
            multilabel_labels[i],neurodoll_index,constants.ultimate_21[neurodoll_index]))

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

def combine_neurodoll_and_multilabel_onebyone(url_or_np_array,multilabel_threshold=0.7,median_factor=1.6,multilabel_to_ultimate21_conversion=constants.binary_classifier_categories_to_ultimate_21,multilabel_labels=constants.binary_classifier_categories):
    '''
    try product of multilabel and nd output and taking argmax
    multilabel_to_ultimate21_conversion=constants.web_tool_categories_v1_to_ultimate_21 , or
    multilabel_to_ultimate21_conversion=constants.binary_classifier_categories_to_ultimate_21

    multilabel_labels=constants.web_tool_categories    , or
    multilabel_labels=constants.binary_classifier_categories

    '''
    multilabel = get_multilabel_output(url_or_np_array)
#    multilabel = get_multilabel_output_using_nfc(url_or_np_array)
    #take only labels above a threshold on the multilabel result
    #possible other way to do this: multiply the neurodoll mask by the multilabel result and threshold that product
    if multilabel is None:
        logging.debug('None result from multilabel')
        return None
    thresholded_multilabel = [ml>multilabel_threshold for ml in multilabel] #
#    print('orig label:'+str(multilabel))
    print('combining multilabel w. neurodoll, watch out')
#    print('incoming label:'+str(multilabel))
#    print('thresholded label:'+str(thresholded_multilabel))
#    print('multilabel to u21 conversion:'+str(multilabel_to_ultimate21_conversion))
#    print('multilabel labels:'+str(multilabel_labels))

    if np.equal(thresholded_multilabel,0).all():  #all labels 0 - nothing found
        logging.debug('no items found')
        return #
#    item_masks =  nfc.pd(image, get_all_graylevels=True)

# hack to combine pants and jeans for better recall
#    pantsindex = constants.web_tool_categories.index('pants')
#    jeansindex = constants.web_tool_categories.index('jeans')
#   if i == pantsindex or i == jeansindex: #
    first_time_thru = True  #hack to dtermine image size coming back from neurodoll
    final_mask = np.zeros([224,224])

    for i in range(len(thresholded_multilabel)):
        if thresholded_multilabel[i]:
            neurodoll_index = multilabel_to_ultimate21_conversion[i]
            if neurodoll_index is None:
                print('no mapping from index {} (label {}) to neurodoll'.format(i,multilabel_labels[i]))
                continue
            print('index {} webtoollabel {} newindex {} neurodoll_label {} was above threshold {} (ml {})'.format(
                i,multilabel_labels[i],neurodoll_index,constants.ultimate_21[neurodoll_index], multilabel_threshold,multilabel[i]))
            item_mask = grabcut_using_neurodoll_output(url_or_np_array,neurodoll_index,median_factor=median_factor)
            if  item_mask is None:
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
    orig_filename = '/home/jeremy/'+url_or_np_array.split('/')[-1]
    name = '/home/jeremy/'+str(timestamp)+'.png'
    name = orig_filename[:-4]+'_mf'+str(median_factor)+'_output.png'
    print('name:'+name)
    cv2.imwrite(name,final_mask)
    orig_filename = '/home/jeremy/'+url_or_np_array.split('/')[-1]
    print('orig filename:'+str(orig_filename))
    nice_output = imutils.show_mask_with_labels(name,constants.ultimate_21,save_images=True,original_image=orig_filename)
#    nice_output = imutils.show_mask_with_labels(name,constants.ultimate_21,save_images=True)

    return final_mask

def combine_neurodoll_and_multilabel(url_or_np_array,multilabel_threshold=0.7,median_factor=1.0,multilabel_to_ultimate21_conversion=constants.binary_classifier_categories_to_ultimate_21,multilabel_labels=constants.binary_classifier_categories):
    '''
    try product of multilabel and nd output and taking argmax
    multilabel_to_ultimate21_conversion=constants.web_tool_categories_v1_to_ultimate_21 , or
    multilabel_to_ultimate21_conversion=constants.binary_classifier_categories_to_ultimate_21

    multilabel_labels=constants.web_tool_categories    , or
    multilabel_labels=constants.binary_classifier_categories

    '''
    thedir = './images'
    Utils.ensure_dir(thedir)
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
        orig_filename = os.path.join(thedir,url_or_np_array.split('/')[-1]+'.jpg')
    elif type(url_or_np_array) == np.ndarray:
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        name_base = 'orig'+hash.hexdigest()[:10]+'.jpg'
        orig_filename = os.path.join(thedir,name_base)
        image = url_or_np_array
    if image is None:
        logging.debug('got None in grabcut_using_neurodoll_output')
    print('writing orig to '+orig_filename)
    cv2.imwrite(orig_filename,image)

    multilabel = get_multilabel_output(url_or_np_array)
#    multilabel = get_multilabel_output_using_nfc(url_or_np_array)
    #take only labels above a threshold on the multilabel result
    #possible other way to do this: multiply the neurodoll mask by the multilabel result and threshold that product
    if multilabel is None:
        logging.debug('None result from multilabel')
        return None
    thresholded_multilabel = [ml>multilabel_threshold for ml in multilabel] #
#    print('orig label:'+str(multilabel))
    print('combining multilabel w. neurodoll, watch out')
#    print('incoming label:'+str(multilabel))
    print('thresholded label:'+str(thresholded_multilabel))
#    print('multilabel to u21 conversion:'+str(multilabel_to_ultimate21_conversion))
#    print('multilabel labels:'+str(multilabel_labels))

    #todo - this may be wrong later if we start taking both nd and multilabel into acct. Maybe ml thinks theres nothing there but nd thinks there is...
    if np.equal(thresholded_multilabel,0).all():  #all labels 0 - nothing found
        logging.debug('no items found')
        return #

    graylevel_nd_output = get_all_category_graylevels(url_or_np_array)
    pixlevel_categorical_output = graylevel_nd_output.argmax(axis=2) #the returned mask is HxWxC so take max along C
#    item_masks =  nfc.pd(image, get_all_graylevels=True)
    print('shape of pixlevel categorical output:'+str(pixlevel_categorical_output.shape))

    count_values(pixlevel_categorical_output,labels=constants.ultimate_21)
    first_time_thru = True  #hack to dtermine image size coming back from neurodoll

    final_mask = np.zeros([224,224])
    final_mask = np.zeros(pixlevel_categorical_output.shape[:])
    print('final_mask shape '+str(final_mask.shape))

    #the grabcut results dont seem too hot so i am moving to a 'nadav style' from-nd-and-ml-to-results system
    #namely : for top , decide if its a top or dress or jacket
    # for bottom, decide if dress/pants/skirt
    #decide on one bottom
 #   for i in range(len(thresholded_multilabel)):
 #       if multilabel_labels[i] in ['dress', 'jeans','shorts','pants','skirt','suit','overalls'] #missing from list is various swimwear which arent getting returned from nd now anyway


##################################
#Make some conclusions nadav style
##################################
    #1. decide on whole body item (dress, suit, overall) vs. non-whole body items.
    #2. if whole body, donate non-whole-body pixels to whole body
    #3. else, take max one upper cover , donate losers to winner
    #4. take at least one upper under, donate losers to winner
    #5. take at least one lower cover, donate losers to winner
    #6. take max one lower under
    #upper_cover: jacket, coat, blazer etc
    #upper under: shirt, top, blouse etc
    #lower cover: skirt, pants, shorts
    #lower under: tights, leggings

    whole_body_indexlist = [multilabel_labels.index(s) for s in  ['dress', 'suit','overalls']] #swimsuits could be added here
    upper_cover_indexlist = [multilabel_labels.index(s) for s in  ['cardigan', 'coat','jacket','sweater','sweatshirt']]
    upper_under_indexlist = [multilabel_labels.index(s) for s in  ['top']]
    lower_cover_indexlist = [multilabel_labels.index(s) for s in  ['jeans','pants','shorts','skirt']]
    lower_under_indexlist = [multilabel_labels.index(s) for s in  ['stocking']]

    print('wholebody indices:'+str(whole_body_indexlist))
    whole_body_ml_values = np.array([multilabel[i] for i in whole_body_indexlist])
    print('wholebody ml_values:'+str(whole_body_indexlist))
    thewinner = whole_body_ml_values.argmax()
    whole_body_winner_value=whole_body_ml_values[thewinner]
    whole_body_winner_index=whole_body_indexlist[thewinner]
    print('winning wholebody:'+str(thewinner)+' mlindex:'+str(whole_body_winner_index)+' value:'+str(whole_body_winner_value))
    if whole_body_winner_value < multilabel_threshold:
        print('winning wholebody is under threshold')

    print('uppercover indices:'+str(upper_cover_indexlist))
    upper_cover_ml_values = np.array([multilabel[i] for i in  upper_cover_indexlist])
    print('upper_cover ml_values:'+str(upper_cover_ml_values))
    upper_cover_winner = upper_cover_ml_values.argmax()
    upper_cover_winner_value=upper_cover_ml_values[upper_cover_winner]
    upper_cover_winner_index=upper_cover_indexlist[upper_cover_winner]
    print('winning upper_cover:'+str(upper_cover_winner)+' mlindex:'+str(upper_cover_winner_index)+' value:'+str(upper_cover_winner_value))

    print('upperunder indices:'+str(upper_under_indexlist))
    upper_under_ml_values = np.array([multilabel[i] for i in  upper_under_indexlist])
    print('upper_under ml_values:'+str(upper_under_ml_values))
    upper_under_winner = upper_under_ml_values.argmax()
    upper_under_winner_value=upper_under_ml_values[upper_under_winner]
    upper_under_winner_index=upper_under_indexlist[upper_under_winner]
    print('winning upper_under:'+str(upper_under_winner)+' mlindex:'+str(upper_under_winner_index)+' value:'+str(upper_under_winner_value))

    print('lowercover indices:'+str(lower_cover_indexlist))
    lower_cover_ml_values = np.array([multilabel[i] for i in lower_cover_indexlist])
    print('lower_cover ml_values:'+str(lower_cover_ml_values))
    lower_cover_winner = lower_cover_ml_values.argmax()
    lower_cover_winner_value=lower_cover_ml_values[lower_cover_winner]
    lower_cover_winner_index=lower_cover_indexlist[lower_cover_winner]
    print('winning lower_cover:'+str(lower_cover_winner)+' mlindex:'+str(lower_cover_winner_index)+' value:'+str(lower_cover_winner_value))

    print('lowerunder indices:'+str(lower_under_indexlist))
    lower_under_ml_values = np.array([multilabel[i] for i in  lower_under_indexlist])
    print('lower_under ml_values:'+str(lower_under_ml_values))
    lower_under_winner = lower_under_ml_values.argmax()
    lower_under_winner_value=lower_under_ml_values[lower_under_winner]
    lower_under_winner_index=lower_under_indexlist[lower_under_winner]
    print('winning lower_under:'+str(lower_under_winner)+' mlindex:'+str(lower_under_winner_index)+' value:'+str(lower_under_winner_value))

    for i in range(len(thresholded_multilabel)):
        if thresholded_multilabel[i]:
            neurodoll_index = multilabel_to_ultimate21_conversion[i]
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
    name = orig_filename[:-4]+'_mf'+str(median_factor)+'_combinedoutput.png'
    print('combined png name:'+name)
    cv2.imwrite(name,final_mask)
    nice_output = imutils.show_mask_with_labels(name,constants.ultimate_21,save_images=True,original_image=orig_filename,visual_output=test_on)
#    nice_output = imutils.show_mask_with_labels(name,constants.ultimate_21,save_images=True)

    #save graymask, this should be identical to nd except no threshold on low amt of pixels
    graymask_filename = orig_filename[:-4]+'graymask.png'
    print('graymask file:'+graymask_filename)
    cv2.imwrite(graymask_filename,pixlevel_categorical_output)
    nice_output = imutils.show_mask_with_labels(graymask_filename,constants.ultimate_21,save_images=True,original_image=orig_filename,visual_output=test_on)

    return final_mask

# Make classifier.
#classifier = caffe.Classifier(MODEL_FILE, PRETRAINED,
#                              image_dims=image_dims, mean=mean,
##                              input_scale=input_scale, raw_scale=raw_scale,
 #                             channel_swap=channel_swap)


if __name__ == "__main__":
    outmat = np.zeros([256*4,256*21],dtype=np.uint8)
    url = 'http://pinmakeuptips.com/wp-content/uploads/2015/02/1.4.jpg'
    urls = [url,
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
    ]
    urls = [urls[0]]
    test_nd_alone = False
    if test_nd_alone:
        raw_input('start test_nd_alone')
        for url in urls:
            #infer-one saves results depending on switch at end
            print('testing nd alone')
            result = infer_one(url)

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
            nd_out = get_neurodoll_output(url)
            orig_filename = '/home/jeremy/'+url.split('/')[-1]
            urllib.urlretrieve(url, orig_filename)
            name = orig_filename[:-4]+'_nd_output.png'
            cv2.imwrite(name,nd_out)
            nice_output = imutils.show_mask_with_labels(name,constants.ultimate_21,save_images=True,original_image=orig_filename)

    #get output of combine_nd_and_ml
    test_combine = True
    if test_combine:
        print('start test_combined_nd')
        for url in urls:
            analyze_graylevels(url)
#            for median_factor in [0.5,0.75,1,1.25,1.5]:
#                print('testing combined ml nd, median factor:'+str(median_factor))
#                out = combine_neurodoll_and_multilabel(url,median_factor=median_factor)
#                print('combined output:'+str(out))