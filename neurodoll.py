#!/usr/bin/env python
__author__ = 'jeremy'

from PIL import Image
import cv2
import caffe
import logging
import copy
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

def infer_one(url_or_np_array,required_image_size=(256,256),threshold = 0.01):
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

#TODO - make the threshold per item ,e.g. small shoes are ok and should be left in
    if required_image_size is not None:
        logging.debug('resizing nd input to '+str(original_h)+'x'+str(original_w))
    #    out = [out,out,out]
        out = cv2.resize(out,(original_w,original_h))
#        out = out[:,:,0]
    image_size = out.shape[0]*out.shape[1]
    uniques = np.unique(out)

    for unique in uniques:
        pixelcount = len(out[out==unique])
        ratio = float(pixelcount)/image_size
#        logging.debug('i {} pixels {} tot {} ratio {} threshold {} ratio<thresh {}'.format(unique,pixelcount,image_size,ratio,threshold,ratio<threshold))
        if ratio < threshold:
#            logging.debug('kicking out index '+str(unique)+' with ratio '+str(ratio))
            out[out==unique] = 0  #set label with small number of pixels to 0 (background)
            pixelcount = len(out[out==unique])
            ratio = float(pixelcount)/image_size
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
    uniques = np.unique(out)
    logging.debug('final uniques:'+str(uniques))
    return out


#MODEL_FILE = "/home/jeremy/voc8_15_pixlevel_deploy.prototxt"
#SINGLE_CLASS_LAYER_DEPLOY = "/home/jeremy/voc8_15_pixlevel_deploy_with_sigmoid.prototxt"
#PRETRAINED = "/home/jeremy/voc8_15_pixlevel_iter120000.caffemodel"

protopath = os.path.join(os.path.dirname(os.path.abspath( __file__ )), 'classifier_stuff/caffe_nns/protos')
modelpath = '/home/jeremy/caffenets/production'

MODEL_FILE = os.path.join(modelpath,'voc8_15_pixlevel_deploy.prototxt')
#PRETRAINED = os.path.join(modelpath,'voc8_15_pixlevel_iter120000.caffemodel')
PRETRAINED = os.path.join(modelpath,'voc8_15_0816_iter10000_pixlevel_deploy.caffemodel')


caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(MODEL_FILE,PRETRAINED, caffe.TEST)

#required_image_size = (256, 256)
required_image_size = None
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

    for url in urls:
        result = infer_one(url,required_image_size=(256,256),threshold=0.01)
        timestamp = int(10*time.time())
        name = str(timestamp)+'.png'
        cv2.imwrite(name,result)
        labels=constants.ultimate_21
        orig_img = url_to_image(url)
        cv2.imwrite('orig.jpg',orig_img)
        imutils.show_mask_with_labels(name,labels,visual_output=False,save_images=True,original_image='orig.jpg')

#    after_nn_result = pipeline.after_nn_conclusions(result,constants.ultimate_21_dict)
#    cv2.imwrite('output_afternn.png',after_nn_result)
#   labels=constants.ultimate_21
