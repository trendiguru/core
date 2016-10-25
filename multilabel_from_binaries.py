__author__ = 'jeremy'

import cv2
import caffe
import os
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import urllib
import time

from trendi.utils import imutils
from trendi import constants

print('starting multilabel_from_binaries.py')
caffemodels = [
'res101_binary_bag_iter_56000.caffemodel',
'res101_binary_belt_iter_71000.caffemodel',
'res101_binary_bikini_iter_60000.caffemodel',
'res101_binary_bracelet_iter_586000.caffemodel',
'res101_binary_cardigan_iter_402000.caffemodel',
'res101_binary_coat_iter_665000.caffemodel',
'res101_binary_dress_iter_39000.caffemodel',
'res101_binary_earrings_r1_iter_9000.caffemodel',
'res101_binary_eyewear_iter_74000.caffemodel',
'res101_binary_footwear_iter_53000.caffemodel',
'res101_binary_hat_r1_iter_16000.caffemodel',
'res101_binary_jacket_r1_iter_25000.caffemodel',
'res101_binary_jeans_iter_15000.caffemodel',
'res101_binary_necklace_r1_iter_20000.caffemodel',
'res101_binary_overalls_iter_69000.caffemodel',
'res101_binary_pants_iter_50000.caffemodel',
'res101_binary_scarf_iter_58000.caffemodel',
'res101_binary_shorts_iter_65000.caffemodel',
'res101_binary_skirt_iter_89000.caffemodel',
'res101_binary_stocking_iter_44000.caffemodel',
'res101_binary_suit_r1_iter_22000.caffemodel',
'res101_binary_sweater_r1_iter_17000.caffemodel',
'res101_binary_sweatshirt_r1_iter_29000.caffemodel',
'res101_binary_swimwear_mens_iter_39000.caffemodel',
'res101_binary_top_iter_31000.caffemodel',
'res101_binary_watch_iter_64000.caffemodel',
'res101_binary_womens_swimwear_nonbikini_iter_35000.caffemodel',
]

modelpath = '/home/jeremy/caffenets/binary/all'
#solverproto = os.path.join(modelpath,'ResNet-101_solver.prototxt')
#trainproto = os.path.join(modelpath,'ResNet-101-train_test.prototxt')
deployproto = os.path.join(modelpath,'ResNet-101-deploy.prototxt')
caffemodel = os.path.join(modelpath,caffemodels[0])
print('deployproto {} caffemodel {}'.format(deployproto,caffemodel))
print('set_mode_gpu()')
caffe.set_mode_gpu()
gpu = 0
print('device '+str(gpu))
caffe.set_device(gpu)
binary_net = caffe.Net(deployproto,caffemodel, caffe.TEST)
binary_nets=[]
binary_nets.append(binary_net)
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


def get_multiple_single_label_outputs(url_or_np_array,binary_nets):
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    for i in range(len(binary_nets)):
        get_binary_output(image,binary_nets[i])

def get_single_label_output(url_or_np_array,net, required_image_size=(224,224),resize=(250,250)):
    '''
    gets the output of a single-label classifier.
    :param url_or_np_array:
    :param net:
    :param required_image_size: the size of the image the net wants (has been trained on), (WxH)
    :param resize: resize img to this dimension. if this is > required_image_size then take center crop.  pls dont make this < required_image_size
    :return:
    '''
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    print('multilabel working on image of shape:'+str(image.shape))
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#    im = Image.open(imagename)
#    im = im.resize(required_imagesize,Image.ANTIALIAS)
#    in_ = in_.astype(float)
    if resize:
        in_ = imutils.resize_keep_aspect(image,output_size=required_image_size,output_file=None)
        print('original resized to '+str(in_.shape))
    height,width,channels = in_.shape
    crop_dx = width - required_image_size[0]
    crop_dy = height - required_image_size[1]
    if crop_dx != 0:
        remove_x_left = crop_dx/2
        remove_x_right = crop_dx - remove_x_left
        in_ = in_[:,remove_x_left:width-remove_x_right,:]  #crop center x
        print('removing {} from left and {} from right leaving {}'.format(remove_x_left,remove_x_right,in_.shape))
    if crop_dy !=0:
        remove_y_top = crop_dy/2
        remove_y_bottom = crop_dy - remove_y_top
        in_ = in_[remove_y_top:width-remove_y_bottom,:,:]  #crop center y
        print('removing {} from top and {} from bottom leaving {}'.format(remove_x_left,remove_x_right,in_.shape))


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
    in_ -= np.array((104.0,116.0,122.0))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
#    out = net.blobs['score'].data[0].argmax(axis=0) #for a parse with per-pixel max
    out = net.blobs['prob'].data[0] #for the nth class layer #siggy is after sigmoid
    the_chosen_one = out.argmax
    min = np.min(out)
    max = np.max(out)
    print('multilabel:  {}  answer:class {}'.format(out,the_chosen_one))
    return the_chosen_one
#   possible return out which has more info (namel the actual values which somehow relate to confidence in the answer)

if __name__ == "__main__":
    urls = ['https://s-media-cache-ak0.pinimg.com/236x/ce/64/a0/ce64a0dca7ad6d609c635432e9ae1413.jpg',
            'http://pinmakeuptips.com/wp-content/uploads/2015/02/1.4.jpg',
            'https://s-media-cache-ak0.pinimg.com/564x/9a/9d/f7/9a9df7455232035c6284ad1961816fd8.jpg',
            'http://2.bp.blogspot.com/-VmiQlqdliyE/U9nyto2L1II/AAAAAAAADZ8/g30i4j_YZeI/s1600/310714+awayfromblue+kameleon+shades+chambray+shift+dress+converse+louis+vuitton+neverfull+mbmj+scarf.png',
            'https://s-media-cache-ak0.pinimg.com/236x/1b/31/fd/1b31fd2182f0243ebc97ca115f04f131.jpg']

    start_time=time.time()
    for url in urls:
        output = get_single_label_output(url,binary_nets[0])
        print('output for {} : cat {}'.format(url,output))
    elapsed_time = time.time()-start_time
    print('time per image:'+str(elapsed_time/len(urls)))
#    cv2.imshow('output',output)
#    cv2.waitKey(0)
#    for i in range(len(output)):
#        print('label:' + constants.web_tool_categories[i]+' value:'+str(output[i]))