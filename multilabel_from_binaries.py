__author__ = 'jeremy'

import cv2
import caffe
import os
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import urllib
import time
import hashlib

from trendi.utils import imutils
from trendi import constants
from trendi import Utils

print('starting multilabel_from_binaries.py')
#ordered according to
# constants.binary_classifier_categories =
# ['bag', 'belt', 'cardigan','coat','dress', 'eyewear', 'footwear', 'hat','jacket','jeans',
#  'pants','shorts', 'skirt','stocking','suit','sweater','top','scarf','womens_swimwear_bikini','womens_swimwear_nonbikini',
#  'overalls','sweatshirt' , 'bracelet','necklace','earrings','watch', 'mens_swimwear']

this_is_instance = 1

gpu = this_is_instance-1
caffemodels = constants.binary_caffemodels
modelpath = '/home/jeremy/caffenets/binary/all'
binary_nets=[]
caffe.set_mode_gpu()
nets_per_gpu = 9
for i in range(nets_per_gpu*(this_is_instance-1),min(nets_per_gpu*(this_is_instance),len(caffemodels))):
    caffe.set_device(gpu)
    print('device '+str(gpu)+' net # '+str(i))
    deployproto = os.path.join(modelpath,'ResNet-101-deploy.prototxt')
    caffemodel = os.path.join(modelpath,caffemodels[i])
    print('deployproto {} caffemodel {}'.format(deployproto,caffemodel))
    binary_net = caffe.Net(deployproto,caffe.TEST,weights=caffemodel)
    binary_nets.append(binary_net)

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


def get_multiple_single_label_outputs(url_or_np_array):
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    all_outs = []
    for i in range(len(binary_nets)):
        out = get_single_label_output(image,binary_nets[i])
        all_outs.append(out)
    print('output from get_multiple_single_label_outputs(1):'+str(all_outs))
    return all_outs

def get_single_label_output(url_or_np_array,net, required_image_size=(224,224),resize=(250,250),analog_output=True,from_binary_net=True):
    '''
    gets the output of a single-label classifier.
    :param url_or_np_array:
    :param net:
    :param required_image_size: the size of the image the net wants (has been trained on), (WxH)
    :param resize: resize img to this dimension. if this is > required_image_size then take center crop.  pls dont make this < required_image_size
    :param analog_output: get the actual class activations not just the max
    :param binary_net: get the 2nd neuron output if this is a binary net (forget abt the first one since the neurons would be 'not item/item'
    :return:
    '''
    #the below could be replaced by a call to
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    image = Utils.get_cv2_img_array(url_or_np_array)
    if image is None:
        print('ug didnt manage to get an image...'+str(url_or_np_array))
        return
    print('multilabel working on image of shape:'+str(image.shape))

#  save image to make sure no rgb/bgr funny business
#    hash = hashlib.sha1()
#    hash.update(str(time.time()))
#    print hash.hexdigest()
#    name=hash.hexdigest()[:10]+'.jpg'
#    print('saving '+name)
#    cv2.imwrite(name,image)

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#    im = Image.open(imagename)
#    im = im.resize(required_imagesize,Image.ANTIALIAS)
#    in_ = in_.astype(float)
    if resize:
        image = imutils.resize_keep_aspect(image,output_size=resize,output_file=None)
        #print('original resized to '+str(image.shape))
    height,width,channels = image.shape
    crop_dx = width - required_image_size[0]
    crop_dy = height - required_image_size[1]
    if crop_dx != 0:
        remove_x_left = crop_dx/2
        remove_x_right = crop_dx - remove_x_left
        image = image[:,remove_x_left:width-remove_x_right,:]  #crop center x
        #print('removing {} from left and {} from right leaving {}'.format(remove_x_left,remove_x_right,image.shape))
    if crop_dy !=0:
        remove_y_top = crop_dy/2
        remove_y_bottom = crop_dy - remove_y_top
        image = image[remove_y_top:width-remove_y_bottom,:,:]  #crop center y
        #print('removing {} from top and {} from bottom leaving {}'.format(remove_x_left,remove_x_right,image.shape))


    image = np.array(image, dtype=np.float32)   #.astype(float)
    if len(image.shape) != 3:  #h x w x channels, will be 2 if only h x w
        print('got 1-chan image, turning into 3 channel')
        #DEBUG THIS , ORDER MAY BE WRONG [what order? what was i thinking???]
        image = np.array([image,image,image])
    elif image.shape[2] != 3:  #for rgb/bgr, some imgages have 4 chan for alpha i guess
        print('got n-chan image, skipping - shape:'+str(image.shape))
        return
#    image = image[:,:,::-1]  for doing RGB -> BGR : since this is loaded nby cv2 its unecessary
#    cv2.imshow('test',image)
    image -= np.array((104.0,116.0,122.0))
    image = image.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *image.shape)
    net.blobs['data'].data[...] = image
    # run net and take argmax for prediction
    net.forward()
#    out = net.blobs['score'].data[0].argmax(axis=0) #for a parse with per-pixel max
    out = net.blobs['prob'].data[0] #for the nth class layer #siggy is after sigmoid
    the_chosen_one = out.argmax()
    min = np.min(out)
    max = np.max(out)
    print('net output:  {}  answer:class {}'.format(out,the_chosen_one))
    if analog_output:
        if from_binary_net:
            second_neuron_output = out[1]
            print('binary output:'+str(out[1]))
            return second_neuron_output
        return out
    return the_chosen_one
#   possible return out which has more info (namel the actual values which somehow relate to confidence in the answer)

if __name__ == "__main__":
    urls = ['https://s-media-cache-ak0.pinimg.com/236x/ce/64/a0/ce64a0dca7ad6d609c635432e9ae1413.jpg',  #bags
            'http://pinmakeuptips.com/wp-content/uploads/2015/02/1.4.jpg',
            'https://s-media-cache-ak0.pinimg.com/564x/9a/9d/f7/9a9df7455232035c6284ad1961816fd8.jpg',
            'http://2.bp.blogspot.com/-VmiQlqdliyE/U9nyto2L1II/AAAAAAAADZ8/g30i4j_YZeI/s1600/310714+awayfromblue+kameleon+shades+chambray+shift+dress+converse+louis+vuitton+neverfull+mbmj+scarf.png',
            'https://s-media-cache-ak0.pinimg.com/236x/1b/31/fd/1b31fd2182f0243ebc97ca115f04f131.jpg',
            'http://healthsupporters.com/wp-content/uploads/2013/10/belt_2689094b.jpg' ,
            'http://static1.businessinsider.com/image/53c96c90ecad04602086591e-480/man-fashion-jacket-fall-layers-belt.jpg', #belts
            'http://gunbelts.com/media/wysiwyg/best-gun-belt-width.jpg',
            'https://i.ytimg.com/vi/5-jWNWUQdFQ/maxresdefault.jpg'
            ]

    start_time=time.time()
    for url in urls:
#        output = get_single_label_output(url,binary_nets[0])
        output = get_multiple_single_label_outputs(url)
        print('final output for {} : cat {}'.format(url,output))
    elapsed_time = time.time()-start_time
    print('time per image:{}, {} elapsed for {} images'.format(elapsed_time/len(urls),elapsed_time,len(urls)))
#    cv2.imshow('output',output)
#    cv2.waitKey(0)
#    for i in range(len(output)):
#        print('label:' + constants.web_tool_categories[i]+' value:'+str(output[i]))