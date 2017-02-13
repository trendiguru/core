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
import copy

from trendi.utils import imutils
from trendi import constants
from trendi import Utils

print('starting multilabel_from_hydra.py')
#ordered according to
# constants.binary_classifier_categories =
# ['bag', 'belt', 'cardigan','coat','dress', 'eyewear', 'footwear', 'hat','jacket','jeans',
#  'pants','shorts', 'skirt','stocking','suit','sweater','top','scarf','womens_swimwear_bikini','womens_swimwear_nonbikini',
#  'overalls','sweatshirt' , 'bracelet','necklace','earrings','watch', 'mens_swimwear']

model_and_proto = constants.hydra_caffemodel_and_proto
caffe.set_mode_gpu()
gpu = 1
caffe.set_device(gpu)
caffemodel = model_and_proto[0]
deployproto = model_and_proto[1]
print('deployproto {} caffemodel {}'.format(deployproto,caffemodel))
hydra_net = caffe.Net(deployproto,caffe.TEST,weights=caffemodel)


def get_hydra_output(url_or_image_arr,out_dir='./',orig_size=(256,256),crop_size=(224,224),mean=(104.0,116.7,122.7),gpu=1):
    '''
    start net, get a bunch of results. DONE: resize to e.g. 250x250 (whatever was done in training) and crop to dims
    :param url_or_image_arr_list:
    :param prototxt:
    :param caffemodel:
    :param out_dir:
    :param dims:
    :param output_layers:
    :param mean:
    :return:
    '''
    caffe.set_mode_gpu()
    caffe.set_device(gpu)
    print('params:'+str(hydra_net.params))
    out_layers = hydra_net.outputs
    print('out layers: '+str(out_layers))
    all_outs = []
    j=0
    start_time = time.time()


    # load image, resize, crop, subtract mean, and make dims C x H x W for Caffe
    im = Utils.get_cv2_img_array(url_or_image_arr)
    if im is None:
        logging.warning('could not get image '+str(url_or_image_arr))
        return None
    if isinstance(url_or_image_arr,basestring):
        print('get_hydra_output working on:'+url_or_image_arr)
    print('img  size:'+str(im.shape))
    im = imutils.resize_keep_aspect(im,output_size=orig_size)
    im = imutils.center_crop(im,crop_size)
    in_ = np.array(im, dtype=np.float32)
    if len(in_.shape) != 3:
        print('got 1-chan image, skipping')
        return
    elif in_.shape[2] != 3:
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
    in_ -= mean
    in_ = in_.transpose((2,0,1)) #W,H,C -> C,W,H
    hydra_net.blobs['data'].reshape(1, *in_.shape)
    hydra_net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    hydra_net.forward()
    out = []
    for output_layer in out_layers:
        one_out = hydra_net.blobs[output_layer].data[0]   #not sure why the data is nested [1xN] matrix and not a flat [N] vector
        out.append(copy.copy(one_out)) #the copy is required - if you dont do it then out gets over-written with each new one_out
        logging.debug('output for {} is {}'.format(output_layer,one_out))
#        print('final till now:'+str(all_outs)+' '+str(all_outs2))
    logging.debug('all output:'+str(out))
    logging.debug('elapsed time:'+str(time.time()-start_time)+' tpi '+str((time.time()-start_time)/j))
    return out


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
        output = get_hydra_output(url)
        print('final output for {} : cat {}'.format(url,output))
    elapsed_time = time.time()-start_time
    print('time per image:{}, {} elapsed for {} images'.format(elapsed_time/len(urls),elapsed_time,len(urls)))
#    cv2.imshow('output',output)
#    cv2.waitKey(0)
#    for i in range(len(output)):
#        print('label:' + constants.web_tool_categories[i]+' value:'+str(output[i]))