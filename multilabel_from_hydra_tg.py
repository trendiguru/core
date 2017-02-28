__author__ = 'jeremy'

import cv2
import caffe
import os
import logging
logging.basicConfig(level=logging.DEBUG)
import random
import string
import json
import numpy as np
import urllib
import time
import hashlib
import copy

from trendi.utils import imutils
from trendi import constants
from trendi import Utils

print('loading net for multilabel_from_hydra_hls.py')
#ordered according to
# constants.binary_classifier_categories =
# ['bag', 'belt', 'cardigan','coat','dress', 'eyewear', 'footwear', 'hat','jacket','jeans',
#  'pants','shorts', 'skirt','stocking','suit','sweater','top','scarf','womens_swimwear_bikini','womens_swimwear_nonbikini',
#  'overalls','sweatshirt' , 'bracelet','necklace','earrings','watch', 'mens_swimwear']

model_and_proto = constants.hydra_tg_caffemodel_and_proto
caffe.set_mode_gpu()
gpu = 1
caffe.set_device(gpu)
caffemodel = model_and_proto[0]
deployproto = model_and_proto[1]
print('deployproto {} caffemodel {}'.format(deployproto,caffemodel))
hydra_net = caffe.Net(deployproto,caffe.TEST,weights=caffemodel)


def get_hydra_output(url_or_image_arr,out_dir='./',orig_size=(256,256),crop_size=(224,224),mean=(104.0,116.7,122.7),
                     gpu=1,save_data=True,save_path='/data/jeremy/caffenets/hydra/production/saves',detection_thresholds=constants.hydra_tg_thresholds):
    '''
    start net, get a bunch of results. DONE: resize to e.g. 250x250 (whatever was done in training) and crop to dims
    :param url_or_image_arr_list:#
    :param prototxt:
    :param caffemodel:
    :param out_dir:
    :param dims:
    :param output_layers:
    :param mean:
    :return:
    '''
    start_time = time.time()
    caffe.set_mode_gpu()
    caffe.set_device(gpu)
#    print('params:'+str(hydra_net.params))
    out_layers = hydra_net.outputs
    out_layers = put_in_numeric_not_alphabetic_order(out_layers)
#    print('out layers: '+str(out_layers))
    j=0
    output_names = constants.hydra_tg_heads

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
    out = {}
    i = 0
    for output_layer in out_layers:
        one_out = hydra_net.blobs[output_layer].data[0]   #not sure why the data is nested [1xN] matrix and not a flat [N] vector
        second_neuron = copy.copy(one_out[1])#the copy is required - if you dont do it then out gets over-written with each new one_out
        second_neuron = round(float(second_neuron),3)
  #      print('type:'+str(type(second_neuron)))
        name = output_names[i]
        if second_neuron > detection_thresholds[i]:
            out[name]=second_neuron
            print('{} is past threshold {} for category {} {}'.format(second_neuron,detection_thresholds[i],i,name))
        logging.debug('output for {} {} is {}'.format(output_layer,name,second_neuron))
#        print('final till now:'+str(all_outs)+' '+str(all_outs2))
        i=i+1
    logging.debug('all output:'+str(out))
    logging.debug('elapsed time:'+str(time.time()-start_time))

    if save_data:
        if isinstance(url_or_image_arr,basestring):
            filename=url_or_image_arr.replace('https://','').replace('http://','').replace('/','_')
            url = url_or_image_arr
        else:
            n_chars=6
            filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n_chars))+'.jpg'
            url = 'not_from_url'
        Utils.ensure_dir(save_path)
        imgname=os.path.join(save_path,filename)
        if imgname[:-4] != '.jpg':
            imgname = imgname + '.jpg'
        cv2.imwrite(imgname,im)
    #    out['imgname']=filename
        out['url']=url
        textfile = os.path.join(save_path,'output.txt')
        with open(textfile,'a') as fp:
            json.dump(out,fp,indent=4)
            fp.close()
        print('wrote image to {} and output text to {}'.format(imgname,textfile))

    return out

def put_in_numeric_not_alphabetic_order(out_layers):
    new_list = [0 for l in out_layers]
    for i in range(len(out_layers)):
  #      print('layer {} '.format(out_layers[i]))
        if out_layers[i] == 'estimate': #0th layer just called 'estimate' this should really be changed in generic_hydra
            new_list[0] = out_layers[i]
            continue
        if not '__' in out_layers[i] :
            logging.warning('didnt find telltale __ in layer name , abort')
            return None
        n = int(out_layers[i].split('__')[1])
#        print('layer {} n {}'.format(out_layers[i],n))
        new_list[n] = out_layers[i] #n-1 because layers start at 1 , change here needed if layer numbering redone to start at 0 in generic_hydra
 #   print(new_list)
    return new_list

if __name__ == "__main__":
    urls = ['https://secure.polyvoreimg.com/cgi/img-thing/size/l/tid/47760829.jpg' ,#skirt
            'https://ae01.alicdn.com/kf/HTB1cZJuIFXXXXa5XXXXq6xXFXXX1/women-high-waist-sexy-font-b-skirts-b-font-lady-font-b-long-b-font-font.jpg',
            'http://fashiongum.com/wp-content/uploads/2015/01/Sequin-Skirts-Best-Street-Style-Looks-5.jpg',
            'https://s-media-cache-ak0.pinimg.com/736x/c1/be/9e/c1be9e3ffa1d7446e9c23a306d1b2bf9.jpg',#dress
            'http://ind5.ccio.co/ud/oM/4K/166351779955768499Z9HaV5O7c.jpg',
            'http://i.styleoholic.com/15-Cool-Dress-And-Boots-Combinations-For-Fall10.jpg',
             'https://s-media-cache-ak0.pinimg.com/236x/ce/64/a0/ce64a0dca7ad6d609c635432e9ae1413.jpg',  #bags
            'http://pinmakeuptips.com/wp-content/uploads/2015/02/1.4.jpg',
            'https://s-media-cache-ak0.pinimg.com/564x/9a/9d/f7/9a9df7455232035c6284ad1961816fd8.jpg',
            'http://2.bp.blogspot.com/-VmiQlqdliyE/U9nyto2L1II/AAAAAAAADZ8/g30i4j_YZeI/s1600/310714+awayfromblue+kameleon+shades+chambray+shift+dress+converse+louis+vuitton+neverfull+mbmj+scarf.png',
            'https://s-media-cache-ak0.pinimg.com/236x/1b/31/fd/1b31fd2182f0243ebc97ca115f04f131.jpg',
            'http://healthsupporters.com/wp-content/uploads/2013/10/belt_2689094b.jpg' , #belt
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