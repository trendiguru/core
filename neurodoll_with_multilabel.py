__author__ = 'jeremy'

#REQUIREMENTS FOR THIS TO RUN
#These files have to exist
#modelpath = '/home/jeremy/caffenets/production/ResNet-101-test.prototxt'
#solverproto = '/home/jeremy/caffenets/production/ResNet-101-test.prototxt')
#deployproto = '/home/jeremy/caffenets/production/ResNet-101-deploy.prototxt')
#caffemodel = '/home/jeremy/caffenets/production/multilabel_resnet101_sgd_iter_200000.caffemodel')  #10Aug2016


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

from trendi.paperdoll import binary_multilabel_falcon_client as bmfc
from trendi.paperdoll import binary_multilabel_falcon_client2 as bmfc2
from trendi.paperdoll import binary_multilabel_falcon_client3 as bmfc3
from trendi import Utils
from trendi import background_removal, Utils, constants
from trendi.utils import imutils
from trendi.paperdoll import neurodoll_falcon_client as nfc


#best multilabel as of 260716, see http://extremeli.trendi.guru/demo/results/ for updates
print('starting (importing) nd w multilabel.py')
multilabel_from_binaries = True

if not multilabel_from_binaries: #dont need this if answers are coming from multilabel_from_binaries
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
    caffe.set_device(0)
    multilabel_net = caffe.Net(deployproto,caffemodel, caffe.TEST)
multilabel_required_image_size = (224,224)

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

def get_multilabel_output(url_or_np_array,required_image_size=(224,224)):
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
    print('grabcut working on image of shape:'+str(image.shape))

        #def neurodoll(image, category_idx):
    dic = nfc.pd(image, category_index=category_index)
    if not dic['success']:
        logging.debug('nfc pd not a success')
        return False, []
    neuro_mask = dic['mask']

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

def combine_neurodoll_and_multilabel(url_or_np_array,multilabel_threshold=0.7,median_factor=1.6,multilabel_to_ultimate21_conversion=constants.web_tool_categories_v1_to_ultimate_21,multilabel_labels=constants.web_tool_categories):
    '''
    try product of multilabel and nd output and taking argmax
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
    print('incoming label:'+str(multilabel))
    print('thresholded label:'+str(thresholded_multilabel))
    print('multilabel to u21 conversion:'+str(multilabel_to_ultimate21_conversion))
    print('multilabel labels:'+str(multilabel_labels))

    if np.equal(thresholded_multilabel,0).all():  #all labels 0 - nothing found
        logging.debug('no items found')
        return #

# hack to combine pants and jeans for better recall
#    pantsindex = constants.web_tool_categories.index('pants')
#    jeansindex = constants.web_tool_categories.index('jeans')
#   if i == pantsindex or i == jeansindex: #
    first_time_thru = True  #hack to dtermine image size coming back from neurodoll
    final_mask = np.zeros([224,224])

    for i in range(len(thresholded_multilabel)):
        if thresholded_multilabel[i]:
            neurodoll_index = multilabel_to_ultimate21_conversion[i]
            print('index {} webtoollabel {} newindex {} neurodoll_label {} was above threshold {}'.format(
                i,multilabel_labels[i],neurodoll_index,constants.ultimate_21[neurodoll_index], multilabel_threshold))
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

#    get_nd_and_multilabel_output_using_nfc(url_or_np_array)
#    out = get_multilabel_output(url)
#    print('ml output:'+str(out))
    #get neurdoll output alone
    for url in urls:
        nd_out = get_neurodoll_output(url)
        orig_filename = '/home/jeremy/'+url.split('/')[-1]
        name = orig_filename[:-4]+'_nd_output.png'
        cv2.imwrite(name,nd_out)
        nice_output = imutils.show_mask_with_labels(name,constants.ultimate_21,save_images=True,original_image=orig_filename)

    #get output of combine_nd_and_ml
    for median_factor in [1.1,1.2,1.4,1.6,2.0]:
        for url in urls:
            out = combine_neurodoll_and_multilabel(url,median_factor=median_factor)

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