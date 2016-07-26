__author__ = 'jeremy'

import cv2
import caffe
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import urllib
from trendi.utils import imutils

from trendi import constants

###########LOAD MULTILABELLER
#these are on braini2
#caffemodel = '/home/jeremy/caffenets/production/multilabel_resnet50_sgd_iter_110000.caffemodel'
#deployproto = '/home/jeremy/core/classifier_stuff/caffe_nns/protos/multilabel/resnet/ResNet-50-deploy.prototxt'
#caffemodel = '/home/jeremy/caffenets/production/multilabel_resnet50_sgd_iter_120000.caffemodel'
#solverproto = '/home/jeremy/caffenets/production/ResNet-50-test.prototxt'
#    caffemodel =  '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/snapshot/train_iter_340000.caffemodel'
#    deployproto = '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/deploy.prototxt'

#best as of 260716, see http://extremeli.trendi.guru/demo/results/ for updates
print('starting multilabel.py')
solverproto = '/home/jeremy/caffenets/multilabel/deep-residual-networks/prototxt/ResNet-101-test.prototxt'
deployproto = '/home/jeremy/caffenets/multilabel/deep-residual-networks/prototxt/ResNet-101-deploy.prototxt'
caffemodel = '/home/jeremy/caffenets/production/multilabel_resnet101_sgd_iter_120000.caffemodel'


print('set_mode_gpu, device 1')
caffe.set_mode_gpu()
caffe.set_device(1)
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
    out = multilabel_net.blobs['prob'].data[0] #for the nth class layer #siggy is after sigmoid
    min = np.min(out)
    max = np.max(out)
    print('multilabel:  {}'.format(out))
    return out

if __name__ == "__main__":
    url = 'http://diamondfilms.com.au/wp-content/uploads/2014/08/Fashion-Photography-Sydney-1.jpg'
    url = 'http://pinmakeuptips.com/wp-content/uploads/2015/02/1.4.jpg'

    output = get_multilabel_output(url)
    print('output:'+str(output))
    for i in range(len(output)):
        print('label:' + constants.web_tool_categories[i]+' value:'+str(output[i]))