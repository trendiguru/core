__author__ = 'jeremy'

import sys
import os
import datetime
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import urllib2,urllib
from copy import copy
import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
import matplotlib.pyplot as plt

from caffe import layers as L, params as P # Shortcuts to define the net prototxt.
import cv2
import argparse

from trendi import constants
from trendi.utils import imutils
from trendi import Utils
from trendi.classifier_stuff.caffe_nns import multilabel_accuracy

import math

def update_confmat(gt,est,confmat):
#    print('gt {} \nest {} sizes tp {} tn {} fp {} fn {} '.format(gt,est,tp.shape,tn.shape,fp.shape,fn.shape))
    pantsindex = constants.web_tool_categories.index('pants')
    jeansindex = constants.web_tool_categories.index('jeans')
    confmat[gt][est]+=1
    return confmat

def test_confmat():
    gt=[5,4,1,0]
    ests=[5,3,1,10]
    confmat = np.zeros([11,11])
    for e in ests:
        confmat = update_confmat(gt,e,tp,tn,fp,fn)
    print('confmat: {}'.format(confmat))


def check_accuracy(net,n_classes,n_tests=200,label_layer='label',estimate_layer='score'):
    confmat = np.zeros([n_classes,n_classes])
    for t in range(n_tests):
        net.forward()
        gts = net.blobs[label_layer].data
#        ests = net.blobs['score'].data > 0  ##why 0????  this was previously not after a sigmoid apparently
        ests = net.blobs[estimate_layer].data
        n_classes = len(ests[0])
        print('gts {} score {} ests {} n_classes'.format(gts,net.blobs['score'], ests,n_classes))
  #   out = net.blobs['seg-score'].data[0].argmax(axis=0)
        print('net output:'+str(net.blobs[estimate_layer].data))
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            if est.shape != gt.shape:
                print('shape mismatch')
                continue
            confmat = update_confmat(gt,est,confmat)
            print('confmat:')
            print(confmat)
    return confmat

def single_label_acc(caffemodel,testproto,net=None,outlayer='label',n_tests=100,gpu=0,classlabels = constants.web_tool_categories_v2):
    #TODO dont use solver to get inferences , no need for solver for that
    print('checking accuracy of net {} using proto {}'.format(caffemodel,testproto))
    n_classes = len(classlabels)
    print('nclasses {} labels {}'.format(n_classes,classlabels))
    if net is None:
        caffe.set_mode_gpu()
        caffe.set_device(gpu)
        net = caffe.Net(testproto,caffemodel, caffe.TEST)  #apparently this is how test is chosen instead of train if you use a proto that contains both test and train
    model_base = caffemodel.split('/')[-1]
    protoname = testproto.replace('.prototxt','')
    netname = multilabel_accuracy.get_netname(testproto)
    if netname:
        dir = 'single_label_results-'+netname+'_'+model_base.replace('.caffemodel','')
    else:
        dir = 'single_label_results-'+protoname+'_'+model_base.replace('.caffemodel','')
    dir = dir.replace('"','')  #remove quotes
    dir = dir.replace(' ','')  #remove spaces
    dir = dir.replace('\n','')  #remove newline
    dir = dir.replace('\r','')  #remove return
    htmlname=dir+'.html'
    print('dir to save stuff in : '+str(dir))
    Utils.ensure_dir(dir)
    confmat = check_accuracy(net,n_classes, n_tests=n_tests,outlayer=outlayer,n_classes=n_classes)
    write_html(htmlname,testproto,caffemodel,confmat,netname,classlabels=classlabels)


def multilabel_infer_one(url):
    image_mean = np.array([104.0,117.0,123.0])
    input_scale = None
    channel_swap = [2, 1, 0]
    raw_scale = 255.0
    print('loading caffemodel for neurodoll (single class layers)')

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
    in_ = imutils.resize_keep_aspect(image,output_size=required_image_size,output_file=None)   #
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

def get_single_label_output(url_or_np_array,required_image_size=(227,227),output_layer_name='prob'):


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
    out = multilabel_net.blobs[output_layer_name].data[0] #for the nth class layer #siggy is after sigmoid
    min = np.min(out)
    max = np.max(out)
    print('out  {}'.format(out))

def write_html(htmlname,proto,caffemodel,confmat,netname=None,classlabels=constants.web_tool_categories_v2):
    model_base = caffemodel.replace('.caffemodel','')
    with open(htmlname,'a') as g:
        g.write('<!DOCTYPE html>')
        g.write('<html>')
        g.write('<head>')
        g.write('<title>')
        dt=datetime.datetime.today()
        g.write(model_base+' '+dt.isoformat())
        g.write('</title>')
        g.write('</head>')
        g.write('<body>')
        g.write('<br>\n')
        g.write('single-label results generated on '+ str(dt.isoformat())+'/n<br>/n')
        g.write('proto:'+proto+'\n<br>')
        g.write('model:'+caffemodel+'\n<br>')
        if netname is not None:
            g.write('netname:'+netname+'\n<br>')
        g.write('<table><br>')
        g.write('<tr>\n')
        for i in range(len(classlabels)):
            g.write('<th align="left">')
            g.write(classlabels[i])
            g.write('</th>\n')
        g.write('</tr>\n')

        confmat_rows = confmat.shape[0]
        if confmat_rows != len(classlabels):
            print('WARNING length of labels is not same as size of confmat')
        for i in range(confmat_rows):
            g.write('<tr>\n')
            for j in range(confmat_rows):
                g.write('<td>')
                g.write(confmat[i][j])
                g.write('</td>\n')
            g.write('</tr>\n')

        g.write('</table><br>')
        plotfilename = 'multilabel_results'+model_base+'.png'

        g.write('<a href=\"'+plotfilename+'\">plot<img src = \"'+plotfilename+'\" style=\"width:300px\"></a>')
        g.write('</html>')

def write_textfile(txtname,proto,caffemodel,confmat,netname=None,classlabels=constants.web_tool_categories_v2):
   with open(txtname,'a') as f:
        f.write('solver:'+proto+'\n')
        f.write('model:'+caffemodel+'\n')
        f.write('net:'+netname+'\n')
        f.write('categories: '+str(classlabels)+ '\n')
        f.write('confmat:\n')
        f.write(str(confmat)+'\n')
        f.close()

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='singe label accuracy tester')
    parser.add_argument('--testproto',  help='test prototxt')
    parser.add_argument('--caffemodel', help='caffemodel')
    parser.add_argument('--gpu', help='gpu #',default=0)
    parser.add_argument('--output_layer_name', help='output layer name',default='prob')
    parser.add_argument('--n_tests', help='number of examples to test',default=1000)
    parser.add_argument('--n_classes', help='number of classes',default=21)
    parser.add_argument('--classlabels', help='class labels (specify a list from trendi.constants)')

    args = parser.parse_args()
    print(args)
#    if args.gpu is not None:
    gpu = int(args.gpu)
#    if args.output_layer_name is not None:
    outlayer = args.output_layer_name
    n_tests = int(args.n_tests)
    n_classes = int(args.n_classes)
    classlabels = []
    if args.classlabels == None:
        for i in range(n_classes):
            classlabels.append('class '+str(i))
    else:
        classlabels = constants.classlabels
    print('classlabels:'+str(classlabels))
    single_label_acc(args.caffemodel,args.testproto,outlayer=outlayer,n_tests=n_tests,gpu=gpu,classlabels=classlabels)




