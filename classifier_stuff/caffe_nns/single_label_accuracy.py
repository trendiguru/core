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
    all_params = [k for k in net.params.keys()]
    print('all params:')
    print all_params
    all_blobs = [k for k in net.blobs.keys()]
    print('all blobs:')
    print all_blobs
    print('looking for label {} and estimate {}'.format(label_layer,estimate_layer))
    confmat = np.zeros([n_classes,n_classes])
    for t in range(n_tests):
        net.forward()
        gts = net.blobs[label_layer].data
#        ests = net.blobs['score'].data > 0  ##why 0????  this was previously not after a sigmoid apparently
        ests = net.blobs[estimate_layer].data  #.data gets the loss
        n_classes = len(ests[0])  #get first batch element
        print('gts {} ests {} n_classes {}'.format(gts, ests,n_classes))
        if np.any(np.isnan(ests)):
            print('got nan in ests, continuing')
            continue
  #   out = net.blobs['seg-score'].data[0].argmax(axis=0)
#        print('net output:'+str(net.blobs[estimate_layer].data))
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
#            if est.shape != gt.shape:
#                print('shape mismatch')
#                continue
            gt_value = gt[0]
            max_est = np.argmax(est)
            print('gt {} gt_val {} est {} maxest {} confmat:'.format(gt,gt_value,est,max_est))
            confmat = update_confmat(gt_value,max_est,confmat)
            print(confmat)
    print('final confmat')
    print(confmat)
    return confmat

def single_label_acc(caffemodel,testproto,net=None,label_layer='label',estimate_layer='loss',n_tests=100,gpu=0,classlabels = constants.web_tool_categories_v2):
    #TODO dont use solver to get inferences , no need for solver for that
    #DONE
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
        dir = 'single_label_'+netname+'_'+model_base.replace('.caffemodel','')
    else:
        dir = 'single_label_'+protoname+'_'+model_base.replace('.caffemodel','')
    dir = dir.replace('"','')  #remove quotes
    dir = dir.replace(' ','')  #remove spaces
    dir = dir.replace('\n','')  #remove newline
    dir = dir.replace('\r','')  #remove return
    htmlname=dir+'.html'
    print('htmlname : '+str(htmlname))
#    Utils.ensure_dir(dir)
    confmat = check_accuracy(net,n_classes, n_tests=n_tests,label_layer=label_layer,estimate_layer=estimate_layer)
    open_html(htmlname,testproto,caffemodel,confmat,netname,classlabels=classlabels)
    for i in range(n_classes):
        p,r,a = precision_recall_accuracy(confmat,i)
        write_confmat_to_html(htmlname,confmat,classlabels=classlabels)
        write_pra_to_html(htmlname,p,r,a,i,classlabels[i])
    close_html(htmlname)

def precision_recall_accuracy(confmat,class_to_analyze):
    npconfmat = np.array(confmat)
    tp = npconfmat[class_to_analyze,class_to_analyze]
    fn = npconfmat[class_to_analyze,:] - tp
    fp = npconfmat[:,class_to_analyze] - tp
    tn = npconfmat[:,:] - tp -fn - fp
    print('confmat:'+str(confmat))
    print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))
    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    accuracy = float(tp+tn)/(tp+fp+tn+fn)
    print('prec {} recall {} acc {}'.format(precision,recall,accuracy))

def normalized_confmat(confmat):
    npconfmat = np.array(confmat)
    normalized_cm = np.zeros_like(npconfmat)
    for i in range(npconfmat.shape[0]):
        n = float(np.sum(npconfmat[i,:]))
        normalized_cm[i,:] = npconfmat[i,:]/n
    print('normalized confmat {}'.format(normalized_cm))
    return normalized_cm

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

def open_html(htmlname,proto,caffemodel,netname=None,classlabels=constants.web_tool_categories_v2):
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
        g.write('<th align="left">')
        g.write('confmat')
        g.write('</th>\n')
        for i in range(len(classlabels)):
            g.write('<th align="left">')
            g.write('pred.'+classlabels[i])
            g.write('</th>\n')
        g.write('</tr>\n')
        g.close()


def write_confmat_to_html(htmlname,confmat,classlabels):
    with open(htmlname,'a') as g:
        confmat_rows = confmat.shape[0]
#        if confmat_rows != len(classlabels):
#            print('WARNING length of labels is not same as size of confmat')
        g.write('<table><br>')
        for i in range(confmat_rows):
            g.write('<tr>\n')
            g.write('<td>')
            g.write(str(classlabels[i]))
            g.write('</td>\n')
            for j in range(confmat_rows):
                g.write('<td>')
                g.write(str(confmat[i][j]))
                g.write('</td>\n')
            g.write('</tr>\n')
        g.write('</table><br>')

        ncm = normalized_confmat(confmat)
        g.write('normalized')
        g.write('<table><br>')
        for i in range(confmat_rows):
            g.write('<tr>\n')
            g.write('<td>')
            g.write(str(classlabels[i])
            g.write('</td>\n')
            for j in range(confmat_rows):
                g.write('<td>')
                g.write(str(confmat[i][j]))
                g.write('</td>\n')
            g.write('</tr>\n')

        g.write('</table><br>')
        g.close()

def write_pra_to_html(htmlname,precision,recall,accuracy,classindex,classlabel):
    with open(htmlname,'a') as g:
        g.write('<br>\n')
        g.write('class {} label {}'.format(classindex,classlabel))
        g.write('<br>\n')
        g.write('precision '+round(precision,3))
        g.write('<br>\n')
        g.write('recall '+round(recall,3))
        g.write('<br>\n')
        g.write('accuracy '+round(accuracy,3))
        g.close()

def close_html(htmlname):
    with open(htmlname,'a') as g:
        g.write('</html>')
        g.close()

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
    parser.add_argument('--output_layer_name', help='output layer name',default='estimate')
    parser.add_argument('--label_layer_name', help='label layer name',default='label')
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
    single_label_acc(args.caffemodel,args.testproto,label_layer='label', estimate_layer=outlayer,n_tests=n_tests,gpu=gpu,classlabels=classlabels)
#def single_label_acc(caffemodel,          testproto,net=None,label_layer='label',estimate_layer='loss',n_tests=100,gpu=0,classlabels = constants.web_tool_categories_v2):




