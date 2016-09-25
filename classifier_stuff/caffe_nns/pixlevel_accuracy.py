__author__ = 'jeremy' #ripped from shelhamer pixlevel iou code at caffe home

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

import math
from trendi.classifier_stuff.caffe_nns import jrinfer
from trendi.classifier_stuff.caffe_nns import multilabel_accuracy

def open_html(htmlname,model_base,solverproto,classes,results_dict):
    netname = multilabel_accuracy.get_netname(solverproto)
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
#        g.write('categories: '+str(constants.web_tool_categories)+'<br>'+'\n')
        g.write('<br>\n')
        g.write('pixlevel results generated on '+ str(dt.isoformat()))
        g.write('<br>\n')
        g.write('solver:'+solverproto+'\n<br>')
        g.write('model:'+model_base+'\n'+'<br>')
        g.write('netname:'+netname+'\n<br>')
        g.write('iter:'+str(results_dict['iter'])+' loss:'+str(results_dict['loss'])+'\n<br>')
        g.write('overall acc:'+str(results_dict['overall_acc'])+' mean acc:'+str(results_dict['mean_acc'])+
                ' fwavac:'+str(results_dict['fwavacc'])+'\n<br>')
        g.write('mean iou:'+str(results_dict['mean_iou'])+'\n<br>')

        g.write('<table style=\"width:100%\">\n')
        g.write('<tr>\n')
        g.write('<th align="left">')
        g.write('metric')
        g.write('</th>\n')
        g.write('<th align="left">')
        g.write('fw avg.')
        g.write('</th>\n')
        for i in range(len(classes)):
            g.write('<th align="left">')
            g.write(classes[i])
            g.write('</th>\n')
        g.write('</tr>\n')

def close_html(htmlname):
    with open(htmlname,'a') as g:
        g.write('</table><br>')
        plotfilename = 'imagename.png'
        g.write('<a href=\"'+plotfilename+'\">plot<img src = \"'+plotfilename+'\" style=\"width:300px\"></a>')
        g.write('</body>')
        g.write('</html>')

def write_html(htmlname,results_dict):
#    results_dict = {'iter':iter,'loss':loss,'class_accuracy':acc.tolist(),'overall_acc':overall_acc.tolist(),'mean_acc':mean_acc.tolist(),'class_iou':iu.tolist(),'mean_iou':mean_iou.tolist(),'fwavacc':fwavacc.tolist()}
    with open(htmlname,'a') as g:
        #write class accuracies
        g.write('<tr>\n')
        g.write('<td>')
        g.write('accuracy')
        g.write('</td>\n')
        g.write('<td>')
        g.write(str(round(results_dict['fwavacc'],3)))
        g.write('</td>\n')
        for i in range(len(results_dict['class_accuracy'])):
            g.write('<td>')
            class_accuracy = results_dict['class_accuracy'][i]
            g.write(str(round(class_accuracy,3)))
            g.write('</td>\n')
        g.write('</tr>\n<br>\n')

        #write class iou
        g.write('<tr>\n')
        g.write('<td>')
        g.write('iou')
        g.write('</td>\n')
        g.write('<td>')
        g.write(str(round(results_dict['mean_iou'],3)))  #the mean iou might not be same as fwiou which is what should go here
        g.write('</td>\n')
        for i in range(len(results_dict['class_iou'])):
            g.write('<td>')
            class_iou = results_dict['class_iou'][i]
            g.write(str(round(class_iou,3)))
            g.write('</td>\n')
        g.write('</tr>\n<br>\n')


def write_textfile(caffemodel, solverproto, threshold,model_base,dir=None,classes=None):
    if dir is None:
        protoname = solverproto.replace('.prototxt','')
        dir = 'multilabel_results-'+protoname+'_'+model_base.replace('.caffemodel','')
    Utils.ensure_dir(dir)
    fname = os.path.join(dir,model_base+'results.txt')
    with open(fname,'a') as f:
        f.write(model_base+' threshold = '+str(threshold)+'\n')
        f.write('solver:'+solverproto+'\n')
        f.write('model:'+caffemodel+'\n')
        f.write('categories: '+str(classes)+ '\n')
        f.close()

def do_pixlevel_accuracy(caffemodel,n_tests,layer,classes=constants.ultimate_21,testproto=None,solverproto=None, iter=0, savepics=True):
#to do accuracy we prob dont need to load solver
    caffemodel_base = os.path.basename(caffemodel)
    dir = 'pixlevel_results-'+caffemodel_base.replace('.caffemodel','')
    Utils.ensure_dir(dir)
    if savepics:
        picsdir = os.path.join(dir,'pics')
        Utils.ensure_dir(picsdir)
    htmlname = os.path.join(dir,dir+'.html')
    detailed_outputname = htmlname[:-5]+'.txt'
    print('saving net of {} {} to dir {} and file {}'.format(caffemodel,solverproto,htmlname,detailed_outputname))

    val = range(n_tests)
    if args.gpu:
        caffe.set_mode_gpu()
        caffe.set_device(int(args.gpu))
    else:
        caffe.set_mode_cpu()

    if(solverproto is not None): #do this the old way with sgdsolver
        solver = caffe.SGDSolver(solverproto)
        solver.net.copy_from(caffemodel)
        print('using net defined by {} and {} '.format(solverproto,caffemodel))
        answer_dict = jrinfer.seg_tests(solver, False, val, layer=layer,outfilename=detailed_outputname,savepics=True)

    elif(testproto is not None):  #try using net without sgdsolver
        net = caffe.Net(testproto,caffemodel, caffe.TEST)
        answer_dict = jrinfer.do_seg_tests(net, iter, False, val, layer=layer, gt='label',outfilename=detailed_outputname,savepics=True)



  #   in_ = np.array(im, dtype=np.float32)
  #   net.blobs['data'].reshape(1, *in_.shape)
  #   net.blobs['data'].data[...] = in_
  #   # run net and take argmax for prediction
  #   net.forward()
  #   out = net.blobs['seg-score'].data[0].argmax(axis=0)

    open_html(htmlname,caffemodel,solverproto,classes,answer_dict)
    write_html(htmlname,answer_dict)
    close_html(htmlname)

if __name__ =="__main__":

    default_solverproto = '/home/jeremy/caffenets/multilabel/deep-residual-networks/prototxt/ResNet-101-test.prototxt'
    default_testproto = '/home/jeremy/caffenets/multilabel/deep-residual-networks/prototxt/ResNet-101-test.prototxt'
    default_caffemodel = '/home/jeremy/caffenets/production/multilabel_resnet101_sgd_iter_120000.caffemodel'

    parser = argparse.ArgumentParser(description='multilabel accuracy tester')
    parser.add_argument('--solverproto',  help='solver prototxt',default=None)
    parser.add_argument('--testproto',  help='val prototxt',default=None)
    parser.add_argument('--caffemodel', help='caffmodel',default = default_caffemodel)
    parser.add_argument('--gpu', help='gpu #',default=0)
    parser.add_argument('--output_layer_name', help='output layer name',default='score')
    parser.add_argument('--n_tests', help='number of examples to test',default=200)
    parser.add_argument('--classes', help='class labels',default=constants.ultimate_21)
    parser.add_argument('--iter', help='iter',default=0)
    parser.add_argument('--savepics', help='iter',default=True)

    args = parser.parse_args()
    print(args)
    gpu = int(args.gpu)
    outlayer = args.output_layer_name
    n_tests = int(args.n_tests)
    caffe.set_mode_gpu()
    caffe.set_device(gpu)
    print('using net defined by valproto {} caffmodel  {} solverproto {}'.format(args.testproto,args.caffemodel,args.solverproto))
    do_pixlevel_accuracy(args.caffemodel,n_tests,args.output_layer_name,args.classes,solverproto = args.solverproto, testproto=args.testproto,iter=int(args.iter),savepics=args.savepics)





