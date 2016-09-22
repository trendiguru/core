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


def open_html(model_base,solverproto,classes,results_dict,dir=None):
#results_dict = {'iter':iter,'loss':loss,'class_accuracy':acc.tolist(),'overall_acc':overall_acc.tolist(),'mean_acc':mean_acc.tolist(),'class_iou':iu.tolist(),'mean_iou':mean_iou.tolist(),'fwavacc':fwavacc.tolist()}
    if dir is None:
        dir = 'pixlevel_results-'+model_base.replace('.caffemodel','')
    Utils.ensure_dir(dir)
    htmlname = os.path.join(dir,model_base+'results.html')
    with open(htmlname,'a') as g:
        g.write('<!DOCTYPE html>')
        g.write('<html>')
        g.write('<head>')
        g.write('<title>')
        dt=datetime.datetime.today()
        g.write(model_base+' '+dt.isoformat())
        g.write('</title>')
        g.write('solver:'+solverproto+'\n'+'<br>')
        g.write('model:'+model_base+'\n'+'<br>')
        g.write('</head>')
#        g.write('categories: '+str(constants.web_tool_categories)+'<br>'+'\n')
        g.write('<br>\n')
        g.write(model_base+' '+dt.isoformat())
        g.write('<br>\n')
        g.write('solver:'+solverproto+'\n'+'<br>')
        g.write('model:'+model_base+'\n'+'<br>')
        g.write('iter:'+str(results_dict['iter'])+' loss:'+str(results_dict['loss'])+'\n<br>')
        g.write('overall acc:'+str(results_dict['overall_acc'])+' mean acc:'+str(results_dict['mean_acc'])+
                ' fwavac:'+str(results_dict['fwavacc'])+'\n<br>')
        g.write('mean iou:'+str(results_dict['mean_iou'])+'\n<br>')

        g.write('<table style=\"width:100%\">\n')
        g.write('<tr>\n')
        g.write('<th>')
        g.write('metric')
        g.write('</th>\n')
        g.write('<th>')
        g.write('fw avg.')
        g.write('</th>\n')
        for i in range(len(classes)):
            g.write('<th>')
            g.write(classes[i])
            g.write('</th>\n')
        g.write('</tr>\n')

def close_html(model_base,solverproto,dir=None):
    if dir is None:
        protoname = solverproto.replace('.prototxt','')
        dir = 'multilabel_results-'+protoname+'_'+model_base.replace('.caffemodel','')
    Utils.ensure_dir(dir)
    htmlname = os.path.join(dir,model_base+'results.html')
    with open(htmlname,'a') as g:
        g.write('</table><br>')
        plotfilename = 'multilabel_results'+model_base+'.png'
        g.write('<a href=\"'+plotfilename+'\">plot<img src = \"'+plotfilename+'\" style=\"width:300px\"></a>')
        g.write('</html>')

def write_html(htmlname,results_dict):
#    results_dict = {'iter':iter,'loss':loss,'class_accuracy':acc.tolist(),'overall_acc':overall_acc.tolist(),'mean_acc':mean_acc.tolist(),'class_iou':iu.tolist(),'mean_iou':mean_iou.tolist(),'fwavacc':fwavacc.tolist()}
    with open(htmlname,'a') as g:
        #write class accuracies
        for i in range(len(p)):
            g.write('<tr>\n')
            g.write('<td>')
            g.write('accuracy')
            g.write('</td>\n')
            g.write('<td>')
            g.write('fw accuracy')
            g.write('</td>\n')
            for i in range(len(p)):
                g.write('<td>')
                class_accuracy = results_dict['class_accuracy'][p]
                g.write(str(round(class_accuracy,3))
                g.write('</td>\n')
            g.write('</tr>\n<br>\n')

        #write class iou
        for i in range(len(p)):
            g.write('<tr>\n')
            g.write('<td>')
            g.write('iou')
            g.write('</td>\n')
            g.write('<td>')
            g.write('fw iou')
            g.write('</td>\n')
            for i in range(len(p)):
                g.write('<td>')
                class_iou = results_dict['class_iou'][p]
                g.write(str(round(class_iou,3))
                g.write('</td>\n')
            g.write('</tr>\n<br>\n')


def write_textfile(threshold,model_base,dir=None,classes=None):
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

def get_netname(solverproto):
    with open(solverproto,'r') as fp:
        l1 = fp.readline()
        l2 = fp.readline()
   # print('line1 '+l1)
   # print('line2 '+l2)
    if 'name' in l1:
        netname = l1[5:]
        print('netname:'+netname)
    elif 'name' in l2:
        netname = l2[5:]
        print('netname:'+netname)
    else:
        netname = None
    return netname



if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='multilabel accuracy tester')
    parser.add_argument('--testproto',  help='test prototxt')
    parser.add_argument('--caffemodel', help='caffmodel')
    parser.add_argument('--gpu', help='gpu #',default=0)
    parser.add_argument('--output_layer_name', help='output layer name',default='prob')
    parser.add_argument('--n_tests', help='number of examples to test',default=1000)

    caffemodel = '/home/jeremy/caffenets/production/multilabel_resnet50_sgd_iter_120000.caffemodel'
    solverproto = '/home/jeremy/caffenets/production/ResNet-50-test.prototxt'
#    caffemodel =  '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/snapshot/train_iter_340000.caffemodel'
#    deployproto = '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/deploy.prototxt'
    solverproto = '/home/jeremy/caffenets/multilabel/deep-residual-networks/prototxt/ResNet-101-test.prototxt'
    caffemodel = '/home/jeremy/caffenets/production/multilabel_resnet101_sgd_iter_120000.caffemodel'
#    multilabel_net = caffe.Net(solverproto,caffemodel, caffe.TEST)

    args = parser.parse_args()
    print(args)
    if args.testproto is not None:
        solverproto = args.testproto
    if args.caffemodel is not None:
        caffemodel = args.caffemodel
    gpu = int(args.gpu)
    outlayer = args.output_layer_name
    n_tests = int(args.n_tests)
    caffe.set_mode_gpu()
    caffe.set_device(gpu)

    print('using net defined by {} and {} '.format(args.prototxt,args.model))
    solver = caffe.SGDSolver(args.prototxt)
    solver.net.copy_from(args.model)
    val = range(0,n_tests)
        #this just runs the train net i think, doesnt test new images
    seg_tests(solver, False, val, layer='score')

    seg_tests(solver, save_format, dataset, layer='score', gt='label',outfilename='net_output.txt'):

#    t = 0.5
#    p,r,a,tp,tn,fp,fn = check_accuracy(solverproto, caffemodel, threshold=t, num_batches=n_tests,outlayer=outlayer)




