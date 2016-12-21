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
import logging
logging.basicConfig(level=logging.DEBUG)

from caffe import layers as L, params as P # Shortcuts to define the net prototxt.
import cv2
import argparse

from trendi import constants
from trendi.utils import imutils
from trendi import Utils
from trendi.classifier_stuff.caffe_nns import multilabel_accuracy

import math
import matplotlib.pyplot as plt

def update_confmat(gt,est,confmat):
    '''

    :param gt: ground truth class (int)
    :param est:  proposed class (int(
    :param confmat: confusion matrix , rows are gt and cols are proposals
    :return:
    '''
#    print('gt {} \nest {} sizes tp {} tn {} fp {} fn {} '.format(gt,est,tp.shape,tn.shape,fp.shape,fn.shape))
#    pantsindex = constants.web_tool_categories.index('pants')
#    jeansindex = constants.web_tool_categories.index('jeans')
    confmat[gt][est]+=1
    return confmat

def test_confmat():
    gt=[5,4,1,0]
    ests=[5,3,1,10]
    confmat = np.zeros([11,11])
    for e in ests:
        confmat = update_confmat(gt,e,tp,tn,fp,fn)
    print('confmat: {}'.format(confmat))

def check_accuracy_deploy(deployproto,caffemodel,n_classes,labelfile,n_tests=200,estimate_layer='prob_0',mean=(110,110,110),scale=None,finalsize=(224,224),resize_size=(250,250),gpu=0):
    '''
    This checks accuracy using the deploy instead of the test net
    Its a more definitive test since it checks ifyou are doing the input transforms (resize, mean, scale)
    correctly
    :param net:
    :param n_classes:
    :param labelfile:
    :param n_tests:
    :param estimate_layer:
    :param mean:
    :param scale:
    :param finalsize: img will be cropped to this after resize if any
    :param resize_size: resize keeping aspect to this .
    in training i have been doing resize to 250x250 and random crop of that to 224x224
    :return:
    '''
    #check accuracy as deployed, using labelfile instead of test net
    if gpu == -1:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu)
#        net = caffe.Net(testproto,caffemodel, caffe.TEST)  #apparently this is how test is chosen instead of train if you use a proto that contains both test and train
    if caffemodel == '':
        caffemodel = None  #hack to keep things working, ideally change refs to caffemodel s.t. None is ok
        net = caffe.Net(deployproto,caffe.TEST)  #apparently this is how test is chosen instead of train if you use a proto that contains both test and train
    else:
        net = caffe.Net(deployproto,caffe.TEST,weights=caffemodel)  #apparently this is how test is chosen instead of train if you use a proto that contains both test and train

    print('checking acc using deploy {} caffemodel {} labels in {} '.format(deployproto,caffemodel,labelfile))
    print('mean {} scale {} gpu {} resize {} finalsize {}'.format(mean,scale,gpu,resize_size,finalsize))

    all_params = [k for k in net.params.keys()]
    print('all params:')
    print all_params
    all_blobs = [k for k in net.blobs.keys()]
    print('all blobs:')
    print all_blobs
    with open(labelfile,'r') as fp:
        lines = fp.readlines()
        if not lines:
            print('coundlt get lines from file '+labelfile)
    imagelist = [line.split()[0] for line in lines]
    labellist = [line.split()[1] for line in lines]
    print('1st label {} and file {}, n_classes {} nlabel {} nfile {}'.format(labellist[0],imagelist[0],n_classes,len(labellist),len(imagelist)))
    confmat = np.zeros([n_classes,n_classes])
    for t in range(n_tests):
        imgfile = imagelist[t]
        label  = labellist[t]
        img_arr = cv2.imread(imgfile)
        if img_arr is None:
            print('couldnt get '+imgfile+' ,skipping')
            continue
        if resize_size is not None:
            img_arr=imutils.resize_keep_aspect(img_arr,output_size = resize_size)
        if finalsize is not None:
            img_arr=imutils.center_crop(img_arr,finalsize)
        print('in shape '+str(img_arr.shape))
        img_arr=np.transpose(img_arr,[2,0,1])
        img_arr = np.array(img_arr, dtype=np.float32)
        #feed img into net
        net.blobs['data'].reshape(1,*img_arr.shape)
        net.blobs['data'].data[...] = img_arr
        net.forward()
        est = net.blobs[estimate_layer].data  #.data gets the loss
        print('test {}/{}: gt {} est {} '.format(t,n_tests,label, est))
        if np.any(np.isnan(est)):
            print('got nan in ests, continuing')
            continue
        best_guess = np.argmax(est)
        confmat = update_confmat(label,best_guess,confmat)
        print(confmat)
    print('final confmat')
    print(confmat)
    return confmat

def check_accuracy(net,n_classes,n_tests=200,label_layer='label',estimate_layer='score'):
    all_params = [k for k in net.params.keys()]
#    print('all params:')
#    print all_params
#    all_blobs = [k for k in net.blobs.keys()]
#    print('all blobs:')
#    print all_blobs
    print('looking for label {} and estimate {}, n_classes {}'.format(label_layer,estimate_layer,n_classes))
    confmat = np.zeros([n_classes,n_classes])
    for t in range(n_tests):
        net.forward()
        gts = net.blobs[label_layer].data
#        ests = net.blobs['score'].data > 0  ##why 0????  this was previously not after a sigmoid apparently
        ests = net.blobs[estimate_layer].data  #.data gets the loss
        n_classes = len(ests[0])  #get first batch element
        print('test {}/{}: gts {} ests {} '.format(t,n_tests,gts, ests))
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

def single_label_acc(caffemodel,testproto,net=None,label_layer='label',estimate_layer='loss',n_tests=100,gpu=0,classlabels = constants.web_tool_categories_v2,save_dir=None):
    #TODO dont use solver to get inferences , no need for solver for that
    #DONE
    print('checking accuracy of net {} using proto {}'.format(caffemodel,testproto))
    n_classes = len(classlabels)
    print('nclasses {} labels {}'.format(n_classes,classlabels))
    if net is None:
        caffe.set_mode_gpu()
        caffe.set_device(gpu)
#        net = caffe.Net(testproto,caffemodel, caffe.TEST)  #apparently this is how test is chosen instead of train if you use a proto that contains both test and train
        if caffemodel == '':
            caffemodel = None  #hack to keep things working, ideally change refs to caffemodel s.t. None is ok
            net = caffe.Net(testproto,caffe.TEST)  #apparently this is how test is chosen instead of train if you use a proto that contains both test and train
        else:
            net = caffe.Net(testproto,caffe.TEST,weights=caffemodel)  #apparently this is how test is chosen instead of train if you use a proto that contains both test and train
        #Net('train_val.prototxt', 1, weights='')
    if caffemodel is not '' and caffemodel is not None:
        model_base = caffemodel.split('/')[-1]
    else:
        model_base = 'scratch'
    protoname = testproto.replace('.prototxt','')
    netname = multilabel_accuracy.get_netname(testproto)
    if netname:
        name = 'single_label_'+netname+'_'+model_base.replace('.caffemodel','')
    else:
        name = 'single_label_'+protoname+'_'+model_base.replace('.caffemodel','')
    name = name.replace('"','')  #remove quotes
    name = name.replace(' ','')  #remove spaces
    name = name.replace('\n','')  #remove newline
    name = name.replace('\r','')  #remove return
    htmlname=name+'.html'
    if save_dir is not None:
        Utils.ensure_dir(save_dir)
        htmlname = os.path.join(save_dir,htmlname)
    print('htmlname : '+str(htmlname))
#    Utils.ensure_dir(dir)


    confmat = check_accuracy(net,n_classes, n_tests=n_tests,label_layer=label_layer,estimate_layer=estimate_layer)
    open_html(htmlname,testproto,caffemodel,netname=netname,classlabels=classlabels) #
    write_confmat_to_html(htmlname,confmat,classlabels=classlabels)
    for i in range(n_classes):
        p,r,a = precision_recall_accuracy(confmat,i)
        write_pra_to_html(htmlname,p,r,a,i,classlabels[i])
    close_html(htmlname)
    return a

def precision_recall_accuracy(confmat,class_to_analyze):
    npconfmat = np.array(confmat)
    tp = npconfmat[class_to_analyze,class_to_analyze]
    fn = np.sum(npconfmat[class_to_analyze,:]) - tp
    fp = np.sum(npconfmat[:,class_to_analyze]) - tp
    tn = np.sum(npconfmat[:,:]) - tp -fn - fp
    print('confmat:'+str(confmat))
    print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))
    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    accuracy = float(tp+tn)/(tp+fp+tn+fn)
    print('prec {} recall {} acc {}'.format(precision,recall,accuracy))
    return precision, recall, accuracy

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
    '''
    CURRENTLY INOPERATIONAL
    :param url_or_np_array:
    :param required_image_size:
    :param output_layer_name:
    :return:
    '''
    if isinstance(url_or_np_array, basestring):
        print('infer_one working on url:'+url_or_np_array)
        image = Utils.get_cv2_img_array(url_or_np_array)
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

#    open_html(htmlname,testproto,caffemodel,confmat,netname,classlabels=classlabels) #
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
        g.write('single-label results generated on '+ str(dt.isoformat())+'<br>')
        g.write('proto:'+proto+'\n<br>')
        g.write('model:'+caffemodel+'\n<br>')
        if netname is not None:
            g.write('netname:'+netname+'\n<br>')
        g.close()


def write_confmat_to_html(htmlname,confmat,classlabels):
    with open(htmlname,'a') as g:
        confmat_rows = confmat.shape[0]
#        if confmat_rows != len(classlabels):
#            print('WARNING length of labels is not same as size of confmat')
        g.write('<table><br>')
        g.write('<tr>\n')
#write confmat headings
        g.write('<th align="left">')
        g.write('confmat')
        g.write('</th>\n')
        for i in range(len(classlabels)):
            g.write('<th align="left">')
            g.write('pred.'+classlabels[i]+'|')
            g.write('</th>\n')
        g.write('</tr>\n')
        g.write('<tr>\n')
        g.write('<th align="left">')
        g.write('____')
        g.write('</th>\n')
#write confmat
        for i in range(len(classlabels)):
            g.write('<th align="left">')
            g.write('____')
            g.write('</th>\n')
        g.write('</tr>\n')
        for i in range(confmat_rows):
            g.write('<tr>\n')
            g.write('<td>')
            g.write('actual '+str(classlabels[i])+'|')
            g.write('</td>\n')
            for j in range(confmat_rows):
                g.write('<td>')
                g.write(str(confmat[i][j]))
                g.write('</td>\n')
            g.write('</tr>\n')

#write normalized confmat
        ncm = normalized_confmat(confmat)
        g.write('<tr>\n')
        g.write('<th align="left">')
        g.write('normalized')
        g.write('</th>\n')
        g.write('</tr>\n')
        for i in range(confmat_rows):
            g.write('<tr>\n')
            g.write('<td>')
            g.write('actual '+str(classlabels[i]))
            g.write('</td>\n')
            for j in range(confmat_rows):
                g.write('<td>')
                g.write(str(round(ncm[i][j],3)))
                g.write('</td>\n')
            g.write('</tr>\n')

        g.write('</table><br>')
        g.close()

def write_pra_to_html(htmlname,precision,recall,accuracy,classindex,classlabel):
    with open(htmlname,'a') as g:
        g.write('<br>\n')
        g.write('class {} label {} '.format(classindex,classlabel))
        g.write('<br>\n')
        g.write('precision '+str(round(precision,3))+' ')
        g.write('recall '+str(round(recall,3))+' ')
        g.write('accuracy '+str(round(accuracy,3)))
        g.write('<br>\n')
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

def histogram_of_results_2classes(labelfile,multilabel_index,class_labels):

    array_boys_success = np.array([])
    array_girls_success = np.array([])
    array_boys_failure = np.array([])
    array_girls_failure = np.array([])

    with open (labelfile,'r') as fp:
        lines = fp.readlines()

    counter = 0
    for line in lines:
        counter += 1

        # split line to full path and label
        path = line.split()

        if path == []:
            logging.warning('got wierd line '+str(line))
            continue

        # Load numpy array (.npy), directory glob (*.jpg), or image file.
        input_file = os.path.expanduser(path[0])
        inputs = [caffe.io.load_image(input_file)]
        #inputs = [Utils.get_cv2_img_array(input_file)]

        print("Classifying %d inputs." % len(inputs))

        # Classify.
        start = time.time()
        predictions = gender_detector.genderator(inputs, path[2])
        print("Done in %.2f s." % (time.time() - start))

        strapless_predict = predictions[0][0]
        spaghetti_straps_predict = predictions[0][1]
        straps_predict = predictions[0][2]
        sleeveless_predict = predictions[0][3]
        cap_sleeve_predict = predictions[0][4]
        short_sleeve_predict = predictions[0][5]
        midi_sleeve_predict = predictions[0][6]
        long_sleeve_predict = predictions[0][7]

        max_result = max(predictions[0])

        max_result_index = np.argmax(predictions[0])

        true_label = int(path[1])
        predict_label = int(max_result_index)







    for root, dirs, files in os.walk(path):
        for file in files:
            #if file.startswith("face-"):
                predictions = gender_detector.genderator(root + "/" + file)
                if predictions[0][0] > predictions[0][1]:
                    array_boys_failure = np.append(array_boys_failure, predictions[0][0])
                    array_girls_failure = np.append(array_girls_failure, predictions[0][1])
                else:
                    array_boys_success=np.append(array_boys_success, predictions[0][0])
                    array_girls_success=np.append(array_girls_success, predictions[0][1])
                female_count += 1
    print ("female_count: %d" % (female_count))

    histogram=plt.figure(1)

    #bins = np.linspace(-1000, 1000, 50)

    plt.hist(array_boys_success, alpha=0.5, label='array_boys_success')
    plt.hist(array_girls_success, alpha=0.5, label='array_girls_success')
    plt.legend()

    plt.hist(array_boys_failure, alpha=0.5, label='array_boys_failure')
    plt.hist(array_girls_failure, alpha=0.5, label='array_girls_failure')
    plt.legend()

    histogram.savefig('test_image_for_faces.png')





    #!/usr/bin/env python

    import caffe
    import numpy as np
    from .. import background_removal, Utils, constants
    import cv2
    import os
    import sys
    import argparse
    import glob
    import time
    import skimage
    from PIL import Image
    from . import gender_detector
    import random
    import matplotlib.pyplot as plt


    array_success_with_plus_minus_category = np.array([])
    array_failure_with_plus_minus_category = np.array([])
    array_success_without = np.array([])
    array_failure_without = np.array([])

    text_file = open("db_dress_sleeve_train.txt", "r")

    counter = 0

    MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_50_dress_sleeve/ResNet-50-deploy.prototxt"
    PRETRAINED = "/home/yonatan/resnet50_caffemodels/caffe_resnet50_snapshot_50_sgd_iter_10000.caffemodel"
    caffe.set_mode_gpu()
    image_dims = [224, 224]
    mean, input_scale = np.array([120, 120, 120]), None
    #mean, input_scale = None, None
    #channel_swap = None
    channel_swap = [2, 1, 0]
    raw_scale = 255.0
    ext = 'jpg'

    # Make classifier.
    classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
                                  image_dims=image_dims, mean=mean,
                                  input_scale=input_scale, raw_scale=raw_scale,
                                  channel_swap=channel_swap)

    success_counter = 0
    failure_counter = 0
    guessed_mini_instead_midi = 0
    guessed_maxi_instead_midi = 0
    guessed_midi_instead_mini = 0
    guessed_maxi_instead_mini = 0
    guessed_midi_instead_maxi = 0
    guessed_mini_instead_maxi = 0

    counter_99_percent = 0
    counter_97_percent = 0
    counter_95_percent = 0
    counter_90_percent = 0

    failure_above_98_percent = 0

    #failure_current_result = 0
    #success_current_result = 0

    for line in text_file:
        counter += 1

        # split line to full path and label
        path = line.split()

        if path == []:
            continue

        # Load numpy array (.npy), directory glob (*.jpg), or image file.
        input_file = os.path.expanduser(path[0])
        inputs = [caffe.io.load_image(input_file)]
        #inputs = [Utils.get_cv2_img_array(input_file)]

        print("Classifying %d inputs." % len(inputs))

        # Classify.
        start = time.time()
        predictions = classifier.predict(inputs)
        print("Done in %.2f s." % (time.time() - start))

        strapless_predict = predictions[0][0]
        spaghetti_straps_predict = predictions[0][1]
        straps_predict = predictions[0][2]
        sleeveless_predict = predictions[0][3]
        cap_sleeve_predict = predictions[0][4]
        short_sleeve_predict = predictions[0][5]
        midi_sleeve_predict = predictions[0][6]
        long_sleeve_predict = predictions[0][7]

        max_result = max(predictions[0])

        max_result_index = np.argmax(predictions[0])

        true_label = int(path[1])
        predict_label = int(max_result_index)

        if predict_label == true_label:
            array_success_with_plus_minus_category = np.append(array_success_with_plus_minus_category, max_result)
            array_success_without = np.append(array_success_without, max_result)
        elif predict_label == 0 and true_label == 1:
            array_success_with_plus_minus_category = np.append(array_success_with_plus_minus_category, max_result)
            array_failure_without = np.append(array_failure_without, max_result)
        elif predict_label == 7 and true_label == 6:
            array_success_with_plus_minus_category = np.append(array_success_with_plus_minus_category, max_result)
            array_failure_without = np.append(array_failure_without, max_result)
        elif predict_label == (true_label + 1) or predict_label == (true_label - 1):
            array_success_with_plus_minus_category = np.append(array_success_with_plus_minus_category, max_result)
            array_failure_without = np.append(array_failure_without, max_result)
        else:
            array_failure_with_plus_minus_category = np.append(array_failure_with_plus_minus_category, max_result)
            array_failure_without = np.append(array_failure_without, max_result)
            print max_result

        print counter
        print predictions


    success_with = len(array_success_with_plus_minus_category)
    failure_with = len(array_failure_with_plus_minus_category)

    success_without = len(array_success_without)
    failure_without = len(array_failure_without)

    if success_with == 0 or failure_with == 0:
        print "wrong!"
    else:
        print 'accuracy percent with +-category: {0}'.format(float(success_with) / (success_with + failure_with))
        print 'accuracy percent without: {0}'.format(float(success_without) / (success_without + failure_without))

    histogram = plt.figure(1)

    plt.hist(array_success_with_plus_minus_category, bins=100, range=(0.9, 1), color='blue', label='array_success_with_plus_minus_category')
    plt.legend()

    plt.hist(array_failure_with_plus_minus_category, bins=100, range=(0.9, 1), color='red', label='array_failure_with_plus_minus_category')
    plt.legend()

    plt.hist(array_success_without, bins=100, range=(0.9, 1), color='green', label='array_success_without')
    plt.legend()

    plt.hist(array_failure_without, bins=100, range=(0.9, 1), color='pink', label='array_failure_without')
    plt.legend()

    histogram.savefig('db_dresses_histogram_iter_5000.png')


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='singe label accuracy tester')
    parser.add_argument('--testproto',  help='test prototxt')
    parser.add_argument('--deployproto',  help='deploy prototxt')
    parser.add_argument('--caffemodel', help='caffemodel')
    parser.add_argument('--gpu', help='gpu #',default=0)
    parser.add_argument('--cpu', help='use cpu')
    parser.add_argument('--output_layer_name', help='output layer name',default='estimate')
    parser.add_argument('--label_layer_name', help='label layer name',default='label')
    parser.add_argument('--label_file', help='label file name',default=None)
    parser.add_argument('--n_tests', help='number of examples to test',default=1000)
    parser.add_argument('--n_classes', help='number of classes',default=21)
    parser.add_argument('--classlabels', help='class labels (specify a list from trendi.constants)')

    args = parser.parse_args()
    print(args)
    if args.cpu is not None:
        gpu = -1
    elif args.gpu is not None:
        gpu = int(args.gpu)
    else:
        gpu = 0
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
#    single_label_acc(args.caffemodel,args.testproto,label_layer='label', estimate_layer=outlayer,n_tests=n_tests,gpu=gpu,classlabels=classlabels)

#def single_label_acc(caffemodel,          testproto,net=None,label_layer='label',estimate_layer='loss',n_tests=100,gpu=0,classlabels = constants.web_tool_categories_v2):
#python  /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/single_label_accuracy.py --caffemodel snapshot/res101_binary_dress_iter_37000.caffemodel --testproto ResNet-101-train_test.prototxt --output_layer_name estimate --n_classes 2 --n_tests 10

    n_classes = 2
    if args.label_file is None:
        args.label_file = '/data/jeremy/image_dbs/tamara_berg_street_to_shop/binary/dress_filipino_labels_balanced_test_250x250.txt'
    check_accuracy_deploy(args.deployproto,args.caffemodel,n_classes,args.label_file,n_tests=200,estimate_layer='prob_0',mean=(110,110,110),scale=None,finalsize=(224,224),resize_size=(250,250),gpu=gpu)


