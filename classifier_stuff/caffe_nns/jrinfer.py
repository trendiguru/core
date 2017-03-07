from __future__ import division
__author__ = 'jeremy'
#get output images for given input
import numpy as np
from PIL import Image
import os
import time
import cv2
import argparse
from datetime import datetime
import caffe
import sys
import logging
logging.basicConfig(level=logging.DEBUG)
import copy

#import json

# from trendi import pipeline
from trendi.utils import imutils
from trendi import constants
from trendi.paperdoll import paperdoll_parse_enqueue
from trendi import Utils
from trendi.utils import augment_images

def img_to_caffe(url_file_or_img_arr,dims=(224,224),mean=(104.0,116.7,122.7)):
    # load image in cv2 (so already BGR), resize, subtract mean, reorder dims to C x H x W for Caffe
    if isinstance(url_file_or_img_arr,basestring):
        print('working on:'+url_file_or_img_arr+' resize:'+str(dims)+' mean:'+str(mean))
    im = Utils.get_cv2_img_array(url_file_or_img_arr)
    if im is None:
        logging.warning('could not get image '+str(url_file_or_img_arr))
        return
    im = imutils.resize_keep_aspect(im,output_size=dims)
#    im = cv2.resize(im,dims)
    in_ = np.array(im, dtype=np.float32)
    if len(in_.shape) != 3:
        print('got 1-chan image, skipping')
        return
    elif in_.shape[2] != 3:
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
    print('shape before:'+str(in_.shape))
 #   in_ = in_[:,:,::-1] #RGB->BGR, not needed if reading with cv2
    in_ -= np.array(mean)
    in_ = in_.transpose((2,0,1)) #W,H,C -> C,W,H
    return in_

def infer_many_pixlevel(image_dir,prototxt,caffemodel,out_dir='./',mean=(104.0,116.7,122.7),filter='.jpg',
                        dims=(224,224),output_layer='pixlevel_sigmoid_output',save_legends=True,labels=constants.pixlevel_categories_v3):
    images = [os.path.join(image_dir,f) for f in os.listdir(image_dir) if filter in f]
    print(str(len(images))+' images in '+image_dir)
    net = caffe.Net(prototxt,caffemodel, caffe.TEST)
    start_time = time.time()
    masks=[]
    Utils.ensure_dir(out_dir)
    for imagename in images:
        print('working on:'+imagename)
            # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
        in_ = img_to_caffe(imagename,dims=dims,mean=mean)

        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        out = net.blobs[output_layer].data[0].argmax(axis=0)

#        result = Image.fromarray(out.astype(np.uint8))
    #        outname = im.strip('.png')[0]+'out.bmp'
        result = out.astype(np.uint8)
        outname = os.path.basename(imagename)
        outname = outname.split('.jpg')[0]+'.bmp'
        outname = os.path.join(out_dir,outname)
        print('outname:'+outname)
        cv2.imwrite(outname,result)
#        result.save(outname)
        masks.append(out.astype(np.uint8))
        if save_legends:
            imutils.show_mask_with_labels(outname,labels=labels,original_image=imagename,save_images=True)
    elapsed_time=time.time()-start_time
    print('elapsed time:'+str(elapsed_time)+' tpi:'+str(elapsed_time/len(images)))
    return masks
    #fullout = net.blobs['score'].data[0]

def infer_one_pixlevel(imagename,prototxt,caffemodel,out_dir='./',caffe_variant=None,dims=[224,224],output_layer='prob',mean=(104.0,116.7,122.7)):
    if caffe_variant == None:
        import caffe
    else:
        pass
    net = caffe.Net(prototxt,caffe.TEST,weights=caffemodel)
#    dims = [150,100] default for something??
    start_time = time.time()
    print('working on:'+imagename)
        # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(imagename)
    im = im.resize(dims,Image.ANTIALIAS)
    in_ = np.array(im, dtype=np.float32)
    if len(in_.shape) != 3:
        print('got 1-chan image, skipping')
        return
    elif in_.shape[2] != 3:
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
    print('shape before:'+str(in_.shape))
    in_ = in_[:,:,::-1]  #rgb-bgr
    in_ -= np.array(mean)
    in_ = in_.transpose((2,0,1))  #whc -> cwh
    print('shape after:'+str(in_.shape))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    #output_layer='prob'
    out = net.blobs[output_layer].data[0].argmax(axis=0)
    result = Image.fromarray(out.astype(np.uint8))
#        outname = im.strip('.png')[0]+'out.bmp'
    outname = os.path.basename(imagename)
    outname = outname.split('.jpg')[0]+'.bmp'
    outname = os.path.join(out_dir,outname)
    print('outname:'+outname)
    result.save(outname)
    #        fullout = net.blobs['score'].data[0]
    elapsed_time=time.time()-start_time
    print('elapsed time:'+str(elapsed_time))
    return out.astype(np.uint8)

def infer_one_hydra(url_or_image_arr,prototxt,caffemodel,out_dir='./',dims=(224,224),output_layers=['prob_0','prob_1','prob_2']):
    im = Utils.get_cv2_img_array(url_or_image_arr)
    if im is None:
        logging.warning('could not get image '+str(url_or_image_arr))
        return
    net = caffe.Net(prototxt, caffe.TEST,weights=caffemodel)
    start_time = time.time()
    print('working on:'+url_or_image_arr)
        # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = cv2.resize(im,dims)
    in_ = np.array(im, dtype=np.float32)
    if len(in_.shape) != 3:
        print('got 1-chan image, skipping')
        return
    elif in_.shape[2] != 3:
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
    print('shape before:'+str(in_.shape))
 #   in_ = in_[:,:,::-1] #RGB->BGR, not needed if reading with cv2
    in_ -= np.array((104.0,116.7,122.7))
    in_ = in_.transpose((2,0,1)) #W,H,C -> C,W,H
    print('img shape after:'+str(in_.shape)+' net data shape '+str(net.blobs['data'].shape))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    #output_layer='prob'
    out = []
    for output_layer in output_layers:
        one_out = net.blobs[output_layer].data
        out.append(one_out)
        print('output for {} is {}'.format(output_layer,one_out))
    print(str(out)+' elapsed time:'+str(time.time()-start_time))
    return out

def infer_many_hydra(url_or_image_arr_list,prototxt,caffemodel,out_dir='./',orig_size=(256,256),crop_size=(224,224),mean=(104.0,116.7,122.7),gpu=0):
    '''
    start net, get a bunch of results. TODO: resize to e.g. 250x250 (whatever was done in training) and crop to dims
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
    net = caffe.Net(prototxt, caffe.TEST,weights=caffemodel)
    print('params:'+str(net.params))
    out_layers = net.outputs
    print('out layers: '+str(out_layers))
    all_outs = []
    j=0
    start_time = time.time()
    for url_or_image_arr in url_or_image_arr_list:
        im = Utils.get_cv2_img_array(url_or_image_arr)
        if im is None:
            logging.warning('could not get image '+str(url_or_image_arr))
            continue
        print('infer_many_hydra working on:'+url_or_image_arr+' '+str(j)+'/'+str(len(url_or_image_arr_list)))
            # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
        im = imutils.resize_keep_aspect(im,output_size=orig_size)
#        im = cv2.resize(im,dims)
        im = imutils.center_crop(im,crop_size)
        in_ = np.array(im, dtype=np.float32)
        if len(in_.shape) != 3:
            print('got 1-chan image, skipping')
            return
        elif in_.shape[2] != 3:
            print('got n-chan image, skipping - shape:'+str(in_.shape))
            return
   #     print('shape before:'+str(in_.shape))
     #   in_ = in_[:,:,::-1] #RGB->BGR, not needed if reading with cv2
        in_ -= mean
        in_ = in_.transpose((2,0,1)) #W,H,C -> C,W,H
    #    print('img shape after:'+str(in_.shape)+' net data shape '+str(net.blobs['data'].shape))
        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        #output_layer='prob'
        out = []
        for output_layer in out_layers:
            one_out = net.blobs[output_layer].data[0]   #not sure why the data is nested [1xN] matrix and not a flat [N] vector
            out.append(copy.copy(one_out)) #the copy is required - if you dont do it then out gets over-written with each new one_out
            logging.debug('output for {} is {}'.format(output_layer,one_out))
        all_outs.append(out)
#        print('final till now:'+str(all_outs)+' '+str(all_outs2))
        j=j+1
    logging.debug('all output:'+str(all_outs))
    logging.debug('elapsed time:'+str(time.time()-start_time)+' tpi '+str((time.time()-start_time)/j))
    return all_outs

def infer_one_single_label(imagename,prototxt,caffemodel,out_dir='./',dims=[224,224],output_layer='prob'):
    net = caffe.Net(prototxt, caffe.TEST,weights=caffemodel)
    start_time = time.time()
    print('working on:'+imagename)
        # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(imagename)
    im = im.resize(dims,Image.ANTIALIAS)
    in_ = np.array(im, dtype=np.float32)
    if len(in_.shape) != 3:
        print('got 1-chan image, skipping')
        return
    elif in_.shape[2] != 3:
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
    print('shape before:'+str(in_.shape))
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.0,116.7,122.7))
    in_ = in_.transpose((2,0,1))
    print('shape after:'+str(in_.shape))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    #output_layer='prob'
    out = net.blobs[output_layer].data
    print(str(out)+' elapsed time:'+str(time.time()-start_time))
    return out

# make sure you have imported the right (nonstandard) version of caffe, e..g by changing pythonpath
def infer_one_deconvnet(imagename,prototxt,caffemodel,out_dir='./',caffe_variant=None,dims=(224,224)):
    net = caffe.Net(prototxt,caffemodel)
    dims = [224,224]
    start_time = time.time()
    print('working on:'+imagename)
        # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(imagename)
    im = im.resize(dims,Image.ANTIALIAS)
    in_ = np.array(im, dtype=np.float32)
    if len(in_.shape) != 3:
        print('got 1-chan image, skipping')
        return
    elif in_.shape[2] != 3:
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
    print('shape before:'+str(in_.shape))
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.0,116.7,122.7))
    in_ = in_.transpose((2,0,1))
    print('shape after:'+str(in_.shape))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['seg-score'].data[0].argmax(axis=0)
    result = Image.fromarray(out.astype(np.uint8))
#        outname = im.strip('.png')[0]+'out.bmp'
    outname = os.path.basename(imagename)
    outname = outname.split('.jpg')[0]+'.bmp'
    outname = os.path.join(out_dir,outname)
    print('outname:'+outname)
    result.save(outname)
    #        fullout = net.blobs['score'].data[0]
    elapsed_time=time.time()-start_time
    print('elapsed time:'+str(elapsed_time))
    return out.astype(np.uint8)

def test_pd_conclusions():
    test_dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/test_200x150/'
    images = [os.path.join(test_dir,f) for f in os.listdir(test_dir) if '.jpg' in f ]
    images = images[0:10]
    if(0):
        for filename in images:
            img_arr = cv2.imread(filename)
            retval = paperdoll_parse_enqueue.paperdoll_enqueue(img_arr, async=False,use_parfor=False)  #True,queue_name='pd_parfor')
            pdmask,pdlabels = retval.result[0:2]
            pdmask_after = pipeline.after_pd_conclusions(pdmask, constants.paperdoll_categories, face=None)
            h,w=pdmask.shape[0:2]
            pdmasks=np.zeros([h,2*w])
            pdmasks[:,0:w]=pdmask
            pdmasks[:,w:]=pdmask_after
            outfilename=filename.split('.jpg')[0]+'pd_masks.bmp'
            print('filename:'+str(outfilename))
            cv2.imwrite(outfilename,pdmasks)
            print('pdlabels:'+str(pdlabels))
            labellist = [x.key() for x in pdlabels]
            indexlist = [x.value() for x in pdlabels]
            print('labellist:'+str(labellist))
            print('indexlist:'+str(indexlist))
            paperdoll_parse_enqueue.show_parse('pd_masks.bmp',save=True)

    image = '/home/jeremy/core/images/vneck.jpg'
    prototxt = '/home/jeremy/caffenets/voc-fcn8s/deploy.prototxt'
    caffemodel = 'snapshot_nn2/train_iter_183534.caffemodel'
    caffemodel = 'snapshot_nn2/train_iter_164620.caffemodel'  #010516 saved
    caffemodel = '/home/jeremy/caffenets/voc-fcn8s/train_iter_457644.caffemodel'  #040516 saved
    #mask = infer_one(image,prototxt,caffemodel)
    masks = infer_many(images,prototxt,caffemodel)
#    imutils.show_mask_with_labels('vneck.bmp',constants.fashionista_categories_augmented)
    i=0
    for nnmask in masks:
        nnmask_after = pipeline.after_nn_conclusions(nnmask, constants.fashionista_categories_for_conclusions, face=None)
        filename=images[i]
        cv2.imwrite(filename+'after_pd_mask.bmp',nnmask_after )
        i=i+1
        h,w=nnmask.shape[0:2]
        nnmasks=np.zeros([h,2*w])
        nnmasks[:,0:w]=nnmask
        nnmasks[:,w:]=nnmask_after
        outfilename=filename.split('.jpg')[0]+'nn_masks.bmp'
        print('outfilename:'+str(outfilename))
        cv2.imwrite(outfilename,nnmasks)
        nice_display=imutils.show_mask_with_labels(outfilename,constants.fashionista_categories_augmented_zero_based,save_images=True,visual_output=True,original_image=filename)
 #       displayname = outfilename.split('.bmp')[0]+'_display.jpg'
 #       cv2.imwrite(displayname,nice_display)

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, n_images, layer='score', gt='label',labels=constants.ultimate_21,mean=(120,120,120),denormalize=True):
    n_cl = net.blobs[layer].channels
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    print('n channels: '+str(n_cl))
    for idx in n_images:
        net.forward()
        print('idx:'+str(idx))

        try:
            x=net.blobs[gt]
        except:
            print "error on x= :", sys.exc_info()[0]
            continue
        try:
            print('gt data type '+str(type(net.blobs[gt])))
#            print('gt data type '+str(type(net.blobs[gt].data)))
#            print('gt data shape:'+str(net.blobs[gt].data.shape))
#            print('gt data [0,0]shape:'+str(net.blobs[gt].data[0,0].shape))
        except:
            print "error on datatype:", sys.exc_info()[0]
            continue
        gt_data = net.blobs[gt].data[0, 0]
        net_data = net.blobs[layer].data[0]

        hist += fast_hist(gt_data.flatten(),
                                net_data.argmax(0).flatten(),
                                n_cl)
        if save_dir:
#            continue
            Utils.ensure_dir(save_dir)
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
#            im = net.blobs[layer].data[0].argmax(0).astype(np.uint8)
            savename = os.path.join(save_dir, str(idx) + '.png')
   #         print('label size:'+str(im.shape))
            im.save(savename)
            orig_image = net.blobs['data'].data[0]
            gt_image =   net.blobs['label'].data[0,0]
            print('orig image size:'+str(orig_image.shape)+' gt:'+str(gt_image.shape))
#            gt_reshaped = np.reshape(gt,[gt.shape[1],gt.shape[2]])
#            gt_reshaped = np.reshape(gt,[gt.shape[1],gt.shape[2]])
            orig_image_transposed = orig_image.transpose((1,2,0))   #CxWxH->WxHxC
            orig_image_transposed += np.array(mean)
            min = np.min(orig_image_transposed)
            max = np.max(orig_image_transposed)
            print('xformed image size:'+str(orig_image_transposed.shape)+' gt:'+str(gt_image.shape))
            print('xformed image max {} min {} :'.format(max,min))
            if denormalize:
                orig_image_transposed = orig_image_transposed-min
                orig_image_transposed = orig_image_transposed*255.0/(max-min)
                min = np.min(orig_image_transposed)
                max = np.max(orig_image_transposed)
                print('after denorm image max {} min {} :'.format(max,min))
            orig_image_transposed = orig_image_transposed.astype(np.uint8)
            orig_savename = os.path.join(save_dir, str(idx) + 'orig.jpg')
            cv2.imwrite(orig_savename,orig_image_transposed)
            gt_savename = os.path.join(save_dir, str(idx) + 'gt.png')
            cv2.imwrite(gt_savename,gt_image)
            imutils.show_mask_with_labels(savename,labels,original_image=orig_savename,save_images=True,visual_output=False) #if these run in docker ontainers then no vis. output :<
            imutils.show_mask_with_labels(gt_savename,labels,original_image=orig_savename,save_images=True,visual_output=False)
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)

def results_from_hist(hist,save_file='./summary_output.txt',info_string='',labels=constants.ultimate_21):
    # mean loss
    overall_acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'overall accuracy', overall_acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(),  'acc per class', str(acc)
    print '>>>', datetime.now(),  'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    print '>>>', datetime.now(), 'fwavacc', \
            fwavacc
    mean_acc = np.nanmean(acc)
    mean_iou = np.nanmean(iu)
    results_dict = {'class_accuracy':acc.tolist(),'overall_acc':overall_acc.tolist(),'mean_acc':mean_acc.tolist(),'class_iou':iu.tolist(),'mean_iou':mean_iou.tolist(),'fwavacc':fwavacc.tolist()}
    if save_file:
        with open(save_file,'a+') as f:  #a+ creates if it doesnt exist
            f.write('net output '+ str(datetime.now())+' ' + info_string+ '\n')
            f.write('<br>\n')
            f.write('classes: \n')
            for i in range(len(labels)):
                f.write(str(i)+':'+labels[i]+' ')
            f.write('<br>\n')
            f.write('acc per class:'+ str(acc)+'\n')
            f.write('<br>\n')
            f.write('overall acc:'+ str(overall_acc)+'\n')
            f.write('<br>\n')
            f.write('mean acc:'+ str(np.nanmean(acc))+'\n')
            f.write('<br>\n')
            f.write('IU per class:'+ str(iu)+'\n')
            f.write('<br>\n')
            f.write('mean IU:'+ str(np.nanmean(iu))+'\n')
            f.write('<br>\n')
            f.write('fwavacc:'+ str((freq[freq > 0] * iu[freq > 0]).sum())+'\n')
            f.write('<br>\n')
            f.write('<br>\n')
    return results_dict

def seg_tests(solver, n_images, output_layer='score', gt_layer='label',outfilename='net_output.txt',save_dir=None,labels=constants.pixlevel_categories_v3):
    print '>>>', datetime.now(), 'Begin seg tests'
    if save_dir is not None:
        print('saving net test output to '+save_dir)
        Utils.ensure_dir(save_dir)
    else:
        save_dir = None
    solver.test_nets[0].share_with(solver.net)
    results_dict = do_seg_tests(solver.test_nets[0], solver.iter, save_dir, n_images, output_layer, gt_layer,outfilename=outfilename,labels=labels)
    return results_dict

def do_seg_tests(net, iter, save_dir, n_images, output_layer='score', gt_layer='label',outfilename='net_output.txt',save_output=False,savedir='testoutput',labels=constants.pixlevel_categories_v3):
    n_cl = net.blobs[output_layer].channels
    if save_dir:
#        save_format = save_format.format(iter)
        Utils.ensure_dir(save_dir)
    hist, loss = compute_hist(net, save_dir, n_images, output_layer, gt_layer,labels=labels)
    # mean loss
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    # overall accuracy
    overall_acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', overall_acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'acc per class', str(acc)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            fwavacc
    mean_acc = np.nanmean(acc)
    mean_iou = np.nanmean(iu)
    results_dict = {'iter':iter,'loss':loss,'class_accuracy':acc.tolist(),'overall_acc':overall_acc.tolist(),'mean_acc':mean_acc.tolist(),'class_iou':iu.tolist(),'mean_iou':mean_iou.tolist(),'fwavacc':fwavacc.tolist()}
    jsonfile = outfilename[:-4]+'.json'
#    with open(jsonfile, 'a+') as outfile:
#        json.dump(results_dict, outfile)
#        outfile.close()

    with open(outfilename,'a') as f:
        f.write('>>>'+ str(datetime.now())+' Iteration:'+ str(iter)+ ' loss:'+ str(loss)+'\n')
        f.write('<br>\n')
        f.write('acc per class:'+ str(acc)+'\n')
        f.write('<br>\n')
        f.write('overall acc:'+ str(overall_acc)+'\n')
        f.write('<br>\n')
        f.write('mean acc:'+ str(np.nanmean(acc))+'\n')
        f.write('<br>\n')
        f.write('IU per class:'+ str(iu)+'\n')
        f.write('<br>\n')
        f.write('mean IU:'+ str(np.nanmean(iu))+'\n')
        f.write('<br>\n')
        f.write('fwavacc:'+ str((freq[freq > 0] * iu[freq > 0]).sum())+'\n')
        f.write('<br>\n')
        f.write('<br>\n')
    return results_dict

#    imutils.show_mask_with_labels('concout.bmp',constants.fashionista_categories_augmented)
def inspect_net(prototxt,caffemodel):
    net = caffe.Net(prototxt,caffemodel, caffe.TEST)
    for key,value in net.blobs.iteritems():
        print(key,value)

if __name__ == "__main__":

    print('starting')

    test_dir = '/root/imgdbs/image_dbs/doorman/irrelevant/'
    out_dir = '/root/imgdbs/image_dbs/doorman/irrelevant_output'
    caffemodel = 'snapshot_nn2/train_iter_183534.caffemodel'

    test_dir = '/root/imgdbs/image_dbs/colorful_fashion_parsing_data/images'
    out_dir = '/root/imgdbs/image_dbs/150x100output_010516/'
    caffemodel = 'snapshot_nn2/train_iter_164620.caffemodel'  #010516 saved

    test_dir = '/root/imgdbs/image_dbs/colorful_fashion_parsing_data/images/test'
    out_dir = './'

    test_dir = '/root/imgdbs/image_dbs/colorful_fashion_parsing_data/test_256x256'
    out_dir = '/home/jeremy/caffenets/pixlevel/voc-fcn8s/voc8.15/output'
    caffemodel = '/home/jeremy/caffenets/pixlevel/voc-fcn8s/voc8.15/snapshot/train_iter_120000.caffemodel'
    prototxt = '/home/jeremy/caffenets/pixlevel/voc-fcn8s/voc8.15/deploy.prototxt'

    parser = argparse.ArgumentParser(description='get Caffe output')
    parser.add_argument('--model', help='caffemodel', default=caffemodel)
    parser.add_argument('--solverproto', help='solver prototxt',default='solver.prototxt')
    parser.add_argument('--image', dest = 'image_file', help='image file',default=None)
    parser.add_argument('--dir', dest = 'image_directory', help='image directory',default=None)
    parser.add_argument('--outdir', dest = 'out_directory', help='result directory',default='.')
    parser.add_argument('--gpu', help='use gpu',default='True')
    parser.add_argument('--caffe_variant', help='caffe variant',default=None)
    parser.add_argument('--dims', help='dims for net',default=None)
    parser.add_argument('--iou',help='do iou test on pixel level net',default=True)
    parser.add_argument('--output_layer',help='output layer of net',default='output')
    args = parser.parse_args()
    print('args:'+str(args))
    print('caffemodel:'+str(args.model))
#    label_dir = '/root/imgdbs/image_dbs/colorful_fashion_parsing_data/labels/'
    if args.caffe_variant:
        infer_one_deconvnet(args.image_file,args.prototxt,args.caffemodel,out_dir=args.out_directory)

    if args.gpu:
        caffe.set_mode_gpu()
        caffe.set_device(int(args.gpu))
    else:
        caffe.set_mode_cpu()

    if args.iou == 'True' or args.iou == 'true' or args.iou =='1':
        print('using net defined by {} and {} '.format(args.solverproto,args.model))
        solver = caffe.SGDSolver(args.solverproto)
        solver.net.copy_from(args.model)
#        if args.image_file:
#            val = range(0,1)
#            seg_tests(solver, False, val, layer='score')
#        elif args.image_directory:
#            images = [os.path.join(args.image_directory,f) for f in os.listdir(args.image_directory) if '.jpg' in f ]
#            print('nimages:'+str(len(images)) + ' in directory '+args.image_directory)
        val = range(0,200)
            #this just runs the train net i think, doesnt test new images
        #seg_tests(solver, n_images, output_layer='score', gt_layer='label',outfilename='net_output.txt',save_dir=None,labels=constants.pixlevel_categories_v3):

        seg_tests(solver,  val, output_layer=args.output_layer,save_dir='outs')
#        else:
#            print('gave neither image nor directory as input to iou test')
    #do image level tests
    else:
        if args.image_file:
            infer_one(args.image_file,args.prototxt,args.caffemodel,out_dir=args.out_directory)
        elif args.image_directory:
            images = [os.path.join(args.image_directory,f) for f in os.listdir(args.image_directory) if '.jpg' in f ]
            print('nimages:'+str(len(images)) + ' in directory '+args.image_directory)
            infer_many(images,args.prototxt,args.caffemodel,out_dir=args.out_directory)
        else:
            print('gave neither image nor directory as input')