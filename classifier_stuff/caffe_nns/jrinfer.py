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
import json

# from trendi import pipeline
from trendi.utils import imutils
from trendi import constants
from trendi.paperdoll import paperdoll_parse_enqueue
from trendi import Utils

def infer_many(images,prototxt,caffemodel,out_dir='./',caffe_variant=None):
    net = caffe.Net(prototxt,caffemodel, caffe.TEST)
    dims = [150,100]
    start_time = time.time()
    masks=[]
    Utils.ensure_dir(out_dir)
    for imagename in images:
        print('working on:'+imagename)
            # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
        im = Image.open(imagename)
#        im = im.resize(dims,Image.ANTIALIAS)
        in_ = np.array(im, dtype=np.float32)
        if len(in_.shape) != 3:
            print('got 1-chan image, skipping')
            continue
        elif in_.shape[2] != 3:
            print('got n-chan image, skipping - shape:'+str(in_.shape))
            continue
        print('size:'+str(in_.shape))
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.0,116.7,122.7))
        in_ = in_.transpose((2,0,1))
        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        out = net.blobs['score'].data[0].argmax(axis=0)
        result = Image.fromarray(out.astype(np.uint8))
    #        outname = im.strip('.png')[0]+'out.bmp'
        outname = os.path.basename(imagename)
        outname = outname.split('.jpg')[0]+'.bmp'
        outname = os.path.join(out_dir,outname)
        print('outname:'+outname)
        result.save(outname)
        masks.append(out.astype(np.uint8))
    elapsed_time=time.time()-start_time
    print('elapsed time:'+str(elapsed_time)+' tpi:'+str(elapsed_time/len(images)))
    return masks
    #fullout = net.blobs['score'].data[0]

def infer_one(imagename,prototxt,caffemodel,out_dir='./',caffe_variant=None):
    if caffe_variant == None:
        import caffe
    else:
        pass
    net = caffe.Net(prototxt,caffemodel, caffe.TEST)
    dims = [150,100]
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
    out = net.blobs['score'].data[0].argmax(axis=0)
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

def compute_hist(net, save_dir, dataset, layer='score', gt='label',labels=constants.ultimate_21):
    n_cl = net.blobs[layer].channels
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)
        if save_dir:
            Utils.ensure_dir(save_dir)
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            savename = os.path.join(save_dir, str(idx) + '.png')
#            print('label size:'+str(im.shape))
            im.save(savename)
            orig_image = net.blobs['data'].data[0]
            gt =         net.blobs['label'].data[0]
            print('orig image size:'+str(orig_image.shape)+' gt:'+str(gt.shape))
            orig_image = orig_image.transpose((1,2,0))
            print('orig image size:'+str(orig_image.shape))
            orig_savename = os.path.join(save_dir, str(idx) + 'orig.jpg')
            cv2.imwrite(orig_savename,orig_image)
            gt_savename = os.path.join(save_dir, str(idx) + 'gt.png')
            cv2.imwrite(gt_savename,gt)
            imutils.show_mask_with_labels(savename,labels,original_image=orig_savename,save_images=True,visual_output=True)
            imutils.show_mask_with_labels(gt_savename,labels,original_image=orig_savename,save_images=True,visual_output=True)
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, layer='score', gt='label',outfilename='net_output.txt'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    results_dict = do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt,outfilename=outfilename)
    return results_dict

def do_seg_tests(net, iter, save_dir, dataset, layer='score', gt='label',outfilename='net_output.txt'):
    n_cl = net.blobs[layer].channels
    if save_dir:
#        save_format = save_format.format(iter)
        Utils.ensure_dir(save_dir)
    hist, loss = compute_hist(net, save_dir, dataset, layer, gt)
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
    parser.add_argument('--prototxt', help='prototxt',default='solver.prototxt')
    parser.add_argument('--image', dest = 'image_file', help='image file',default=None)
    parser.add_argument('--dir', dest = 'image_directory', help='image directory',default=None)
    parser.add_argument('--outdir', dest = 'out_directory', help='result directory',default='.')
    parser.add_argument('--gpu', help='use gpu',default='True')
    parser.add_argument('--caffe_variant', help='caffe variant',default=None)
    parser.add_argument('--dims', help='dims for net',default=None)
    parser.add_argument('--iou',help='do iou test on pixel level net',default=True)
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
        print('using net defined by {} and {} '.format(args.prototxt,args.model))
        solver = caffe.SGDSolver(args.prototxt)
        solver.net.copy_from(args.model)
#        if args.image_file:
#            val = range(0,1)
#            seg_tests(solver, False, val, layer='score')
#        elif args.image_directory:
#            images = [os.path.join(args.image_directory,f) for f in os.listdir(args.image_directory) if '.jpg' in f ]
#            print('nimages:'+str(len(images)) + ' in directory '+args.image_directory)
        val = range(0,200)
            #this just runs the train net i think, doesnt test new images
        seg_tests(solver, False, val, layer='score')
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