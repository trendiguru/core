__author__ = 'jeremy'
#get output images for given input
import numpy as np
from PIL import Image
import caffe
import os
import time
import cv2

from trendi import pipeline
from trendi.utils import imutils
from trendi import constants
from trendi.paperdoll import paperdoll_parse_enqueue


def infer_many(images,prototxt,caffemodel,out_dir='./'):
    net = caffe.Net(prototxt,caffemodel, caffe.TEST)
    dims = [150,100]
    start_time = time.time()
    masks=[]
    for imagename in images:
        print('working on:'+imagename)
            # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
        im = Image.open(imagename)
        im = im.resize(dims,Image.ANTIALIAS)
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

def infer_one(imagename,prototxt,caffemodel,out_dir='./'):
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
    #        fullout = net.blobs['score'].data[0]
    elapsed_time=time.time()-start_time
    print('elapsed time:'+str(elapsed_time))
    return out.astype(np.uint8)

def test_pd_conclusions():
    test_dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/test_200x150/'
    images = [os.path.join(test_dir,f) for f in os.listdir(test_dir) if '.jpg' in f ]
    images = images[0:1]
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
        img_arr = cv2.imread(filename)
        retval = paperdoll_parse_enqueue.paperdoll_enqueue(img_arr, async=False,use_parfor=False)  #True,queue_name='pd_parfor')
        pdmask,pdlabels = retval.result[0:2]
        pdmask_after = pipeline.after_pd_conclusions(pdmask, constants.paperdoll_categories, face=None)
        i=i+1
        h,w=nnmask.shape[0:2]
        nnmasks=np.zeros([h,2*w])
        pdmasks=np.zeros([h,2*w])
        nnmasks[:,0:w]=nnmask
        nnmasks[:,w:]=nnmask_after
        outfilename=filename.split('.jpg')[0]+'nn_masks.bmp'
        print('filename:'+str(outfilename))
        cv2.imwrite(filename,nnmasks)
        nice_display=imutils.show_mask_with_labels(filename,constants.fashionista_categories_augmented_zero_based,save_images=True,visual_output=True)
        pdmasks[:,0:w]=pdmask
        pdmasks[:,w:]=pdmask_after
        outfilename=filename.split('.jpg')[0]+'pd_masks.bmp'
        cv2.imwrite(outfilename,nnmask_after )
        print('pdlabels:'+str(pdlabels))
        labellist = [x.key() for x in pdlabels]
        indexlist = [x.value() for x in pdlabels]
        print('labellist:'+str(labellist))
        print('indexlist:'+str(indexlist))
#        nice_display=imutils.show_mask_with_labels(filename,constants.fashionista_categories_augmented_zero_based)
        paperdoll_parse_enqueue.show_parse('pd_masks.bmp',save=True)

#    imutils.show_mask_with_labels('concout.bmp',constants.fashionista_categories_augmented)


if __name__ == "__main__":
    caffe.set_mode_gpu();
    caffe.set_device(0);
    print('starting')
    test_dir = '/root/imgdbs/image_dbs/doorman/irrelevant/'
    out_dir = '/root/imgdbs/image_dbs/doorman/irrelevant_output'
    test_dir = '/root/imgdbs/image_dbs/colorful_fashion_parsing_data/images/test_200x150/'
    out_dir = '/root/imgdbs/image_dbs/150x100output_010516/'
#    label_dir = '/root/imgdbs/image_dbs/colorful_fashion_parsing_data/labels/'
#    images = [os.path.join(test_dir,f) for f in os.listdir(test_dir) if '.jpg' in f or 'jpeg' in f or '.bmp' in f]
    images = [os.path.join(test_dir,f) for f in os.listdir(test_dir) if '.jpg' in f ]
    print('nimages:'+str(len(images)))
#    images = [f.strip('.jpg')[0]+'.png' for f in images]
#    print('images:'+str(images))
#    images = [os.path.join(label_dir,f) for f in images]
#    print('images:'+str(images))
    prototxt = 'deploy.prototxt'
    caffemodel = 'snapshot_nn2/train_iter_183534.caffemodel'
    caffemodel = 'snapshot_nn2/train_iter_164620.caffemodel'  #010516 saved
    infer_many(images,prototxt,caffemodel,out_dir=out_dir)
