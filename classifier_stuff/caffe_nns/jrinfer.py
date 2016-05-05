__author__ = 'jeremy'
#get output images for given input
import numpy as np
from PIL import Image
import caffe
import os
import time

from trendi import pipeline

def infer_many(images,prototxt,caffemodel,out_dir='./'):
    net = caffe.Net(prototxt,caffemodel, caffe.TEST)
    dims = [150,100]
    start_time = time.time()
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
    elapsed_time=time.time()-start_time
    print('elapsed time:'+str(elapsed_time)+' tpi:'+str(elapsed_time/len(images)))
    return result
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
    return result

def test_pd_conclusions():
#    images = [os.path.join(test_dir,f) for f in os.listdir(test_dir) if '.jpg' in f ]
    image = '/home/jeremy/core/images/vneck.jpg'
    prototxt = '/home/jeremy/caffenets/voc-fcn8s/deploy.prototxt'
    caffemodel = 'snapshot_nn2/train_iter_183534.caffemodel'
    caffemodel = 'snapshot_nn2/train_iter_164620.caffemodel'  #010516 saved
    caffemodel = '/home/jeremy/caffenets/voc-fcn8s/train_iter_457644.caffemodel'  #040516 saved
    infer(image,prototxt,caffemodel,out_dir=out_dir)

  #after_nn_conclusions(mask, labels, face=None):


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
    infer(images,prototxt,caffemodel,out_dir=out_dir)
