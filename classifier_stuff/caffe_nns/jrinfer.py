__author__ = 'jeremy'
#get output images for given input
import numpy as np
from PIL import Image
import caffe
import os

def infer(images,prototxt,caffemodel):
    # load net
    net = caffe.Net(prototxt,caffemodel, caffe.TEST)
    for imagename in images:
        # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
        im = Image.open(imagename)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))
        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        out = net.blobs['score'].data[0].argmax(axis=0)
        result = Image.fromarray(out.astype(np.uint8))
        outname = im.strip('.png')[0]+'out.bmp'
        result.save(outname)
        fullout = net.blobs['score'].data[0]


if __name__ == "__main__":
    test_dir = '/root/imgdbs/image_dbs/colorful_fashion_parsing_data/images/test_200x150/'
    label_dir = '/root/imgdbs/image_dbs/colorful_fashion_parsing_data/labels/'
    images = [os.path.join(label_dir,f.split('jpg')[0]+'png') for f in test_dir if '.jpg' in f]
    prototxt = 'deploy.prototxt'
    caffemodel = 'snapshot/train_iter_88000.caffemodel'
    infer(images,prototxt,caffemodel)
