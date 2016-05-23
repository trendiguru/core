__author__ = 'jeremy'

__author__ = 'jeremy'
#get output images for given input
import numpy as np
from PIL import Image
import caffe
import os
import time
import cv2
import urllib
import logging
logging.basicConfig(level=logging.DEBUG)

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

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format

    if url.count('jpg') > 1:
        return None

    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if image.size == 0:
        return None
    new_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return new_image

def pixelparse(url_or_np_array):

    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
    else:
        return None

    start_time = time.time()
    in_ = np.array(image, dtype=np.float32)
#    possibly check size and resize if big
    dims = [150,100]
    in_ = cv2.resize(in_,dims)
    cv2.imshow('image',np.array(in_,dtype=np.uint8))
    cv2.waitKey(0)
    if len(in_.shape) != 3:
        logging.warning('got 1-chan image in neurodoll, making into 3chan')
        in_ = [in_,in_,in_]
    elif in_.shape[2] != 3:
        print('got n-chan image, skipping - shape is:'+str(in_.shape))
        return
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.0,116.7,122.7))
    print('image shape:'+str(in_.shape))
#    in_ = in_.transpose((2,0,1))   # dont need RGB->BGR if img is coming from cv2
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    elapsed_time=time.time()-start_time
    print('elapsed time:'+str(elapsed_time))

    return out.astype(np.uint8)



caffe.set_mode_gpu()
caffe.set_device(0)
print('loading caffemodel for neurodoll')
prototxt = '/home/jeremy/caffenets/pixlevel/voc-fcn8s/deploy.prototxt'
#caffemodel = 'snapshot_nn2/train_iter_183534.caffemodel'
#caffemodel = '/home/jeremy/caffenets/pixlevel/voc-fcn8s/train_iter_164620.caffemodel'  #010516 saved
caffemodel = '/home/jeremy/caffenets/pixlevel/voc-fcn8s/train_iter_457644.caffemodel'  #040516 saved
net = caffe.Net(prototxt,caffemodel, caffe.TEST)


if __name__ == "__main__":

    pixelparse('http://diamondfilms.com.au/wp-content/uploads/2014/08/Fashion-Photography-Sydney-1.jpg')
