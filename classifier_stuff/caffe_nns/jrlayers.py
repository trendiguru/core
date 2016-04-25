import copy
import os
import caffe
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
from PIL import Image

import random

class SBDDSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - sbdd_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.images_dir = params['images_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.labels_dir = params.get('labels_dir',self.images_dir)
        self.imagesfile = params.get('imagesfile',os.path.join(self.images_dir,self.split+'images.txt'))
        self.labelsfile = params.get('labelsfile',None)
        #if there is no labelsfile specified then rename imagefiles to make labelfile names
        self.labelfile_suffix = params.get('labelfile_suffix','.png')

        print('PRINTlabeldir {} imagedir {} labelfile {} imagefile {}'.format(self.labels_dir,self.images_dir,self.labelsfile,self.imagesfile))
        logging.debug('LOGGINGlabeldir {} imagedir {} labelfile {} imagefile {}'.format(self.labels_dir,self.images_dir,self.labelsfile,self.imagesfile))
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
    #if file not found and its not a path then tack on the training dir as a default locaiton for the trainingimages file
    if not os.path.isfile(self.imagesfile) and not '/' in self.imagesfile:
        self.imagesfile = os.path.join(self.images_dir,self.imagesfile)
    if not os.path.isfile(self.imagesfile):
        print('COULD NOT OPEN IMAGES FILE '+str(self.imagesfile))

    self.imagefiles = open(self.imagesfile, 'r').read().splitlines()
    self.n_files = len(self.imagefiles)
#        self.indices = open(split_f, 'r').read().splitlines()
    if self.labelsfile is not None:  #if labels flie is none then get labels from images
        if not os.path.isfile(self.labelsfile) and not '/' in self.labelsfile:
            self.labelsfile = os.path.join(self.labels_dir,self.labelsfile)
        if not os.path.isfile(self.labelsfile):
            print('COULD NOT OPEN labelS FILE '+str(self.labelsfile))
            self.labelfiles = open(self.labelsfile, 'r').read().splitlines()

    self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.imagefiles)-1)
        logging.debug('self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))

    def reshape(self, bottom, top):
        # load image + label image pair
#	logging.debug('self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))
        self.data = self.load_image(self.idx)
#	if self.load_labels_from_mat:
#            self.label = self.load_label(self.indices[self.idx])#
#	else:
        self.label = self.load_label_image(self.idx)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def next_idx(self):
        if self.random:
            self.idx = random.randint(0, len(self.imagefiles)-1)
        else:
            self.idx += 1
            if self.idx == len(self.imagefiles):
                self.idx = 0


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.imagefiles)-1)
        else:
            self.idx += 1
            if self.idx == len(self.imagefiles):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def determine_label_filename(self,idx):
        if self.labelsfile is not None:
                filename = self.labelfiles[idx]
        #if there is no labelsfile specified then rename imagefiles to make labelfile names
        #so strip imagefile to get labelfile name
        else:
            filename = self.imagefiles[idx]
            filename = filename.split('.jpg')[0]
            filename = filename+self.labelfile_suffix

        full_filename=os.path.join(self.labels_dir,filename)
        return full_filename

    def load_image(self,idx):
            """
            Load input image and preprocess for Caffe:
            - cast to float
            - switch channels RGB -> BGR
            - subtract mean
            - transpose to channel x height x width order
            """
    #	print('IN LOAD IMAGE self idx is :'+str(idx)+' type:'+str(type(idx)))
            filename = self.imagefiles[idx]
        full_filename=os.path.join(self.images_dir,filename)
        print('imagefile:'+full_filename)
        while(1):
            filename = self.imagefiles[idx]
            full_filename=os.path.join(self.images_dir,filename)
            label_filename=self.determine_label_filename(self.idx)
            if not(os.path.isfile(label_filename) and os.path.isfile(full_filename)):
                print('ONE OF THESE IS NOT A FILE:'+str(label_filename)+','+str(full_filename))
                self.next_idx()
            else:
                break
            im = Image.open(full_filename)
            in_ = np.array(im, dtype=np.float32)
            in_ = in_[:,:,::-1]
    #        in_ -= self.mean
            in_ = in_.transpose((2,0,1))
    #	print('uniques of img:'+str(np.unique(in_))+' shape:'+str(in_.shape))
            return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        label = label[np.newaxis, ...]
        return label


    def load_label_image(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
#        import scipy.io
#        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
        full_filename = self.determine_label_filename(idx)
        print('labelfile:'+full_filename)
            im = Image.open(full_filename)
        if im is None:
            print(' COULD NOT LOAD FILE '+full_filename)
            in_ = np.array(im, dtype=np.uint8)
        in_ = in_ - 1
        print('uniques of label:'+str(np.unique(in_))+' shape:'+str(in_.shape))
            label = copy.copy(in_[np.newaxis, ...])
        print('after extradim shape:'+str(label.shape))

        return label
