import caffe

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
        self.labels_dir = params['labels_dir']
        self.images_dir = params['images_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.imagesfile = params.get('imagefile',self.split+'images.txt')
        self.labelsfile = params.get('labelfile',self.split+'labels.txt')

        print('labeldir {} imagedir {} ;abe;file {} imagefile {}'.format(self.labels_dir,self.images_dir,self.labelsfile,self.images$
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.imagefiles = open(imagesfile, 'r').read().splitlines()
        self.labelfiles = open(labelsfile, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
#       if self.load_labels_from_mat:
#            self.label = self.load_label(self.indices[self.idx])#
#       else:
        self.label = self.load_label_image(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self,idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        filename = self.imagefiles[idx]
        full_filename=os.path.join(self.images_dir,filename)
        print('imagefile:'+full_filename)
        im = Image.open(filename)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
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

        filename = self.labelfiles[idx]

        filename = self.imagefiles[idx]
        filename = filename.split('.jpg')[0]
        filename = filename+'.png'
        full_filename=os.path.join(self.labels_dir,filename)
        print('labelfile:'+full_filename)
        im = Image.open(full_filename)
        if im is None:
            print(' COULD NOT LOAD FILE '+full_filename)
        in_ = np.array(im, dtype=np.uint8)
        label = in_[np.newaxis, ...]
        return label



