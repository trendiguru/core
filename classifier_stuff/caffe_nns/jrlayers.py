import copy
import os
import caffe
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
from PIL import Image
import cv2
import random

class JrLayer(caffe.Layer):
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
        layer {
            name: "data"
            type: "Python"
            top: "data"
            top: "label"
            python_param {
            module: "jrlayers"
            layer: "JrLayer"
            param_str: "{\'images_dir\': \'/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/train_u21_256x256\', \'labels_dir\':\'/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels_256x256/\', \'mean\': (104.00699, 116.66877, 122.67892)}"
            }
        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.images_dir = params['images_dir']
        self.labels_dir = params.get('labels_dir',self.images_dir)
        self.mean = np.array(params['mean'])
        self.random_init = params.get('random_initialization', True) #start from random point in image list
        self.random_pick = params.get('random_pick', True) #pick random image from list every time
        self.seed = params.get('seed', 1337)
#        self.imagesfile = params.get('imagesfile',os.path.join(self.images_dir,self.split+'images.txt'))
        self.imagesfile = params.get('imagesfile',None)
        self.labelsfile = params.get('labelsfile',None)
        #if there is no labelsfile specified then rename imagefiles to make labelfile names
        self.labelfile_suffix = params.get('labelfile_suffix','.png')
        self.imagefile_suffix = params.get('labelfile_suffix','.jpg')
        self.new_size = params.get('new_size',None)

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
        if self.imagesfile is not None:
            if not os.path.isfile(self.imagesfile) and not '/' in self.imagesfile:
                self.imagesfile = os.path.join(self.images_dir,self.imagesfile)
            if not os.path.isfile(self.imagesfile):
                print('COULD NOT OPEN IMAGES FILE '+str(self.imagesfile))
            self.imagefiles = open(self.imagesfile, 'r').read().splitlines()
            self.n_files = len(self.imagefiles)
    #        self.indices = open(split_f, 'r').read().splitlines()
        else:
            self.imagefiles = [f for f in os.listdir(self.images_dir) if self.imagefile_suffix in f]

        if self.labelsfile is not None:  #if labels flie is none then get labels from images
            if not os.path.isfile(self.labelsfile) and not '/' in self.labelsfile:
                self.labelsfile = os.path.join(self.labels_dir,self.labelsfile)
            if not os.path.isfile(self.labelsfile):
                print('COULD NOT OPEN labelS FILE '+str(self.labelsfile))
                self.labelfiles = open(self.labelsfile, 'r').read().splitlines()
        else:
            self.labelfiles = [f for f in os.listdir(self.labels_dir) if self.labelfile_suffix in f]
            self.n_files = len(self.imagefiles)
        print(str(self.n_files)+' label files in label dir '+str(self.labels_dir))

        self.idx = 0
        # randomization: seed and pick
        if self.random_init:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.imagefiles)-1)
        if self.random_pick:
            random.shuffle(self.imagefiles)
        logging.debug('initial self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))

        ##check that all images are openable and have labels
        check_files = False
        if(check_files):
            good_img_files = []
            good_label_files = []
            print('checking image files')
            for ind in range(len(self.imagefiles)):
                img_arr = self.load_image(ind)
                if img_arr is not None:
                    label_arr = self.load_label_image(ind)
                    if label_arr is not None:
                        if label_arr.shape[1:3] == img_arr.shape[1:3]:  #the first dim is # channels (3 for img and 1 for label
                            good_img_files.append(self.imagefiles[ind])
                            good_label_files.append(self.labelfiles[ind])
                        else:
                            print('match , image {} and label {}'.format(img_arr.shape,label_arr.shape))
                else:
                    print('got bad image:'+self.imagefiles[ind])
            self.imagefiles = good_img_files
            self.labelfiles = good_label_files
            assert(len(self.imagefiles) == len(self.labelfiles))
            print('{} images and {} labels'.format(len(self.imagefiles),len(self.labelfiles)))
            self.n_files = len(self.imagefiles)
            print(str(self.n_files)+' good files in image dir '+str(self.images_dir))

    def reshape(self, bottom, top):
        print('reshaping')
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
        if self.random_pick:
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
        if self.random_pick:
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
        while(1):
            filename = self.imagefiles[idx]
            full_filename=os.path.join(self.images_dir,filename)
#            print('the imagefile:'+full_filename+' index:'+str(idx))
            label_filename=self.determine_label_filename(idx)
            if not(os.path.isfile(label_filename) and os.path.isfile(full_filename)):
                print('ONE OF THESE IS NOT A FILE:'+str(label_filename)+','+str(full_filename))
                self.next_idx()
            else:
                break
        im = Image.open(full_filename)
        if self.new_size:
            im = im.resize(self.new_size,Image.ANTIALIAS)

        in_ = np.array(im, dtype=np.float32)
        if in_ is None:
            logging.warning('could not get image '+full_filename)
            return None
#        print(full_filename+ ' has dims '+str(in_.shape))
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
        im = Image.open(full_filename)
        if im is None:
            print(' COULD NOT LOAD FILE '+full_filename)
            logging.warning('couldnt load file '+full_filename)
        if self.new_size:
            im = im.resize(self.new_size,Image.ANTIALIAS)

        in_ = np.array(im, dtype=np.uint8)

        if len(in_.shape) == 3:
#            logging.warning('got 3 layer img as mask, taking first layer')
            in_ = in_[:,:,0]
    #        in_ = in_ - 1
 #       print('uniques of label:'+str(np.unique(in_))+' shape:'+str(in_.shape))
 #       print(full_filename+' has dims '+str(in_.shape))
        label = copy.copy(in_[np.newaxis, ...])
#        print('after extradim shape:'+str(label.shape))

        return label

























######################################################################################3
# MULTILABEL
#######################################################################################

class JrMultilabel(caffe.Layer):
    """
    Load (input image, label vector) pairs where label vector is like [0 1 0 0 0 1 ... ]

    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        example
        layer {
            name: "data"
            type: "Python"
            top: "data"
            top: "label"
            python_param {
            module: "jrlayers"
            layer: "JrMultilabel"
            param_str: "{\'images_and_labels_file\': \'/home/jeremy/image_dbs/tamara_berg/web1\', \'mean\': (104.00699, 116.66877, 122.67892)}"
            }
        """
        # config
        params = eval(self.param_str)

        self.images_and_labels_file = params['images_and_labels_file']
        self.mean = np.array(params['mean'])
        self.images_dir = params.get('images_dir')
        self.random_init = params.get('random_initialization', True) #start from random point in image list
        self.random_pick = params.get('random_pick', True) #pick random image from list every time
        self.seed = params.get('seed', 1337)
        self.new_size = params.get('new_size',None)

        print('images+labelsfile {} mean {}'.format(self.images_and_labels_file,self.mean))
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        #if file not found and its not a path then tack on the training dir as a default locaiton for the trainingimages file
        if self.images_and_labels_file is not None:
            if not os.path.isfile(self.images_and_labels_file) and not '/' in self.images_and_labels_file:
                self.images_and_labels_file = os.path.join(self.images_dir,self.images_and_labels_file)
            if not os.path.isfile(self.images_and_labels_file):
                print('COULD NOT OPEN IMAGES/LABELS FILE '+str(self.images_and_labels_file))
                return
            self.images_and_labels_list = open(self.images_and_labels_file, 'r').read().splitlines()
            self.n_files = len(self.images_and_labels_list)
    #        self.indices = open(split_f, 'r').read().splitlines()
        else:
            print('option not supported')
#            return
#            self.imagefiles = [f for f in os.listdir(self.images_dir) if self.imagefile_suffix in f]

        self.idx = 0
        # randomization: seed and pick
        if self.random_init:
            random.seed(self.seed)
            self.idx = random.randint(0, self.n_files-1)
        if self.random_pick:
            random.shuffle(self.images_and_labels)
        logging.debug('initial self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))

        ##check that all images are openable and have labels
        ## and ge t
        good_img_files = []
        good_label_vecs = []
        print('checking image files')
        for line in self.images_and_labels_list:
            imgfilename = line.split()[0]
            img_arr = Image.open(imgfilename)
            if img_arr is not None:
                label_vec = [int(i) for i in line.split() if i>0 ]
                if label_vec is not None:
                    if len(label_vec.shape) == 1:  #got a vec
                        good_img_files.append(imgfilename)
                        good_label_vecs.append(label_vec)
                        print('got good image of size {} and label of size {}'.format(img_arr.shape,label_vec.shape))
                    else:
                        print('something wrong w. image of size {} and label of size {}'.format(img_arr.shape,label_vec.shape))
            else:
                print('got bad image:'+self.imagefiles[ind])
        self.imagefiles = good_img_files
        self.label_vecs = good_label_vecs
        assert(len(self.imagefiles) == len(self.label_vecs))
        print('{} images and {} labels'.format(len(self.imagefiles),len(self.label_vecs)))
        self.n_files = len(self.imagefiles)
        print(str(self.n_files)+' good files in image dir '+str(self.images_dir))

    def reshape(self, bottom, top):
        print('reshaping')
#	logging.debug('self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))
        imgfilename, self.data, self.label = self.load_image_and_label(self.idx)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)
        print('top 0 shape {} top 1 shape {}'.format(top[0].shape,top[1].shape))

    def next_idx(self):
        if self.random_pick:
            self.idx = random.randint(0, len(self.imagefiles)-1)
        else:
            self.idx += 1
            if self.idx == len(self.imagefiles):
                print('hit end of labels, going back to first')
                self.idx = 0

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        # pick next input
        if self.random_pick:
            self.idx = random.randint(0, len(self.imagefiles)-1)
        else:
            self.idx += 1
            if self.idx == len(self.imagefiles):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image_and_label(self,idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        while(1):
            filename = self.imagefiles[idx]
            label_vec = self.label_vecs[idx]
            if self.images_dir:
                filename=os.path.join(self.images_dir,filename)
            print('the imagefile:'+filename+' index:'+str(idx))
            if not(os.path.isfile(filename)):
                print('NOT A FILE:'+str(filename))
                self.next_idx()
            else:
                break
        im = Image.open(filename)
        if self.new_size:
            im = im.resize(self.new_size,Image.ANTIALIAS)

        in_ = np.array(im, dtype=np.float32)
        if in_ is None:
            logging.warning('could not get image '+full_filename)
            return None
#        print(full_filename+ ' has dims '+str(in_.shape))
        in_ = in_[:,:,::-1]
#        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
#	print('uniques of img:'+str(np.unique(in_))+' shape:'+str(in_.shape))
        return filename, in_, label_vec

