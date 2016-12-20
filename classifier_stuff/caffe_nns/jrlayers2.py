import copy
import os
import caffe
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
import numpy as np
from PIL import Image
import cv2
import random
import string
import time
import multiprocessing
import lmdb

from trendi.utils import augment_images
from trendi.utils import imutils
from trendi import constants

class JrPixlevel(caffe.Layer):
    """
    loads images and masks for use with pixel level segmentation nets
    does augmentation on the fly
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
            layer: "JrLayer"
            param_str: "{\'images_and_labels_file\': \'train_u21_256x256.txt\', \'mean\': (104.00699, 116.66877, 122.67892)}"
            }
#            param_str: "{\'images_dir\': \'/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/train_u21_256x256\', \'labels_dir\':\'/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels_256x256/\', \'mean\': (104.00699, 116.66877, 122.67892)}"
#            }
        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.images_and_labels_file = params['images_and_labels_file']
        self.mean = np.array(params['mean'])
        self.random_init = params.get('random_initialization', True) #start from random point in image list
        self.random_pick = params.get('random_pick', True) #pick random image from list every time
        self.seed = params.get('seed', 1337)
        self.batch_size = params.get('batch_size',1)  #######Not implemented, batchsize = 1
        self.kaggle = params.get('kaggle',False)  #######Not implemented, batchsize = 1
        self.resize = params.get('resize',False)
        self.save_visual_output = params.get('save_visual_output',False)
        self.augment_images = params.get('augment',False)
        self.augment_max_angle = params.get('augment_max_angle',10)
        self.augment_max_offset_x = params.get('augment_max_offset_x',10)
        self.augment_max_offset_y = params.get('augment_max_offset_y',10)
        self.augment_max_scale = params.get('augment_max_scale',1.2)
        self.augment_max_noise_level = params.get('augment_max_noise_level',0)
        self.augment_max_blur = params.get('augment_max_blur',0)
        self.augment_do_mirror_lr = params.get('augment_do_mirror_lr',True)
        self.augment_do_mirror_ud = params.get('augment_do_mirror_ud',False)
        self.augment_crop_size = params.get('augment_crop_size',(224,224)) #
        self.augment_show_visual_output = params.get('augment_show_visual_output',False)
        self.augment_distribution = params.get('augment_distribution','uniform')
        self.n_labels = params.get('n_labels',21)

        print('##############')
        print('params coming into jrlayers2')
        print('batchsize {}\n type {}'.format(self.batch_size,type(self.batch_size)))
        print('imfile {} \nmean {}  \nrandinit {} \nrandpick {}'.format(self.images_and_labels_file, self.mean,self.random_init, self.random_pick))
        print('seed {} \nresize \n{} \nbatchsize {} \naugment \n{} \naugmaxangle {}'.format(self.seed,self.resize,self.batch_size,self.augment_images,self.augment_max_angle))
        print('augmaxdx {} \naugmaxdy {} \naugmaxscale {} \naugmaxnoise {} \naugmaxblur {}'.format(self.augment_max_offset_x,self.augment_max_offset_y,self.augment_max_scale,self.augment_max_noise_level,self.augment_max_blur))
        print('augmirrorlr {} \naugmirrorud {} \naugcrop {} \naugvis {}'.format(self.augment_do_mirror_lr,self.augment_do_mirror_ud,self.augment_crop_size,self.augment_show_visual_output))
        print('##############')

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        #if file not found and its not a path then tack on the training dir as a default locaiton for the trainingimages file
        if self.images_and_labels_file is not None:
            if os.path.isfile(self.images_and_labels_file):
                print('opening images_and_labelsfile '+str(self.images_and_labels_file))
                lines = open(self.images_and_labels_file, 'r').read().splitlines()
                self.imagefiles = [s.split()[0] for s in lines]
                self.labelfiles = [s.split()[1] for s in lines]
                self.n_files = len(self.imagefiles)
            else:
                logging.debug('COULD NOT OPEN  '+self.images_and_labels_file)
                return

#######begin vestigial code for separate images/labels files
        elif self.imagesfile is not None:
            if not os.path.isfile(self.imagesfile) and not '/' in self.imagesfile:
                self.imagesfile = os.path.join(self.images_dir,self.imagesfile)
            if not os.path.isfile(self.imagesfile):
                logging.warning('COULD NOT OPEN IMAGES FILE '+str(self.imagesfile))
            self.imagefiles = open(self.imagesfile, 'r').read().splitlines()
            self.n_files = len(self.imagefiles)

        elif self.labelsfile is not None:  #if labels flie is none then get labels from images
            if not os.path.isfile(self.labelsfile) and not '/' in self.labelsfile:
                self.labelsfile = os.path.join(self.labels_dir,self.labelsfile)
            if not os.path.isfile(self.labelsfile):
                print('COULD NOT OPEN labelS FILE '+str(self.labelsfile))
                self.labelfiles = open(self.labelsfile, 'r').read().splitlines()
###########end vestigial code

        print('found {} imagefiles and {} labelfiles'.format(len(self.imagefiles),len(self.labelfiles)))

        self.idx = 0
        # randomization: seed and pick
        if self.random_init:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.imagefiles)-1)
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
#            print(str(self.n_files)+' good files in image dir '+str(self.images_dir))

    def reshape(self, bottom, top):
        start_time=time.time()
   #     print('reshaping')
        # reshape tops to fit (leading 1 is for batch dimension)

#        self.data,self.label = self.load_image_and_mask()
        if self.batch_size == 1:
            self.data, self.label = self.load_image_and_mask()
        #add extra batch dimension
            top[0].reshape(1, *self.data.shape)
            top[1].reshape(1, *self.label.shape)
            logging.debug('batchsize 1 datasize {} labelsize {} '.format(self.data.shape,self.label.shape))
        else:
            all_data = np.zeros((self.batch_size,3,self.augment_crop_size[0],self.augment_crop_size[1]))      #Batchsizex3channelsxWxH
            all_labels = np.zeros((self.batch_size,1, self.augment_crop_size[0],self.augment_crop_size[1]) )

            multiprocess=False
            if multiprocess:
                pool = multiprocessing.Pool(4)
                output = pool.map(self.load_image_and_mask_helper, range(self.batch_size))
                for o in output:
                    all_data[i,...]=o[0]
                    all_labels[i,...]=o[1]
            else:
                for i in range(self.batch_size):
                    data, label = self.load_image_and_mask()
                    all_data[i,...]=data
                    all_labels[i,...]=label
                    self.next_idx()
            self.data = all_data
            self.label = all_labels
            #no extra dimension needed
            top[0].reshape(*self.data.shape)
            top[1].reshape(*self.label.shape)
            logging.debug('batchsize {} datasize {} labelsize {}'.format(self.batch_size,self.data.shape,self.label.shape))
        elapsed=time.time()-start_time
        print('reshape elapsed time:'+str(elapsed))

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
        self.next_idx()

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

    def load_label_image(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
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

    def load_image_and_mask_helper(self,idxs):
        #this is to alow multiprocess to send an unneeded argument , probably there is some way to multiprocess without args
        #without this hack but this works and is easy
        out1,out2=self.load_image_and_mask()
        return out1,out2

    def load_image_and_mask(self):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #add idx and allow randomness of next_idx to take care of multiprocessing case where
        #multiple threads are using this simultanesouly and dont want to get the same image
        self.next_idx()
        while(1):
            filename = self.imagefiles[self.idx]
            label_filename=self.labelfiles[self.idx]
            print('imagefile:'+filename+'\nlabelfile:'+label_filename+' index:'+str(self.idx))
            if not(os.path.isfile(label_filename) and os.path.isfile(filename)):
                print('ONE OF THESE IS NOT ACCESSIBLE:'+str(label_filename)+','+str(filename))
                self.next_idx()
                continue
                ####todo - check that the image is coming in correctly wrt color etc
            im = Image.open(filename)
            if im is None:
                logging.warning('could not get image1 '+filename)
                self.next_idx()
                continue

            if self.resize:
                im = im.resize(self.resize,Image.ANTIALIAS)
                print('resizing image')
            in_ = np.array(im, dtype=np.float32)
            in_ = in_[:,:,::-1]   #RGB -> BGR
            if in_ is None:
                logging.warning('could not get image2 '+filename)
                self.next_idx()
                continue
            """
            Load label image as 1 x height x width integer array of label indices.
            The leading singleton dimension is required by the loss.
            """
            im = Image.open(label_filename)
            if im is None:
                logging.warning('could not get label1 '+filename)
                self.next_idx()
                continue
            if self.resize:
                im = im.resize(self.resize,Image.ANTIALIAS)
                print('resizing mask')
            if im is None:
                logging.warning('couldnt load label '+label_filename)
                self.next_idx()
                continue
    #        if self.new_size:
    #            im = im.resize(self.new_size,Image.ANTIALIAS)
            label_in_ = np.array(im, dtype=np.uint8)
            if in_ is None:
                logging.warning('could not get image '+filename)
                self.next_idx()
                continue
            break  #we finally made it past all the checks
        if self.kaggle is not False:
            print('kagle image, moving 255 -> 1')
            label_in_[label_in_==255] = 1
#        in_ = in_ - 1
 #       print('uniques of label:'+str(np.unique(label_in_))+' shape:'+str(label_in_.shape))
#        print('after extradim shape:'+str(label.shape))
#        out1,out2 = augment_images.generate_image_onthefly(in_, mask_filename_or_nparray=label_in_)
        logging.debug('img/mask sizes in jrlayers2: {} and {}, cropsize {} maxangle {}'.format(in_.shape,label_in_.shape,self.augment_crop_size,self.augment_max_angle))
#        print('img/mask sizes in jrlayers2: {} and {}, cropsize {} angle {}'.format(in_.shape,label_in_.shape,self.augment_crop_size,self.augment_max_angle))

        out1, out2 = augment_images.generate_image_onthefly(in_, mask_filename_or_nparray=label_in_,
            gaussian_or_uniform_distributions=self.augment_distribution,
            max_angle = self.augment_max_angle,
            max_offset_x = self.augment_max_offset_x,max_offset_y = self.augment_max_offset_y,
            max_scale=self.augment_max_scale,
            max_noise_level=self.augment_max_noise_level,noise_type='gauss',
            max_blur=self.augment_max_blur,
            do_mirror_lr=self.augment_do_mirror_lr,
            do_mirror_ud=self.augment_do_mirror_ud,
            crop_size=self.augment_crop_size,
            show_visual_output=self.augment_show_visual_output)

        if self.save_visual_output:
            lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(30)]
            name = "".join(lst)
            cv2.imwrite(name+'.jpg',out1)
            maskname = name+'_mask.png'
            cv2.imwrite(maskname,out2)
            imutils.show_mask_with_labels(maskname,labels=constants.pixlevel_categories_v3,original_image=name+'.jpg',visual_output=False,savename=name+'_legend.jpg',save_images=True)
#        out1 = out1[:,:,::-1]   #RGB -> BGR - not necesary since this is done above (line 303)
        out1 -= self.mean  #assumes means are BGR order, not RGB
        out1 = out1.transpose((2,0,1))  #wxhxc -> cxwxh
        if len(out2.shape) == 3:
            logging.warning('got 3 layer img as mask from augment, taking first layer')
            out2 = out2[:,:,0]
        out2 = copy.copy(out2[np.newaxis, ...])
        return out1,out2



    def squared_cubed(self,x):
        return x**2,x**3

    def noargs(self):
        return 3**2,3**3

    def helpnoargs(self,X):
        r=noargs()
        return r

    def test_multi(self,bsize):
    #    bsize = 4
        pool = multiprocessing.Pool(20)
        ins = range(bsize)
    #    outs = zip(*pool.map(squared, range(bsize)))
        outs = pool.map(self.squared_cubed, ins)
        print outs
        outs = pool.map(self.helpnoargs, ins)
        print outs
        theout1=[]
        theout2=[]
        for o in outs:
            theout1.append(o[0])
            theout2.append(o[1])
        print theout1
        print theout2






























######################################################################################3
# SINGLE/MULTILABEL
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
#mandatory argument
#        self.parname = params['param_name']
#optional argument
 #       self.parname = params.get('param_name',default_value)

        self.images_and_labels_file = params.get('images_and_labels_file',None)
        self.lmdb = params.get('lmdb',None)
        self.mean = np.array(params['mean'])
        self.random_init = params.get('random_initialization', True) #start from random point in image list
        self.random_pick = params.get('random_pick', True) #pick random image from list every time
        self.seed = params.get('seed', 1337)
        self.new_size = params.get('resize',None)
        self.batch_size = params.get('batch_size',1)
        self.regression = params.get('regression',False)
        self.scale = params.get('scale',False)
        self.save_visual_output = params.get('save_visual_output',False)
        self.augment_images = params.get('augment',False)
        self.augment_max_angle = params.get('augment_max_angle',10)
        self.augment_max_offset_x = params.get('augment_max_offset_x',20)
        self.augment_max_offset_y = params.get('augment_max_offset_y',20)
        self.augment_max_scale = params.get('augment_max_scale',1.4)
        self.augment_max_noise_level = params.get('augment_max_noise_level',0)
        self.augment_max_blur = params.get('augment_max_blur',0)
        self.augment_do_mirror_lr = params.get('augment_do_mirror_lr',True)
        self.augment_do_mirror_ud = params.get('augment_do_mirror_ud',False)
        self.augment_crop_size = params.get('augment_crop_size',(224,224)) #
        self.augment_show_visual_output = params.get('augment_show_visual_output',False)
        self.augment_save_visual_output = params.get('augment_save_visual_output',False)
        self.augment_distribution = params.get('augment_distribution','uniform')
        self.n_labels = params.get('n_labels',0)  #this will obvious from the image/label file. in case of multilabel this is number of classes, i n case of single label this is 1
        self.counter = 0

        print('############net params for jrlayers2#########')
        if self.images_and_labels_file is not None:
            print('im/label file {}'.format(self.images_and_labels_file))
        if self.lmdb is not None:
            print('lmdb {}'.format(self.lmdb))
        print('mean {}  randinit {} randpick {} '.format( self.mean,self.random_init, self.random_pick))
        print('seed {} newsize {} batchsize {} augment {} augmaxangle {} '.format(self.seed,self.new_size,self.batch_size,self.augment_images,self.augment_max_angle))
        print('augmaxdx {} augmaxdy {} augmaxscale {} augmaxnoise {} augmaxblur {} '.format(self.augment_max_offset_x,self.augment_max_offset_y,self.augment_max_scale,self.augment_max_noise_level,self.augment_max_blur))
        print('augmirrorlr {} augmirrorud {} augcrop {} augvis {}'.format(self.augment_do_mirror_lr,self.augment_do_mirror_ud,self.augment_crop_size,self.augment_show_visual_output))
        print('############end of net params for jrlayers2#########')

        self.idx = 0
        self.images_processed = 0
        self.analysis_time = time.time()
        self.previous_images_processed=0
        # print('images+labelsfile {} mean {}'.format(self.images_and_labels_file,self.mean))
        # two tops: data and label
        if len(top) != 2:
            print('len of top is '+str(len(top)))
#            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
#
        # load indices for images and labels
        #if file not found and its not a path then tack on the training dir as a default locaiton for the trainingimages file
        if self.images_and_labels_file is not None:
            print('using images/labels file '+self.images_and_labels_file)
            if not os.path.isfile(self.images_and_labels_file) and not '/' in self.images_and_labels_file:
                if self.images_dir is not None:
                    self.images_and_labels_file = os.path.join(self.images_dir,self.images_and_labels_file)
            if not os.path.isfile(self.images_and_labels_file):
                print('COULD NOT OPEN IMAGES/LABELS FILE '+str(self.images_and_labels_file))
                logging.debug('COULD NOT OPEN IMAGES/LABELS FILE '+str(self.images_and_labels_file))
                return
            self.images_and_labels_list = open(self.images_and_labels_file, 'r').read().splitlines()
   #         print('imgs:'+str(self.images_and_labels_list))
            time.sleep(10)
            if self.images_and_labels_list is None or len(self.images_and_labels_list)==0:
                print('COULD NOT FIND ANYTHING IN  IMAGES/LABELS FILE '+str(self.images_and_labels_file))
                logging.debug('COULD NOT FIND ANYTHING IN IMAGES/LABELS FILE '+str(self.images_and_labels_file))
                return
            self.n_files = len(self.images_and_labels_list)
            logging.debug('images and labels file: {} n: {}'.format(self.images_and_labels_file,self.n_files))
    #        self.indices = open(split_f, 'r').read().splitlines()
    #build list of files
            good_img_files = []
            good_label_vecs = []
            for line in self.images_and_labels_list:
                imgfilename = line.split()[0]
                vals = line.split()[1:]
                self.n_labels = len(vals)
                if self.regression:
                    label_vec = [float(i) for i in vals]
                else:
                    try:
                        label_vec = [int(i) for i in vals]
                    except:
                        logging.debug('got something that coulndt be turned into a string in the following line from file '+self.images_and_labels_file)
                        logging.debug(line)
                        logging.debug('error:'+str(sys.exc_info()[0])+' , skipping line')
                        continue
                label_vec = np.array(label_vec)
                self.n_labels = len(label_vec)
                if self.n_labels == 1:
  #                  print('length 1 label')
                    label_vec = label_vec[0]
                good_img_files.append(imgfilename)
                good_label_vecs.append(label_vec)
            self.imagefiles = good_img_files
            self.label_vecs = good_label_vecs

            check_files = False
            ##check that all images are openable and have labels
            if check_files:
                print('checking image files')
                for line in self.images_and_labels_list:
                    imgfilename = line.split()[0]
                    img_arr = Image.open(imgfilename)
                    in_ = np.array(img_arr, dtype=np.float32)
                    if img_arr is not None:
                        vals = line.split()[1:]
                        label_vec = [int(i) for i in vals]
                        label_vec = np.array(label_vec)
                        self.n_labels = len(label_vec)   #the length of the label vector for multiclass data
                        if self.n_labels == 1:  #for the case of single_class data
                            label_vec = label_vec[0]    #                label_vec = label_vec[np.newaxis,...]  #this is required by loss whihc otherwise throws:
                        if label_vec is not None:
                            if len(label_vec) > 0:  #got a vec
                                good_img_files.append(imgfilename)
                                good_label_vecs.append(label_vec)
                                sys.stdout.write(spinner.next())
                                sys.stdout.flush()
                                sys.stdout.write('\b')
                          #      print('got good image of size {} and label of size {}'.format(in_.shape,label_vec.shape))
                            else:
                                print('something wrong w. image of size {} and label of size {}'.format(in_.shape,label_vec.shape))
                    else:
                        print('got bad image:'+line)

            assert(len(self.imagefiles) == len(self.label_vecs))
            #print('{} images and {} labels'.format(len(self.imagefiles),len(self.label_vecs)))
            self.n_files = len(self.imagefiles)
            print(str(self.n_files)+' good files found in '+self.images_and_labels_file)
            time.sleep(1)

    #use lmdb
        elif self.lmdb is not None:
            self.lmdb_env = lmdb.open(self.lmdb, readonly=True)
            with self.lmdb_env.begin() as self.txn:
                self.n_files = self.txn.stat()['entries']
                print('lmdb {} opened\nstat {}\nentries {}'.format(self.lmdb,self.txn.stat(),self.n_files))
                print self.n_files
            #get first dB entry to determine label size
                str_id = '{:08}'.format(0)
                raw_datum = self.txn.get(str_id.encode('ascii'))
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(raw_datum)
                flat_x = np.fromstring(datum.data, dtype=np.uint8)
                y = datum.label
#                vals = y.split() #in the meantime lmdb cant handle multilabel
                self.n_labels = 1
                print('lmdb label {} length {}'.format(y,self.n_labels))

        self.idx = 0
        # randomization: seed and pick
        if self.random_init:
            random.seed(self.seed)
            self.idx = random.randint(0, self.n_files-1)
        if self.random_pick:
            random.shuffle(self.images_and_labels_list)
        logging.debug('initial self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))

        spinner = spinning_cursor()
        logging.debug('self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))

        #if images are being augmented then dont do this resize
        if self.augment_images == False:
            if self.new_size == None:   #the old size of 227 is not actually correct, original vgg/resnet wants 224
                print(' got no size for self.newsize')
#                self.new_size = (224,224)
            top[0].reshape(self.batch_size, 3, self.new_size[0], self.new_size[1])
        else:  #only resize if specified in the param line in prototxt /resize:(h,w)
#            self.new_size=(self.augment_crop_size[0],self.augment_crop_size[1])
            top[0].reshape(self.batch_size, 3,self.augment_crop_size[0], self.augment_crop_size[0])
#            top[0].reshape(self.batch_size, 3, self.augment_crop_size[0], self.augment_crop_size[1])
#        logging.debug('doing reshape of top[0] to img size:'+str(self.new_size))
#        logging.debug('reshaping labels to '+str(self.batch_size)+'x'+str(self.n_labels))
        top[1].reshape(self.batch_size, self.n_labels)



    def reshape(self, bottom, top):
        pass
        #print('start reshape')
#        logging.debug('self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))
        if self.batch_size == 1:
            if self.lmdb is not None:
                imgfilename, self.data, self.label = self.load_image_and_label_from_lmdb()
            else:
                imgfilename, self.data, self.label = self.load_image_and_label()
            self.images_processed += 1
        else:
            if self.augment_images is True and self.augment_crop_size is not None:
                size_for_shaping=self.augment_crop_size
            elif self.new_size is not None:
                size_for_shaping=self.new_size
            all_data = np.zeros((self.batch_size,3,size_for_shaping[0],size_for_shaping[1]))
            all_labels = np.zeros((self.batch_size,self.n_labels))
            for i in range(self.batch_size):
                if self.lmdb is not None:
                    imgfilename, data, label = self.load_image_and_label_from_lmdb()
                else:
                    imgfilename, data, label = self.load_image_and_label()
                all_data[i,...]=data
                all_labels[i,...]=label
                self.next_idx()
            self.data = all_data
            self.label = all_labels
            self.previous_images_processed = self.images_processed
            self.images_processed += self.batch_size
        ## reshape tops to fit (leading 1 is for batch dimension)
 #       top[0].reshape(1, *self.data.shape)
 #       top[1].reshape(1, *self.label.shape)
 #        print('top 0 shape {} top 1 shape {}'.format(top[0].shape,top[1].shape))
 #        print('data shape {} label shape {}'.format(self.data.shape,self.label.shape))
##       the above just shows objects , top[0].shape is an object apparently

    def next_idx(self):
        if self.random_pick:
            self.idx = random.randint(0, len(self.imagefiles)-1)
            print('next idx='+str(self.idx))
        else:
            self.idx += 1
            if self.idx == len(self.imagefiles):
                print('hit end of labels, going back to first')
                self.idx = 0

    def forward(self, bottom, top):
        # assign output
        #print('forward start')
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        # pick next input
        self.next_idx()
        #print('forward end')
        self.counter += 1
   #     print('data shape {} labelshape {} label {} '.format(self.data.shape,self.label.shape,self.label))
        dt = time.time() - self.analysis_time
        dN = self.images_processed - self.previous_images_processed
        print(str(self.counter)+' fwd passes, '+str(self.images_processed)+' images processed, dN/dt='+str(round(float(dN)/dt,3)))
        self.analysis_time=time.time()

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image_and_label(self,idx=None):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - do random xforms (rotate, translate, crop, possibly noise)
        - subtract mean
        - transpose to channel x height x width order
        """
        #print('load_image_and_label start')
        while(1):
            filename = self.imagefiles[self.idx]
            label_vec = self.label_vecs[self.idx]
            if self.images_dir:
                filename=os.path.join(self.images_dir,filename)
            #print('the imagefile:'+filename+' index:'+str(idx))
            if not(os.path.isfile(filename)):
                print('NOT A FILE:'+str(filename)+' ; trying next')
                self.next_idx()   #bad file, goto next
                continue
            #print('calling augment_images with file '+filename)
#############start added code to avoid cv2.imread############
            try:
                im = Image.open(filename)
                if im is None:
                    logging.warning('jrlayers2 could not get im '+filename)
                    self.next_idx()
                    continue

                in_ = np.array(im, dtype=np.float32)
                if self.new_size is not None and (in_.shape[0] != self.new_size[0] or in_.shape[1] != self.new_size[1]):
           #         im = im.resize(self.new_size,Image.ANTIALIAS)
                    print('resizing {} from {} to {}'.format(filename, in_.shape,self.new_size))
                    in_ = imutils.resize_keep_aspect(in_,output_size=self.new_size)
##                     print('new shape '+str(in_.shape))

                if in_ is None:
                    logging.warning('jrlayers2 could not get in_ '+filename)
                    self.next_idx()
                    continue
                logging.debug('IN_ SHAPE in jrlayers2:'+str(in_.shape))
                if in_.shape[2] != 3:
                    logging.debug('got channels !=3 in jrlayers2.load_image_and_labels')
                    self.next_idx()
                    continue
            except:
                e = sys.exc_info()[0]
                logging.debug( "Error {} in jrlayers2 checking image {}".format(e,filename))
                self.next_idx()
                continue
            try:
                in_ = in_[:,:,::-1]  #RGB->BGR - since we're using PIL Image to read in .  The caffe default is BGR so at inference time images are read in as BGR
            except:
                e = sys.exc_info()[0]
                logging.debug( "Error in jrlayers2 transposing image rgb->bgr: %s" % e )
                self.next_idx()
                continue

#############end added code to avoid cv2.imread############

            out_ = augment_images.generate_image_onthefly(in_, gaussian_or_uniform_distributions=self.augment_distribution,
                max_angle = self.augment_max_angle,
                max_offset_x = self.augment_max_offset_x,max_offset_y = self.augment_max_offset_y,
                max_scale=self.augment_max_scale,
                max_noise_level=self.augment_max_noise_level,noise_type='gauss',
                max_blur=self.augment_max_blur,
                do_mirror_lr=self.augment_do_mirror_lr,
                do_mirror_ud=self.augment_do_mirror_ud,
                crop_size=self.augment_crop_size,
                show_visual_output=self.augment_show_visual_output,
                                save_visual_output=self.augment_save_visual_output)

#            out_,unused = augment_images.generate_image_onthefly(in_,mask_filename_or_nparray=in_)
#            out_ = augment_images.generate_image_onthefly(in_)

            #print('returned from augment_images')
            #im = Image.open(filename)
            #if im is None:
            #    logging.warning('could not get image '+filename)
            #    self.next_idx()
            #    idx = self.idx
            #    continue
            #if self.new_size:
            #    im = im.resize(self.new_size,Image.ANTIALIAS)
            if out_ is None:
                logging.warning('could not get image '+filename)
                self.next_idx()
                continue
            if len(out_.shape) != 3 :
                print('got strange-sized img not having 3 dimensions ('+str(out_.shape) + ') when expected shape is hxwxc (3 dimensions)')
                print('weird file:'+filename)
                self.next_idx()  #goto next
                continue

    #if there's a crop then check resultsize=cropsize.
            if self.augment_crop_size is not None and (out_.shape[0] != self.augment_crop_size[0] or out_.shape[1] != self.augment_crop_size[1]):
                    print('got strange-sized img of size '+str(out_.shape) + ' when expected cropped hxw is '+str(self.augment_crop_size))
                    print('weird file:'+filename)
                    self.next_idx()  #goto next
                    continue
    #If there's no crop but there is a resize, check resultsize=resize_size
            if self.augment_crop_size is None and self.new_size is not None and (out_.shape[0] != self.new_size[0] or out_.shape[1] != self.new_size[1]):
                    print('got strange-sized img of size '+str(out_.shape) + ' when expected resized hxw is '+str(self.new_size))
                    print('weird file:'+filename)
                    self.next_idx()  #goto next
                    continue

            if out_.shape[2] !=3 :
                print('got non-3-chan img of size '+str(out_.shape) + ' when expected n_channels is 3 '+str(self.new_size))
                print('weird file:'+filename)
                self.next_idx()  #goto next
                continue
            break #got good img after all that , get out of while

        if self.augment_save_visual_output:
            name = str(self.idx)+str(label_vec)+'jr.jpg'
            cv2.imwrite(name,out_)
            print('saving '+name)
        out_ = np.array(out_, dtype=np.float32)

        #print(str(filename) + ' has dims '+str(out_.shape)+' label:'+str(label_vec)+' idex'+str(idx))
        #todo maybe also normalize to -1:1
        out_ -= self.mean
        out_ = out_.transpose((2,0,1))  #Row Column Channel -> Channel Row Column
#	print('uniques of img:'+str(np.unique(in_))+' shape:'+str(in_.shape))
        #print('load_image_and_label end')
        return filename, out_, label_vec

    def load_image_and_label_from_lmdb(self,idx=None):
        """
        Load input image, label from lmdb and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - do random xforms (rotate, translate, crop, possibly noise)
        - subtract mean
        - transpose to channel x height x width order
        - note this currently only works with single-label info as the lmdb label is expected (by caffe)
          to be an int or long, so no way to cram a vector in there
        """
        #print('load_image_and_label start')
        while(1):
            str_id = '{:08}'.format(idx)
            raw_datum = self.txn.get(str_id.encode('ascii'))
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)
            flat_x = np.fromstring(datum.data, dtype=np.uint8) #is this right, given that this may be neg and float numbers...maybe just save as un-normalized positive  uint8s to save space
            logging.debug('db {} strid {} channels {} width {} height {} datumsize {} flatxsize {}'.format(self.lmdb,str_id,datum.channels,datum.width,datum.height,len(raw_datum),len(flat_x)))
            orig_x = flat_x.reshape(datum.channels, datum.height, datum.width)
            print('lmdb image shape {}'.format(orig_x.shape))
            if datum.channels == 3:
                logging.debug('before transpose shape:'+str(orig_x.shape))
            # as the input is transposed to c,h,w  by transpose(2,0,1) we have to undo it with transpose(1,2,0)
            #h w c  transpose(2,0,1) -> c h w
            #c h w  transpose(1,2,0) -> h w c
                x = orig_x.transpose((1,2,0)) #get hwc image
                logging.debug('after transpose shape:'+str(x.shape))
                x[:,:,0] = x[:,:,0]-self.mean[0] #add mean
                x[:,:,1] = x[:,:,1]-self.mean[1] #maybe do this after augment as it will force floatiness
                x[:,:,2] = x[:,:,2]-self.mean[2]
            elif datum.channels == 1:
#                 print('reshaping 1 chan')
                x = flat_x.reshape(datum.height, datum.width)
                x[:,:] = x[:,:]+self.mean[0]
            if self.scale:
                if self.scale==True:
                    x=x/256.0
                else:
                    x=x/self.scale
            y = datum.label
            vals = y.split()
            print('lmdb label {} length {}'.format(vals,self.n_labels))
            if self.regression:
                label_vec = [float(i) for i in vals]
            else:
                try:
                    label_vec = [int(i) for i in vals]
                except:
                    logging.debug('got something that coulndt be turned into a string in the following line from file '+self.images_and_labels_file)
                    logging.debug('error:'+str(sys.exc_info()[0])+' , skipping line')
                    self.next_idx()
                    continue
            label_vec = np.array(label_vec)
            logging.debug('label vec:'+str(label_vec))
            in_ = np.array(x, dtype=np.float32)
            if self.new_size is not None and (in_.shape[0] != self.new_size[0] or in_.shape[1] != self.new_size[1]):
       #         im = im.resize(self.new_size,Image.ANTIALIAS)
                print('resizing from {} to {}'.format(in_.shape,self.new_size))
                in_ = imutils.resize_keep_aspect(in_,output_size=self.new_size)
##                     print('new shape '+str(in_.shape))
            if in_ is None:
                logging.warning('jrlayers2 could not get in_ of idx:'+str(self.idx))
                self.next_idx()
                continue
            logging.debug('IN_ SHAPE in jrlayers2:'+str(in_.shape))
            if in_.shape[2] != 3:
                logging.debug('got channels !=3 in jrlayers2.load_image_and_labels')
                self.next_idx()
                continue
            try:
                in_ = in_[:,:,::-1]  #RGB->BGR - since we're using PIL Image to read in .  The caffe default is BGR so at inference time images are read in as BGR
            except:
                e = sys.exc_info()[0]
                logging.debug( "Error in jrlayers2 transposing image rgb->bgr: %s" % e )
                self.next_idx()
                continue
#############end added code to avoid cv2.imread############

            out_ = augment_images.generate_image_onthefly(in_, gaussian_or_uniform_distributions=self.augment_distribution,
                max_angle = self.augment_max_angle,
                max_offset_x = self.augment_max_offset_x,max_offset_y = self.augment_max_offset_y,
                max_scale=self.augment_max_scale,
                max_noise_level=self.augment_max_noise_level,noise_type='gauss',
                max_blur=self.augment_max_blur,
                do_mirror_lr=self.augment_do_mirror_lr,
                do_mirror_ud=self.augment_do_mirror_ud,
                crop_size=self.augment_crop_size,
                show_visual_output=self.augment_show_visual_output,
                                save_visual_output=self.augment_save_visual_output)
            if out_ is None:
                logging.warning('could not get augmented image idx'+str(self.idx))
                self.next_idx()
                continue
            if len(out_.shape) != 3 :
                print('got strange-sized img not having 3 dimensions ('+str(out_.shape) + ') when expected shape is hxwxc (3 dimensions)')
                print('weird file:'+str(self.idx))
                self.next_idx()  #goto next
                continue

    #if there's a crop then check resultsize=cropsize.
            if self.augment_crop_size is not None and (out_.shape[0] != self.augment_crop_size[0] or out_.shape[1] != self.augment_crop_size[1]):
                    print('got strange-sized img of size '+str(out_.shape) + ' when expected cropped hxw is '+str(self.augment_crop_size))
                    print('weird file , crop+resize idx:'+str(self.idx))
                    self.next_idx()  #goto next
                    continue
    #If there's no crop but there is a resize, check resultsize=resize_size
            if self.augment_crop_size is None and self.new_size is not None and (out_.shape[0] != self.new_size[0] or out_.shape[1] != self.new_size[1]):
                    print('got strange-sized img of size '+str(out_.shape) + ' when expected resized hxw is '+str(self.new_size))
                    print('weird file, no crop+resize idx:'+str(self.idx))
                    self.next_idx()  #goto next
                    continue

            if out_.shape[2] !=3 :
                print('got non-3-chan img of size '+str(out_.shape) + ' when expected n_channels is 3 '+str(self.new_size))
                print('weird file:'+str(self.idx))
                self.next_idx()  #goto next
                continue
            break #got good img after all that , get out of while

        if self.save_visual_output:
            name = str(self.idx)+str(label_vec)+'.jpg'
            cv2.imwrite(name,out_)
            print('saving '+name)
        out_ = np.array(out_, dtype=np.float32)

        #print(str(filename) + ' has dims '+str(out_.shape)+' label:'+str(label_vec)+' idex'+str(idx))
        #todo maybe also normalize to -1:1
        out_ -= self.mean
        out_ = out_.transpose((2,0,1))  #Row Column Channel -> Channel Row Column
#	print('uniques of img:'+str(np.unique(in_))+' shape:'+str(in_.shape))
        #print('load_image_and_label end')
        return self.idx, out_, label_vec


def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor
















































######################################################################################3
# test
#######################################################################################

class JrTestInput(caffe.Layer):
    """
    Load (input image, label vector) pairs where label vector is like [0 1 0 0 0 1 ... ]
    """

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        ## reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(bottom[0].shape)
        print('top 0 shape {} selfdata shape {}'.format(top[0].shape,bottom[0].shape))

    def next_idx(self):
        pass

    def forward(self, bottom, top):
        top[0].data = bottom[0].data
        data = top[0].data
        print('data shape:'+str(data.shape))
        firstvals = data[0,:,0,0]
        print('data first vals:'+str(firstvals))

    def backward(self, top, propagate_down, bottom):
        pass
