# coding: utf-8
__author__ = 'jeremy'
from pylab import *
import caffe
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import socket
import lmdb
from PIL import Image
import random
import logging
import copy
import random
import shutil

from trendi.utils import imutils
from trendi import Utils
from trendi import constants

logging.basicConfig(level=logging.WARNING)

#shellscript for mean comp:
#TOOLS=/home/ubuntu/repositories/caffe/build/tools
#DATA=/home/ubuntu/AdienceFaces/lmdb/Test_fold_is_0/gender_train_lmdb
#OUT=/home/ubuntu/AdienceFaces/mean_image/Test_folder_is_0

#$TOOLS/compute_image_mean.bin $DATA $OUT/mean.binaryproto

#get_ipython().system(u'data/mnist/get_mnist.sh')
#get_ipython().system(u'examples/mnist/create_mnist.sh')

#label = L.Data(batch_size=99, backend=P.Data.LMDB, source='train_label', transform_param=dict(scale=1./255), ntop=1)
#data = L.Data(batch_size=99, backend=P.Data.LMDB, source='train_data', transform_param=dict(scale=1./255), ntop=1)

################LMDB FUN (originally) RIPPED FROM http://deepdish.io/2015/04/28/creating-lmdb-in-python/
#############changes by awesome d.j. jazzy jer  awesomest hAckz0r evarr
def db_size(dbname):
    env = lmdb.open(dbname)
    db_stats=env.stat()
    #print db_stats
    db_size = db_stats['entries']
    print('size of db {}:{}'.format(dbname,db_size))
    return db_size

def labelfile_to_lmdb(labelfile,dbname=None,max_images = None,resize=(250,250),mean=(0,0,0),resize_w_bb=True,scale=False,use_visual_output=False,shuffle=True,regression=False):
    if dbname is None:
        dbname = labelfile+'.lmdb'
    if max_images == None:
        max_images = 10**8
    print('writing to lmdb {} maximages {} resize to {} subtract mean {} scale_images {}'.format(dbname,max_images,resize,mean,scale)
    initial_only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
    initial_only_dirs.sort()
 #   print(str(len(initial_only_dirs))+' dirs:'+str(initial_only_dirs)+' in '+dir_of_dirs)
    # txn is a Transaction object
    with open(labelfile,'r') as fp:
        lines = fp.readlines()
    n_files = len(lines)
    n_pixels = resize[0]*resize[1]*n_files
    bytes_per_pixel = 3 #assuming rgb
    n_bytes = n_pixels*bytes_per_pixel
    print('n pixels {} nbytes {} files {}'.format(n_pixels,n_bytes,n_files))
    map_size = 1e13  #size of db in bytes, can also be done by 10X actual size  as in:
    map_size = n_bytes*10  #size of db in bytes, can also be done by 10X actual size
    # We need to prepare the database for the size. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single transaction.
    print('writing to db:'+dbname)
    classno = 0
    image_number =0
    n_for_each_class = []
    env = lmdb.open(dbname, map_size=map_size)
    with env.begin(write=True) as txn:
    # txn is a Transaction object
            #maybe open and close db every class to cut down on memory
            #assuming this is irrelevant and we can do this once
        if shuffle is True:
            random.shuffle(lines)
        print('n files {} in {}'.format(len(lines),labelfile))
        first_time = True
        for line in lines:
            if image_number>max_images:
                break
            file = line.split()[0]
            vals = line.split()[1:]
            if regression:
                label = [float(l) for l in vals]
            else:
                label = [int(l) for l in vals]
            if first_time:
                class_populations = np.zeros(len(label))
            if not os.path.exists(file):
                print('could not find file '+file)
                continue
            img_arr = cv2.imread(file)
            print('type of image:'+str(type(img_arr)))
            if img_arr is None:
                print('couldnt read '+file)
                continue
            h_orig=img_arr.shape[0]
            w_orig=img_arr.shape[1]
            if(resize is not None):
#                            img_arr = imutils.resize_and_crop_image(img_arr, output_side_length = resize_x)
            #    resized = imutils.resize_and_crop_image_using_bb(fullname, output_file=cropped_name,output_w=resize_x,output_h=resize_y,use_visual_output=use_visual_output)
                resized = imutils.resize_keep_aspect(img_arr,output_size=resize)
                if resized is not None:
                    img_arr = resized
                else:
                    print('resize failed')
                    continue  #didnt do good resize
            h=img_arr.shape[0]
            w=img_arr.shape[1]
            logging.debug('img {} after resize w:{} h:{} (before was {}x{} name:{}'.format(image_number, h,w,h_orig,w_orig,file))
            if use_visual_output is True:
                cv2.imshow('img',img_arr)
                cv2.waitKey(0)
            if mean is not None:  #this subtraction can prob be done in 1 step, broadcasting dimensions
                img_arr[:,:,0] = img_arr[:,:,0]-mean[0]
                img_arr[:,:,1] = img_arr[:,:,1]-mean[1]
                img_arr[:,:,2] = img_arr[:,:,2]-mean[2]
            if scale: #this will scale from -.5 to 0.5 or 0 to 1 dep. on whether mean was subtracted
                img_arr = img_arr/256
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = img_arr.shape[2]
            datum.height = img_arr.shape[0]
            datum.width = img_arr.shape[1]
#                    img_reshaped = img_arr.reshape((datum.channels,datum.height,datum.width))
#                    print('reshaped size: '+str(img_reshaped.shape))
            datum.data = img_arr.tobytes()  # or .tostring() if numpy < 1.9
            datum.label = label.tobytes()
            str_id = '{:08}'.format(image_number)  #up to 99,999,999 imgs
            print('strid:{} w:{} h:{} d:{} class:{}'.format(str_id,datum.width,datum.height,datum.channels,datum.label))
            # The encode is only essential in Python 3
            try:
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
    #            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
                image_number += 1
                for j in range(len(label)):
                    class_populations[label[j]]+=1
            except:
                e = sys.exc_info()[0]
                print('some problem with lmdb:'+str(e))
        print
        print('{} items in classes'.format(class_populations))
    env.close()
    return class_populations,image_number


def dir_of_dirs_to_lmdb(dbname,dir_of_dirs,test_or_train=None,max_images_per_class = 1000,resize_x=128,resize_y=128,avg_B=None,avg_G=None,avg_R=None,resize_w_bb=True,use_visual_output=False,shuffle=True):
    print('writing to lmdb {} test/train {} max {} new_x {} new_y {} avgB {} avg G {} avgR {}'.format(dbname,test_or_train,max_images_per_class,resize_x,resize_y,avg_B,avg_G,avg_R))
    initial_only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
    initial_only_dirs.sort()
 #   print(str(len(initial_only_dirs))+' dirs:'+str(initial_only_dirs)+' in '+dir_of_dirs)
    # txn is a Transaction object
    only_dirs = []
    for a_dir in initial_only_dirs:
        if (not test_or_train) or a_dir[0:4]==test_or_train[0:4]:
            #open and close db every class to cut down on memory
            #maybe this is irrelevant and we can do this once
            only_dirs.append(a_dir)
    print(str(len(only_dirs))+' relevant dirs:'+str(only_dirs)+' in '+dir_of_dirs)

    map_size = 1e13  #size of db in bytes, can also be done by 10X actual size  as in:
    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
#    map_size = X.nbytes * 10

    if test_or_train:
        dbname = dbname+'.'+test_or_train
    print('writing to db:'+dbname)
    classno = 0
    image_number =0
    n_for_each_class = []
    env = lmdb.open(dbname, map_size=map_size)
    with env.begin(write=True) as txn:
    # txn is a Transaction object
            #maybe open and close db every class to cut down on memory
            #assuming this is irrelevant and we can do this once
        if shuffle is True:
            random.shuffle(only_dirs)
        for a_dir in only_dirs:
            # do only test or train dirs if this param was sent
            image_number_in_class = 0
            fulldir = os.path.join(dir_of_dirs,a_dir)
            print('fulldir:'+str(fulldir))
            only_files = [f for f in os.listdir(fulldir) if os.path.isfile(os.path.join(fulldir, f))]
            n = len(only_files)
            print('n files {} in {}'.format(n,dir))
            print('maximages to do:{} of {}'.format(max_images_per_class,n))
            for n in range(0,min(max_images_per_class,n)):
                a_file =only_files[n]
                fullname = os.path.join(fulldir,a_file)
                cropped_dir= os.path.join(fulldir,'cropped')
                Utils.ensure_dir(cropped_dir)
                cropped_name= os.path.join(cropped_dir,'cropped_'+a_file)

                #img_arr = mpimg.imread(fullname)  #if you don't have cv2 handy use matplotlib
                img_arr = cv2.imread(fullname)
                if img_arr is not None:
                    h_orig=img_arr.shape[0]
                    w_orig=img_arr.shape[1]
                    if(resize_x is not None):
#                            img_arr = imutils.resize_and_crop_image(img_arr, output_side_length = resize_x)
                        resized = imutils.resize_and_crop_image_using_bb(fullname, output_file=cropped_name,output_w=resize_x,output_h=resize_y,use_visual_output=use_visual_output)
                        if resized is not None:
                            img_arr = resized
                        else:
                            print('resize failed')
                            continue  #didnt do good resize
                    h=img_arr.shape[0]
                    w=img_arr.shape[1]
                    print('img {} after resize w:{} h:{} (before was {}x{} name:{}'.format(image_number, h,w,h_orig,w_orig,fullname))
                    #    N = 1000
                    #    # Let's pretend this is interesting data
                    #    X = np.zeros((N, 3, 32, 32), dtype=np.uint8)
                     #   y = np.zeros(N, dtype=np.int64)
                    if use_visual_output is True:
                        cv2.imshow('img',img_arr)
                        cv2.waitKey(0)
                    if avg_B is not None and avg_G is not None and avg_R is not None:
                        img_arr[:,:,0] = img_arr[:,:,0]-avg_B
                        img_arr[:,:,1] = img_arr[:,:,1]-avg_G
                        img_arr[:,:,2] = img_arr[:,:,2]-avg_R
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.channels = img_arr.shape[2]
                    datum.height = img_arr.shape[0]
                    datum.width = img_arr.shape[1]
#                    img_reshaped = img_arr.reshape((datum.channels,datum.height,datum.width))
#                    print('reshaped size: '+str(img_reshaped.shape))
                    datum.data = img_arr.tobytes()  # or .tostring() if numpy < 1.9
                    datum.label = classno
                    str_id = '{:08}'.format(image_number)
                    print('strid:{} w:{} h:{} d:{} class:{}'.format(str_id,datum.width,datum.height,datum.channels,datum.label))
                    # The encode is only essential in Python 3
                    try:
                        txn.put(str_id.encode('ascii'), datum.SerializeToString())
            #            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
                        image_number += 1
                        image_number_in_class += 1
                    except:
                        e = sys.exc_info()[0]
                        print('some problem with lmdb:'+str(e))
                else:
                    print('couldnt read '+a_file)
                print
            print('{} items in class {}'.format(image_number_in_class,classno))
            classno += 1
            n_for_each_class.append(image_number_in_class)
    env.close()
    return classno, n_for_each_class,image_number

def crop_dir(dir_to_crop,resize_x,resize_y,save_cropped=False,use_bb_from_name=True):
#fix this up to make it work
    all_files = [f for f in os.listdir(dir_to_crop) if os.path.isfile(f)]
    cropped_dir = os.path.join(dir_to_crop,'cropped')
    Utils.ensure_dir(cropped_dir)

    for a_file in all_files:
        cropped_name = os.path.join(cropped_dir,'cropped_'+a_file)
        fullname = os.path.join(dir_to_crop,a_file)
        if use_bb_from_name:
            resized = imutils.resize_and_crop_image_using_bb(fullname, output_file=cropped_name,output_w=resize_x,output_h=resize_y,use_visual_output=use_visual_output)
        else:
            resized = cv2.resize(img_arr,(resize_x,resize_y))
        if resized is not None:
            img_arr = resized
        else:
            logging.warning('resize failed')
            continue  #didnt do good resize
        h=img_arr.shape[0]
        w=img_arr.shape[1]

def generate_binary_dbs(dir_of_dirs,filter='test'):
    '''
    This will generate all binary combos from a directory, e.g. dir_a,dir_b,dir_c will lead to db_a_vs_b, db_a_vs_c, db_b_vs_c
    :param dir_of_dirs:
    :param filter: required word in dirs , e.g. 'train' to limit dirs to those that have 'train' in the name
    :return:list of db locations
    '''
    if filter is not None:
        only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir)) and filter in dir]
    else:
        only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir)) ]
    only_dirs.sort()
    print(str(len(only_dirs))+' relevant dirs in '+dir_of_dirs)

    for dir1 in only_dirs:
        remaining_dirs = only_dirs.de

def interleaved_dir_of_dirs_to_lmdb(dbname,dir_of_dirs,positive_filter=None,max_images_per_class = 15000,
                                    resize_x=None,resize_y=None,write_cropped=False,
                                    avg_B=None,avg_G=None,avg_R=None,
                                    use_visual_output=False,use_bb_from_name=True,n_channels=3,binary_class_filter=None):
# maybe try randomize instead of interleave, cn use del list[index]
    print('writing to lmdb {} filter {} max {} new_x {} new_y {} avgB {} avg G {} avgR {} binfilt {}'.format(dbname,positive_filter,max_images_per_class,resize_x,resize_y,avg_B,avg_G,avg_R,binary_class_filter))
    initial_only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
    initial_only_dirs.sort()
 #   print(str(len(initial_only_dirs))+' dirs:'+str(initial_only_dirs)+' in '+dir_of_dirs)
    # txn is a Transaction object
    #prepare directories
    only_dirs = []
    for a_dir in initial_only_dirs:
        #only take 'test' or 'train' dirs, if test_or_train is specified
        if (not positive_filter or positive_filter in a_dir):
            only_dirs.append(a_dir)
    only_dirs.sort()
    print(str(len(only_dirs))+' relevant dirs in '+dir_of_dirs)
   # print only_dirs

    #prepare files
#    random.shuffle(only_dirs)  #this gets confusing as now the class labels change every time
    all_files = []
    classno = 0
    # setup db for binary (yes or no) classes - any dirs with filter word are class 0, everything else class 1
    if binary_class_filter is not None:
        n_classes = 2
        all_files.append([])  #init the empty classes with empty lists
        all_files.append([])
        for a_dir in only_dirs:
            # do only test or train dirs if this param was sent
            fulldir = os.path.join(dir_of_dirs,a_dir)
            only_files = [os.path.join(fulldir,f) for f in os.listdir(fulldir) if os.path.isfile(os.path.join(fulldir, f))]
            if binary_class_filter in a_dir: #class 0 (usu. will be positives)
                print('class 0: dir:'+str(fulldir))
                all_files[0] += only_files  #add only_files to all_files[0] (concatenates lists)
            else:               #class 1 (usu will be negatives)
                print('class 1: dir:'+str(fulldir))
                all_files[1] += only_files
    #shuffle the entries in the two classes since one (the second) is made of grouped cats
        random.shuffle(all_files[0])
        random.shuffle(all_files[1])
    #keep same number of positives and negatives
        n_positives = len(all_files[0])
        n_negatives = len(all_files[1])
        n_min = min(n_positives,n_negatives)
        n_min = min(n_min,max_images_per_class)

        all_files[0] = all_files[0][0:n_min-1]
        all_files[1] = all_files[1][0:n_min-1]
        print('positives {} negatives {} after pos {} neg {}'.format(n_positives,n_negatives,len(all_files[0]),len(all_files[1])))
# setup db for multiple classes in alphabetical order of directory
    else:
        n_classes = len(only_dirs)
        classno=0
        for a_dir in only_dirs:
            # do only test or train dirs if this param was sent
            fulldir = os.path.join(dir_of_dirs,a_dir)
            print('class:'+str(classno)+' dir:'+str(fulldir))
            only_files = [os.path.join(fulldir,f) for f in os.listdir(fulldir) if os.path.isfile(os.path.join(fulldir, f))]
            random.shuffle(only_files)
            n_min = min(len(only_files),max_images_per_class)
            truncated_files = only_files[0:n_min]
            print('len of {} is {} truncated to {}'.format(a_dir,len(only_files),len(truncated_files)))
            all_files.append(truncated_files)
            classno += 1
    print('{} classes, binary filter is {}'.format(n_classes,binary_class_filter))

    map_size = 1e13  #size of db in bytes, can also be done by 10X actual size  as in:
    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
#    map_size = X.nbytes * 10

    print('writing to db:'+dbname)
    got_image = False
    classno = -1
    image_number =0
    image_number_in_class = 0
    n_for_each_class = np.zeros(n_classes)
    env = lmdb.open(dbname, map_size=map_size)
    with env.begin(write=True) as txn:      # txn is a Transaction object
        while image_number_in_class<max_images_per_class:
    #        raw_input('enter to continue')
            classno += 1
            if classno == n_classes:
                classno = 0
                image_number_in_class += 1
                if got_image == False: #Will only be false if we got thru all classes but processed no images
                    print('no images left in any dirs')
                    break  #no images left
                got_image = False

#            a_dir = only_dirs[classno]
#            fulldir = os.path.join(dir_of_dirs,a_dir)
#            print('fulldir:'+str(fulldir))
            only_files = all_files[classno]
            n = len(only_files)
            if image_number_in_class >= n:
 #               print('reached end of images in class'+str(classno)+' which has '+str(n)+' files, skipping to next class')
                continue
   #         print('n files {} in {} current {} class {}'.format(n,a_dir,image_number_in_class,classno),end='')
            a_file =only_files[image_number_in_class]
#            fullname = os.path.join(fulldir,a_file)
            fullname = a_file
            #img_arr = mpimg.imread(fullname)  #if you don't have cv2 handy use matplotlib
            img_arr = cv2.imread(fullname)
            if img_arr is  None:
                logging.warning('could not read:'+fullname)
                continue
            h_orig=img_arr.shape[0]
            w_orig=img_arr.shape[1]
            if h_orig < constants.nn_img_minimum_sidelength or w_orig < constants.nn_img_minimum_sidelength:
                logging.warning('skipping {} due to  width {} or height {} being less than {}:'.format(fullname,w_orig,h_orig,constants.nn_img_minimum_sidelength))
                continue
            if(resize_x is not None):
                cropped_name = None
                if write_cropped is True:
                    base_dir = os.path.dirname(a_file)
                    base_name = os.path.basename(a_file)
                    cropped_dir = os.path.join(base_dir,'cropped')
                    Utils.ensure_dir(cropped_dir)
                    cropped_name = os.path.join(cropped_dir,'cropped_'+base_name)
                if use_bb_from_name:
                    resized = imutils.resize_and_crop_image_using_bb(fullname, output_file=cropped_name,output_w=resize_x,output_h=resize_y,use_visual_output=use_visual_output)
                else:
                    resized = cv2.resize(img_arr,(resize_x,resize_y))
                if resized is not None:
                    img_arr = resized
                else:
                    logging.warning('resize failed')
                    continue  #didnt do good resize
            h=img_arr.shape[0]
            w=img_arr.shape[1]
        #            print('img {} after resize w:{} h:{} (before was {}x{} name:{}'.format(image_number, h,w,h_orig,w_orig,fullname))
            if use_visual_output is True:
                cv2.imshow('img',img_arr)
                cv2.waitKey(0)
            #these pixel value offsets can be removed using caffe (in the test/train protobuf)- so currently these are None and this part is not entered
            if avg_B is not None and avg_G is not None and avg_R is not None:
                img_arr[:,:,0] = img_arr[:,:,0]-avg_B
                img_arr[:,:,1] = img_arr[:,:,1]-avg_G
                img_arr[:,:,2] = img_arr[:,:,2]-avg_R

            datum = caffe.proto.caffe_pb2.Datum()
            datum.height = img_arr.shape[0]
            datum.width = img_arr.shape[1]

            if n_channels == 1:  #for grayscale img
                datum.channels = 1
                blue_chan = img_arr[:,:,0]
                datum.data = blue_chan.tobytes()  # or .tostring() if numpy < 1.9
            else:
                datum.channels = img_arr.shape[2]

        #see https://github.com/BVLC/caffe/issues/1698 -
            #reverse order of channels  - BGR -> RGB (and vice versa)
#            img_arr = img_arr[:,:,::-1]
        # and reorder with channel first, channel x  height x width
            img_arr = img_arr.transpose((2,0,1))

#                    img_reshaped = img_arr.reshape((datum.channels,datum.height,datum.width))
#            print('reshaped size: '+str(img_arr.shape))
            datum.data = img_arr.tobytes()  # or .tostring() if numpy < 1.9
            datum.label = classno
            str_id = '{:08}'.format(image_number)
            print('db: {} strid:{} w:{} h:{} d:{} shape {} class:{} name {}'.format(dbname,str_id,datum.width,datum.height,datum.channels,img_arr.shape,datum.label,a_file)),
            # The encode is only essential in Python 3
            try:
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
    #            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
                image_number += 1
                n_for_each_class[classno] += 1
                got_image = True
            except:
                e = sys.exc_info()[0]
                logging.warning('some problem with lmdb:'+str(e))
            print
    env.close()
    return n_classes, n_for_each_class,image_number


    #You can also open up and inspect an existing LMDB database from Python:
# assuming here that dataum.data, datum.channels, datum.width etc all exist as in dir_of_dirs_to_lmdb
def label_images_and_images_to_lmdb(image_dbname,label_dbname,image_dir,label_dir,resize=None,avg_pixval=None,max_pixval=None,
                     use_visual_output=False,imgsuffix='.jpg',labelsuffix='.png',do_shuffle=False,maxfiles=1000000000,labels=None):
    '''
    this puts data images and label images into separate dbs
    :param dbname:
    :param image_dir:
    :param label_dir:
    :param resize_x:
    :param resize_y:
    :param avg_B:
    :param avg_G:
    :param avg_R:
    :param use_visual_output:
    :param imgfilter:
    :param labelsuffix:
    :param shuffle:
    :return:
    '''
# maybe try randomize instead of interleave, cn use del list[index]
    print
    print('writing to lmdb {} lbldb {} filter {} lblsuffix {} resize {} avgPixval {} max {}'.format(image_dbname,label_dbname,imgsuffix,labelsuffix,resize,avg_pixval,max_pixval))
    if imgsuffix:
        imagefiles = [f for f in os.listdir(image_dir) if imgsuffix in f]
    else:
        imagefiles = [f for f in os.listdir(image_dir)]
    imagefiles.sort()
    if do_shuffle:
        random.shuffle(imagefiles)
    imagefiles=imagefiles[0:maxfiles]
    print(str(len(imagefiles))+' relevant images in '+image_dir)
#    if shuffle:
#        print('shuffling images')
#        random.shuffle(imagefiles)  #this gets confusing as now the class labels change every time


    map_size = 1e12  #size of db in bytes, can also be done by 10X actual size  as in: map_size = X.nbytes * 10

    got_image = False
    image_number =0
    env_image = lmdb.open(image_dbname, map_size=map_size)
    env_label = lmdb.open(label_dbname, map_size=map_size)

    with env_image.begin(write=True) as txn_image:      # txn is a Transaction object
        with env_label.begin(write=True) as txn_label:      # txn is a Transaction object
            for a_file in imagefiles:
                label_file = a_file.split(imgsuffix)[0]+labelsuffix
                full_image_name = os.path.join(image_dir,a_file)
                full_label_name = os.path.join(label_dir,label_file)
                #label_name = a_file.split(imgsuffix)[0]
                #img_arr = mpimg.imread(fullname)  #if you don't have cv2 handy use matplotlib
                print('imagefile:'+full_image_name)
                print('labelfile:'+full_label_name)
                if not os.path.exists(full_image_name):
                    logging.warning('could not find image file '+full_image_name)
                    continue
                if not os.path.exists(full_label_name):
                    logging.warning('could not find label file '+full_label_name)
                    continue
                img_arr = cv2.imread(full_image_name)
                if img_arr is  None:
                    logging.warning('could not read image:'+full_image_name)
                    continue
                label_arr = cv2.imread(full_label_name,cv2.IMREAD_GRAYSCALE)
                if label_arr is  None:
                    logging.warning('could not read label:'+full_label_name)
                    continue

                h_orig=img_arr.shape[0]
                w_orig=img_arr.shape[1]
                if len(img_arr.shape) == 2:
                    n_channels = 1
                else:
                    n_channels = 3
                if h_orig < constants.nn_img_minimum_sidelength or w_orig < constants.nn_img_minimum_sidelength:
                    logging.warning('skipping {} due to  width {} or height {} being less than {}:'.format(full_image_name,w_orig,h_orig,constants.nn_img_minimum_sidelength))
                    continue
                if(resize is not None):
                    resized_image = imutils.resize_keep_aspect(img_arr, output_file=None, output_size = resize,use_visual_output=False)
                    resized_label = imutils.resize_keep_aspect(label_arr, output_file=None, output_size = resize,use_visual_output=False)
#                    resized_image = cv2.resize(img_arr,(resize_x,resize_y))
#                    resized_label = cv2.resize(label_arr,(resize_x,resize_y))
                    if resized_image is not None and resized_label is not None:
                        uniques = np.unique(label_arr)
                        resized_uniques = np.unique(resized_label)
                        print('orig uniques:'+str(uniques))
 #                       print('resized unqiues:'+str(resized_uniques))
                        print('orig bincount:'+str(np.bincount(label_arr.flatten())))
#                        print('resized bincount:'+str(np.bincount(resized_label.flatten())))
                        extras = [i for i in resized_uniques if not i in uniques]
                        for i in extras:
                            resized_label[resized_label==i] = 0
                        resized_uniques = np.unique(resized_label)
                        print('resized unqiues:'+str(resized_uniques))
                        print('resized bincount:'+str(np.bincount(resized_label.flatten())))
                        img_arr = resized_image
                        label_arr = resized_label
                        assert(img_arr.shape[0:2]==resize)
                        assert(label_arr.shape[0:2]==resize)
                        print('img shape {} lbl shape {}'.format(img_arr.shape,label_arr.shape))
                    else:
                        logging.warning('resize failed')
                        continue  #didnt do good resize
                h=img_arr.shape[0]
                w=img_arr.shape[1]
            #            print('img {} after resize w:{} h:{} (before was {}x{} name:{}'.format(image_number, h,w,h_orig,w_orig,fullname))
                if use_visual_output is True:
                    cv2.imshow('img',img_arr)
    #                cv2.imshow('label',label_arr)
                    cv2.waitKey(0)
                    imutils.show_mask_with_labels_from_img_arr(label_arr,labels=labels)
                #these pixel value offsets can be removed using caffe (in the test/train protobuf)- so currently these are None and this part is not entered
            #FORCE TYPE TO UINT8
                img_arr=img_arr.astype(np.uint8)
                print('img arr shape:'+str(img_arr.shape)+ ' type:'+str(img_arr.dtype))
                label_arr=label_arr.astype(np.uint8)
                print('label arr shape:'+str(label_arr.shape)+ ' type:'+str(label_arr.dtype))
                if avg_pixval is not None:
                    imgmean=np.average(img_arr)
                    imgstd=np.std(img_arr)
                    imgmeanBGR = [np.average(img_arr[:,:,0]),np.average(img_arr[:,:,1]),np.average(img_arr[:,:,2])]
                    print('mean {} std {} imgmeanvals {}'.format(imgmean,imgstd,imgmeanBGR))
                    img_arr[:,:,0] = img_arr[:,:,0]-avg_pixval[0]
                    img_arr[:,:,1] = img_arr[:,:,1]-avg_pixval[1]
                    img_arr[:,:,2] = img_arr[:,:,2]-avg_pixval[2]
                    imgmean=np.average(img_arr)
                    imgstd=np.std(img_arr)
                    print('after subtraction mean {} std {}'.format(imgmean,imgstd))
                if max_pixval is not None:
                    img_arr = np.divide(img_arr.astype(np.float),float(max_pixval))
                    imgmean=np.average(img_arr)
                    imgstd=np.std(img_arr)
                    print('after norm mean {} std {}'.format(imgmean,imgstd))
            ###write image
                datum = caffe.proto.caffe_pb2.Datum()
                datum.height = img_arr.shape[0]
                datum.width = img_arr.shape[1]
                if n_channels == 1:  #for grayscale img
                    datum.channels = 1
                    blue_chan = img_arr[:,:,0]  #this doesnt look like it should work
                    datum.data = blue_chan.tobytes()  # or .tostring() if numpy < 1.9
                else:
                    datum.channels = img_arr.shape[2]
                img_arr = img_arr.transpose((2,0,1))
                print('img arr shape:'+str(img_arr.shape)+ ' type:'+str(img_arr.dtype))
                datum.data = img_arr.tobytes()  # or .tostring() if numpy < 1.9
                str_id = '{:08}'.format(image_number)
                try:
                    txn_image.put(str_id.encode('ascii'), datum.SerializeToString())
        #            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
                    image_number += 1
                    got_image = True
                except:
                    e = sys.exc_info()[0]
                    logging.warning('some problem with lmdb:'+str(e))

            ###, write label
        #redoing thiws with  3 channels due to cafe complaint - 240K vs 720 K
        #obviously misguided attempt being redone:
#   F0502 10:10:28.617626 15482 softmax_loss_layer.cpp:42] Check failed: outer_num_ * inner_num_ == bottom[1]->count() (240000 vs. 720000) Number of labels must match number of predictions; e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), label count (number of labels) must be N*H*W, with integer values in {0, 1, ..., C-1}.
                print('label array shape:'+str(label_arr.shape)+' type:'+str(label_arr.dtype))
#                label_arr = label_arr - 1  #!@)(#*@! MATLAB DIE DIE
                if len(label_arr.shape) != 2:
                    print('read multichann label, taking first layer')
#                    label_arr = np.array([label_arr[:,:],label_arr[:,:],label_arr[:,:]])
                    label_arr = label_arr[:,:,0]
#                    label_arr = np.array([label_arr])
                else:
                    pass
                    #print('read singlechann label')
#                    label_arr = np.array([label_arr])
#                    label_arr = label_arr.transpose((2,0,1))
                print('db: {} strid:{} imgshape {} lblshape {} imgname {} lblname {}'.format(image_dbname,str_id,img_arr.shape,label_arr.shape,a_file,label_file))

                labeldatum = caffe.proto.caffe_pb2.Datum()
                labeldatum.channels = 1
                labeldatum.height = label_arr.shape[0]
                labeldatum.width = label_arr.shape[1]
                labeldatum.data = label_arr.tobytes()  # or .tostring() if numpy < 1.9
                try:
                    txn_label.put(str_id.encode('ascii'), labeldatum.SerializeToString())
                except:
                    e = sys.exc_info()[0]
                    logging.warning('some problem with label lmdb:'+str(e))
                print
        env_label.close()
    env_image.close()
    return image_number

def inspect_db(dbname,show_visual_output=True,B=0,G=0,R=0):
    env = lmdb.open(dbname, readonly=True)
    with env.begin() as txn:
        n=0
        while(1):
            try:
                str_id = '{:08}'.format(n)
                raw_datum = txn.get(str_id.encode('ascii'))
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(raw_datum)
                flat_x = np.fromstring(datum.data, dtype=np.uint8)

                print('db {} strid {} channels {} width {} height {} datumsize {} flatxsize {}'.format(dbname,str_id,datum.channels,datum.width,datum.height,len(raw_datum),len(flat_x)))


                orig_x = flat_x.reshape(datum.channels, datum.height, datum.width)
                if datum.channels == 3:
                    logging.debug('before transpose shape:'+str(orig_x.shape))
# as the input is transposed to c,h,w  by transpose(2,0,1) we have to undo it with transpose(1,2,0)
#h w c  transpose(2,0,1) -> c h w
#c h w  transpose(1,2,0) -> h w c
                    x = orig_x.transpose((1,2,0))
                    logging.debug('after transpose shape:'+str(x.shape))
      #              x = flat_x.reshape(datum.height, datum.width,datum.channels)
                    x[:,:,0] = x[:,:,0]+B
                    x[:,:,1] = x[:,:,1]+G
                    x[:,:,2] = x[:,:,2]+R
                elif datum.channels == 1:
   #                 print('reshaping 1 chan')
                    x = flat_x.reshape(datum.height, datum.width)
                    x[:,:] = x[:,:]+B
                y = datum.label
                print('db {} image# {} datasize {} class {} w {} h {} ch {} rawsize {} flatsize {}'.format(dbname,n,x.shape,y,datum.width,datum.height,datum.channels,len(raw_datum),len(flat_x)))

                n+=1
                if show_visual_output is True:
                    cv2.imshow(dbname,x)
                    if cv2.waitKey(0) == ord('q'):
                        break
            except:
                print('error getting record {} from db'.format(n))
                break

#    with env.begin() as txn:
 #       cursor = txn.cursor()
  #      n=0
   #     for key, value in cursor:
    #        print('img {}  class {}'.format(n,value))
#            print(key, value)
   #         n=n+1


def inspect_fcn_db(img_dbname,label_dbname,show_visual_output=True,avg_pixval=(0,0,0),max_pixval=255,labels=constants.ultimate_21,expected_size=None):
    print('looking at fcn db')
    print('imdb {} lbldb {} '.format(img_dbname,label_dbname))
    env_1 = lmdb.open(img_dbname, readonly=True)
    env_2 = lmdb.open(label_dbname, readonly=True)
    with env_1.begin() as txn1:
        with env_2.begin() as txn2:
            n=0
            while(1):
                try:
                    print('doing image db')
                    str_id = '{:08}'.format(n)
                    raw_datum = txn1.get(str_id.encode('ascii'))
   #                 print('strid {} rawdat size {}'.format(str_id,len(raw_datum)))
    #                raw_datum = txn.get(b'00000000')
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.ParseFromString(raw_datum)
             #       flat_x = np.fromstring(datum.data, dtype=np.uint8)
                    flat_x = np.fromstring(datum.data, dtype=np.float)
                    print('strid {} channels {} width {} height {} datumsize {} flatxsize {}'
                          .format(str_id,datum.channels,datum.width,datum.height,len(raw_datum),len(flat_x)))
                    assert(len(flat_x) == datum.height*datum.width*datum.channels)
                    orig_x = flat_x.reshape(datum.channels, datum.height, datum.width)
                    imgmean=np.average(orig_x)
                    imgstd=np.std(orig_x)
                    print('mean {} std {} shape {}'.format(imgmean,imgstd,orig_x.shape))
                    orig_x = np.multiply(orig_x,float(max_pixval))
                    if datum.channels == 3:
                        logging.debug('before transpose shape:'+str(orig_x.shape))
# as the input is transposed to c,h,w  by transpose(2,0,1) we have to undo it with transpose(1,2,0)    #h w c  transpose(2,0,1) -> c h w  #c h w  transpose(1,2,0) -> h w c
                        x = orig_x.transpose((1,2,0))
                        logging.debug('after transpose shape:'+str(x.shape))
          #              x = flat_x.reshape(datum.height, datum.width,datum.channels)

                        x[:,:,0] = x[:,:,0]+avg_pixval[0]
                        x[:,:,1] = x[:,:,1]+avg_pixval[1]
                        x[:,:,2] = x[:,:,2]+avg_pixval[2]
                        imgmean=np.average(x)
                        imgstd=np.std(x)
                        print('mean {} std {} shape {}'.format(imgmean,imgstd,x.shape))

                    elif datum.channels == 1:
       #                 print('reshaping 1 chan')
                        x = flat_x.reshape(datum.height, datum.width)
                        x[:,:] = x[:,:]+avg_pixval[0]

                    if expected_size:
                        print('')
                        assert(x.shape[0:2]==expected_size)
                    x=x.astype(np.uint8)
                    if show_visual_output is True:
                        cv2.imshow(img_dbname,x)
     #                   imutils.show_mask_with_labels(orig_label,constants.fashionista_categories_augmented)
                except:
                    print('error getting image {} from image db'.format(n))
                    raw_input('enter to continue')


                try:  #get label mask
                    print('doing label db')
                    str_id = '{:08}'.format(n)
    #                print('strid:{} '.format(str_id))
                 # The encode is only essential in Python 3
                 #   txn.put(str_id.encode('ascii'), datum.SerializeToString())
                    raw_datum = txn2.get(str_id.encode('ascii'))
                    print('strid {} rawdat size {}'.format(str_id,len(raw_datum)))
    #                raw_datum = txn.get(b'00000000')
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.ParseFromString(raw_datum)
                    flat_y = np.fromstring(datum.data, dtype=np.uint8)
                    print('db {} strid {} channels {} width {} height {} datumsize {} flatxsize {}'.format(label_dbname,str_id,datum.channels,datum.width,datum.height,len(raw_datum),len(flat_x)))
                    assert(len(flat_y) == datum.height*datum.width*datum.channels)
                    orig_y = flat_y.reshape(datum.channels, datum.height, datum.width)
                    if datum.channels == 3:
                        y = orig_y.transpose((1,2,0))
                        print('got a 3 chan image as label , thats not right but taking chan 0 anyway')
                        logging.debug('after transpose shape:'+str(y.shape))
          #              x = flat_x.reshape(datum.height, datum.width,datum.channels)
                        y=y[:,:,0]
                    else:
                        y = flat_y.reshape(datum.height, datum.width)
                    if expected_size:
                        assert(y.shape[0:2]==expected_size)
                    if show_visual_output is True:
                        tmpfilename = '/tmp/tmpout.bmp'
                        cv2.imwrite(tmpfilename,y)
#                        cv2.imshow(label_dbname,y)
                        imutils.show_mask_with_labels(tmpfilename,labels,visual_output=True)
                        if cv2.waitKey(0) == ord('q'):
                            break
     #                   imutils.show_mask_with_labels(orig_label,constants.fashionista_categories_augmented)
                    n+=1

                except:
                    print('error getting label {} from db'.format(n))
                    raw_input('enter to continue')



def crude_lmdb():
    in_db = lmdb.open('image-lmdb', map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(inputs):
            # load image:
            # - as np.uint8 {0, ..., 255}
            # - in BGR (switch from RGB)
            # - in Channel x Height x Width order (switch from H x W x C)
 #           im = np.array(Image.open(in_)) # or load whatever ndarray you need
  #          im = im[:,:,::-1]
   #         im = im.transpose((2,0,1))
    #        im_dat = caffe.io.array_to_datum(im)
      #      in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
            im = np.array(Image.open(in_)) # or load whatever ndarray you need
            im = im[:,:,::-1]
            im = im.transpose((2,0,1))
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
    in_db.close()

def kill_db(db_name):
    print('ABOUT TO DELETE DB '+str(db_name)+'!!!!')
    raw_input('press enter to continue or ctrl-C to not')
    in_db = lmdb.open(db_name)
    with in_db.begin(write=True) as in_txn:
        db = in_db.open_db()
        in_txn.drop(db)
        print in_txn.stat()
#
def generate_textfile_for_deconvnet(d1,d2,textfile,d1filter='.jpg',d2filter=None,maxfiles=10000000):
    '''
    textfile with imagepath, corresponding labelpath per line
    for this thing https://github.com/HyeonwooNoh/DeconvNet
    d1 - directory of imagefiles
    d2 - directory of labelfiles
    :return:
    '''

    if d1filter:
        imagefiles = [f for f in os.listdir(d1) if d1filter in f]
    else:
        imagefiles = [f for f in os.listdir(d1)]

    random.shuffle(imagefiles)
    imagefiles = imagefiles[0:maxfiles]
    print(str(len(imagefiles))+ ' imagefiles to process in '+d1)
    with open(textfile,'a') as thefile:
        for f in imagefiles:
            full_imgfile = os.path.join(d1,f)
            if d2filter == None:
                labelfile = f
            else:
                labelfile = f[:-4] + d2filter
            full_labelfile = os.path.join(d2,labelfile)
            if os.path.exists(full_imgfile) and os.path.exists(full_labelfile):
                line = full_imgfile+' '+full_labelfile+'\n'
                thefile.write(line)
            else:
                print('one of these does not exist:'+full_imgfile+','+full_labelfile)





def generate_textfile_for_binary_classifiers(test_or_train):
    sure_negatives_dict = constants.exclusion_relations

    dirs_from_cats = {'dress':'/home/jeremy/image_dbs/tamara_berg/dataset/resized_256x256/dresses_'+test_or_train,
                      'skirt':'/home/jeremy/image_dbs/tamara_berg/dataset/resized_256x256/skirts_'+test_or_train,
                      'pants':'/home/jeremy/image_dbs/tamara_berg/dataset/resized_256x256/pants_'+test_or_train,
                      'top':'/home/jeremy/image_dbs/tamara_berg/dataset/resized_256x256/tops_'+test_or_train,
                      'outerwear':'/home/jeremy/image_dbs/tamara_berg/dataset/resized_256x256/outerwear_'+test_or_train}
    more_negatives_dir = '/home/jeremy/image_dbs/doorman/irrelevant'
    for cat in sure_negatives_dict:
        textfilename = cat+'.'+test_or_train+'.txt'
#        add_dir_listing_to_caffe_textfile(textfilename,more_negatives_dir,1)
        posdir = dirs_from_cats[cat]
        add_dir_listing_to_caffe_textfile(textfilename,posdir,0)
        for neg in sure_negatives_dict[cat]:
            negdir = dirs_from_cats[neg]
            add_dir_listing_to_caffe_textfile(textfilename,negdir,1)

def add_dir_listing_to_caffe_textfile(filename,dirname,class_label,filter='.jpg'):
    print('processing dir {} into file {} with cat {}'.format(dirname,filename,class_label))
    if filter:
        files = [f for f in os.listdir(dirname) if filter in f]
    else:
        files = [f for f in os.listdir(dirname)]
    with open(filename,'a') as f:
        for filename in files:
            f.write(os.path.join(dirname,filename)+' ' + str(class_label)+'\n')

def make_negatives_dir(source_dir,dest_dir,negfile=None):
    '''
    copies images from source into dest by user choice - good eg for making negatives (e.g. all images not containing outerwear)
    :param source_dir:
    :param dest_dir:
    :return:
    '''
    Utils.ensure_dir(dest_dir)
    print('dest dir:'+dest_dir)
 #   BASE_PATH = os.getcwd()
 #   BASE_PATH2 = os.path.join(BASE_PATH, 'unknown')
 #   print('basepath:' + BASE_PATH2)
#    males = []
    dest_dir_base=os.path.basename(dest_dir)
    delete_dir = os.path.join(Utils.parent_dir(source_dir),os.path.basename(source_dir)+'_deleted')
    print('delete dir:'+delete_dir)
    Utils.ensure_dir(delete_dir)
    if not negfile:
        negfile = os.path.join(source_dir,dest_dir_base+'negs.txt')
    print('negs file:'+negfile)
    with open(negfile,'a+') as negfile_p:
        files = os.listdir(source_dir)
        files.sort()
        for f in files:
            src = os.path.join(source_dir, f)
            print('path:' + src)
            img_arr = cv2.imread(src)
            if img_arr is None:
                continue
            showsize=400.0
            if img_arr.shape[0]>showsize:
                factor = showsize/img_arr.shape[0]
                img_arr = cv2.resize(img_arr,(int(factor*img_arr.shape[1]),int(showsize)))
            if img_arr.shape[1]>showsize:
                factor = showsize/img_arr.shape[1]
                img_arr = cv2.resize(img_arr,(int(showsize), int(factor*img_arr.shape[1])))
            cv2.imshow('file', img_arr)
            a = cv2.waitKey(0)
            print a
            print('(n)ext, (c)opy, (d)elete')
            if a == ord('c') or a == ord('C'):
                dst = os.path.join(dest_dir,f)
                print('not actuall moving ' + f + ' to ' + dst)
#                shutil.move(src, dst)
                line = src+' 1\n'
                print('writing:'+line)
                negfile_p.write(line)
            elif a == ord('n') or a == ord('N'):
                print('next')
            elif a == ord('D') or a == ord('d'):
                print('delete')
                dst = os.path.join(delete_dir, f)
                print('"deleting" ' + src + ' to ' + dst)
                shutil.move(src, dst)
    negfile_p.close()


host = socket.gethostname()
print('host:'+str(host))
#
if __name__ == "__main__":
    if host == 'jr-ThinkPad-X1-Carbon':
        dir_of_dirs = '/home/jr/core/classifier_stuff/caffe_nns/dataset'
        binary_class_filter = 'dresses'
    else:
        dir_of_dirs = '/home/jeremy/core/classifier_stuff/caffe_nns/dataset/cropped'
#    dir_of_dirs = '/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/dataset'
    print('dir:'+dir_of_dirs)
#    h,w,d,B,G,R,n = imutils.image_stats_from_dir_of_ditestrs(dir_of_dirs)
    B=142
    G=151
    R=162
    B= 104.01
    G = 116.7
    R = 122.7
    resize_x=150
    resize_y=200
    resize_x=None
    resize_y=None
#    kill_db('testdb.test')
 #   kill_db('testdb.train')
    db_name = 'fcnn_fullsize_allcats'
    image_dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/test'
    label_dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels_u21'
    image_dbname='/home/jeremy/image_dbs/lmdb/images_u21_test_256x256'
    label_dbname='/home/jeremy/image_dbs/lmdb/labels_u21_test_256x256'
    label_images_and_images_to_lmdb(image_dbname,label_dbname,image_dir,label_dir,resize=(256,256),
                                    use_visual_output=False,imgsuffix='.jpg',labelsuffix='.png',do_shuffle=True,maxfiles=100000)

    raw_input('enter to continue checking db')
    inspect_fcn_db(image_dbname,label_dbname,avg_pixval=(B,G,R),show_visual_output=False)


    image_dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/train'
    label_dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels_u21'
    image_dbname='/home/jeremy/image_dbs/lmdb/images_u21_train_256x256'
    label_dbname='/home/jeremy/image_dbs/lmdb/labels_u21_train_256x256'
    label_images_and_images_to_lmdb(image_dbname,label_dbname,image_dir,label_dir,resize=(256,256),
                                    use_visual_output=False,imgsuffix='.jpg',labelsuffix='.png',do_shuffle=True,maxfiles=100000)

    raw_input('enter to continue checking db')
#    fcn_dirs_to_lmdb(db_name,image_dir,label_dir,resize_x=None,resize_y=None,avg_B=B,avg_G=G,avg_R=R,
    #                 use_visual_output=True,imgfilter='.jpg',labelsuffix='.png',shuffle=True,label_strings=constants.fashionista_categories_augmented)
    inspect_fcn_db(image_dbname,label_dbname,avg_pixval=(B,G,R),max_pixval=1.0,show_visual_output=False)

#    n_test_classes,test_populations,test_imageno = interleaved_dir_of_dirs_to_lmdb(db_name,dir_of_dirs,max_images_per_class =3000,
#                                                                                   positive_filter='test',use_visual_output=use_visual_output,
#                                                                                  n_channels=3,resize_x=resize_x,resize_y=resize_y,
#                                                                                  binary_class_filter='dresses')
#    print('n_test classes {} pops {} test_imageno {}'.format(n_test_classes,test_populations,test_imageno))

 #   n_test_classes,test_populations,test_imageno = interleaved_dir_of_dirs_to_lmdb(db_name,dir_of_dirs,max_images_per_class =13000,
 #                                                                                  positive_filter='train',use_visual_output=use_visual_output,
 #                                                                                  n_channels=3,resize_x=resize_x,resize_y=resize_y,
 #                                                                                  binary_class_filter='dresses')
 #   print(yp'n_test classes {} populationss {} test_imageno {}'.format(n_test_classes,test_populations,test_imageno))

#    n_test_classes,test_populations,test_imageno = interleaved_dir_of_dirs_to_lmdb('todel',dir_of_dirs,max_images_per_class =150000,test_or_train='test',resize_x=resize_x,resize_y=resize_y,
#                                                                                   avg_B=B,avg_G=G,avg_R=R,use_visual_output=use_visual_output,n_channels=1)
#    n_train_classes,train_populations,train_imageno = interleaved_dir_of_dirs_to_lmdb('mydb2',dir_of_dirs,max_images_per_class =150,test_or_train='train',resize_x=resize_x,resize_y=resize_y,avg_B=B,avg_G=G,avg_R=R)
   # print('{} test classes with {} files'.format(n_test_classes,test_populations))
   # print('{} train classes with {} files'.format(n_train_classes,train_populations))
#    inspect_db('highly_populated_cropped.test',show_visual_output=True,B=B,G=G,R=R)
#    inspect_db('highly_populated_cropped.train',show_visual_output=True,B=B,G=G,R=R)
   # inspect_db('mydb.train',show_visual_output=False,B=B,G=G,R=R)

#  weighted averages of 16 directories: h:1742.51040222 w1337.66435506 d3.0 B 142.492848614 G 151.617458606 R 162.580921717 totfiles 1442


#    dir_of_dirs_to_lmdb('testdb',dir_of_dirs,test_or_train='test',resize_x=128,resize_y=90,avg_B=101,avg_G=105,avg_R=123)
 #   inspect_db('testdb.test')

#    test_or_training_textfile(dir_of_dirs,test_or_train='train')
#    Utils.remove_duplicate_files('/media/jr/Transcend/my_stuff/tg/tg_ultimate_image_db/ours/pd_output_brain1/')
#    image_stats_from_dir('/home/jr/