from __future__ import print_function

__author__ = 'jeremy'

import sys
import os
import cv2
import hashlib
import shutil
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
from trendi import Utils

def image_stats_from_dir_of_dirs(dir_of_dirs,filter=None):
    only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
    if filter is not None:
        only_dirs = [dir for dir in only_dirs if filter in dir]
    only_dirs.sort()
    hlist = []
    wlist = []
    dlist = []
    Blist = []
    Glist = []
    Rlist = []
    nlist = []
    n=0
    for a_dir in only_dirs:
        fulldir = os.path.join(dir_of_dirs,a_dir)
        print('analyzing dir '+fulldir)
        results = image_stats_from_dir(fulldir)
        if results is not None:
            hlist.append(results[0])
            wlist.append(results[1])
            dlist.append(results[2])
            Blist.append(results[3])
            Glist.append(results[4])
            Rlist.append(results[5])
            nlist.append(results[6])
            n += 1
    avg_h = np.average(hlist,weights=nlist)
    avg_w = np.average(wlist,weights=nlist)
    avg_d = np.average(dlist,weights=nlist)
    avg_B = np.average(Blist,weights=nlist)
    avg_G = np.average(Glist,weights=nlist)
    avg_R = np.average(Rlist,weights=nlist)
    totfiles = np.sum(nlist)
    print('weighted averages of {} directories: h:{} w{} d{} B {} G {} R {} totfiles {}'.format(n,avg_h,avg_w,avg_d,avg_B,avg_G,avg_R,totfiles))
    return([avg_h,avg_w,avg_d,avg_B,avg_G,avg_R,totfiles])


def image_chooser(source_dir,dest_dir,removed_dir=None):
    if removed_dir is None:
        removed_dir = os.path.join(source_dir,'removed')
    Utils.ensure_dir(removed_dir)
    print('choosing images from dir:'+str(source_dir)+', goood to '+str(dest_dir)+' and removed to '+str(removed_dir))
    only_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    for a_file in only_files:
        fullname = os.path.join(source_dir,a_file)
        img_arr = cv2.imread(fullname)
        if img_arr is not None:
            shape = img_arr.shape
            print('img:'+a_file+' shape:'+str(shape))
            print('(q)uit (d)elete (k)eep')
            cv2.imshow('img',img_arr)
            k = cv2.waitKey(0)
            if k==ord('q'):    # q to stop
                break
            elif k==ord('d'):  # normally -1 returned,so don't print it
                print('removing '+a_file+' to '+removed_dir)
                dest_fullname = os.path.join(removed_dir,a_file)
                shutil.move(fullname, dest_fullname)
            elif k== ord('k'):
                print('placing '+a_file+' in '+dest_dir)
                dest_fullname = os.path.join(dest_dir,a_file)
                shutil.move(fullname, dest_fullname)
        else:
            print('trouble opening image '+str(fullname))


def image_stats_from_dir(dirname):
    only_files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    hlist = []
    wlist = []
    dlist = []
    Blist = []
    Glist = []
    Rlist = []
    n=0
    for filename in only_files:
        fullpath = os.path.join(dirname,filename)
        results = image_stats(fullpath)
        if results is not None:
    #        print(results)
            hlist.append(results[0])
            wlist.append(results[1])
            dlist.append(results[2])
            Blist.append(results[3])
            Glist.append(results[4])
            Rlist.append(results[5])
            n += 1
            print(str(n) +' of '+str(len(only_files)), end='\r')
            sys.stdout.flush()
    avg_h = np.mean(hlist)
    avg_w = np.mean(wlist)
    avg_d = np.mean(dlist)
    avg_B = np.mean(Blist)
    avg_G = np.mean(Glist)
    avg_R = np.mean(Rlist)
    print('dir:{} avg of {} images: h:{} w{} d{} B {} G {} R {}'.format(dirname,n,avg_h,avg_w,avg_d,avg_B,avg_G,avg_R))
    if n == 0 :
        return None
    return([avg_h,avg_w,avg_d,avg_B,avg_G,avg_R,n])


def image_stats(filename):
    img_arr = cv2.imread(filename)
    if img_arr is not None:
        use_visual_output = False
        if(use_visual_output):
            cv2.imshow('current_fig',img_arr)
            cv2.waitKey(10)
        shape = img_arr.shape
        if len(shape)>2:   #BGR
            h=shape[0]
            w = shape[1]
            d = shape[2]
            avgB = np.mean(img_arr[:,:,0])
            avgG = np.mean(img_arr[:,:,1])
            avgR = np.mean(img_arr[:,:,2])
            return([h,w,d,avgB,avgG,avgR])
        else:  #grayscale /single-channel image has no 3rd dim
            h=shape[0]
            w=shape[1]
            d=1
            avgGray = np.mean(img_arr[:,:])
            return([h,w,1,avgGray,avgGray,avgGray])

    else:
        logging.warning('could not open {}'.format(filename))
        return None

def test_or_training_textfile(dir_of_dirs,test_or_train=None):
    '''
    takes dir of dirs each with different class, makes textfile suitable for training/test set
    :param dir_of_dirs:
    :return:
    '''
    only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
    only_dirs.sort()
    print(str(len(only_dirs))+' dirs:'+str(only_dirs))
    if test_or_train:
        filename = os.path.join(dir_of_dirs,test_or_train+'.txt')
    else:
        filename = os.path.join(dir_of_dirs,'fileclasses.txt')
    with open(filename,'a') as myfile:  #append , don't clobber
        classno = 0
        for dir in only_dirs:
            if (not test_or_train) or dir[0:4]==test_or_train[0:4]:
                fulldir = os.path.join(dir_of_dirs,dir)
                print('fulldir:'+str(fulldir))
                only_files = [f for f in os.listdir(fulldir) if os.path.isfile(os.path.join(fulldir, f))]
                n = len(only_files)
                print('n files {} in {}'.format(n,dir))
                for a_file in only_files:
                    line = os.path.join(dir_of_dirs,dir, a_file) + ' '+ str(classno) + '\n'
                    myfile.write(line)
                classno += 1

def resize_and_crop_image( input_file_or_np_arr, output_file=None, output_side_length = 256,use_visual_output=False):
    '''Takes an image name, resize it and crop the center square
    '''
    #TODO - implement nonsquare crop
    if isinstance(input_file_or_np_arr,basestring):
        input_file_or_np_arr = cv2.imread(input_file_or_np_arr)
    height, width, depth = input_file_or_np_arr.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = output_side_length * height / width
    else:
        new_width = output_side_length * width / height
    resized_img = cv2.resize(input_file_or_np_arr, (new_width, new_height))
    height_offset = (new_height - output_side_length) / 2
    width_offset = (new_width - output_side_length) / 2
    cropped_img = resized_img[height_offset:height_offset + output_side_length,
                              width_offset:width_offset + output_side_length]
    if use_visual_output is True:
        cv2.imshow('cropped', cropped_img)
        cv2.imshow('orig',input_file_or_np_arr)
        cv2.waitKey(0)
    if output_file is not None:
        cv2.imwrite(output_file, cropped_img)
    return cropped_img

def resize_and_crop_image_using_bb( input_file_or_np_arr, bb=None, output_file=None, output_w = 128,output_h = 128,use_visual_output=False):
    '''Takes an image name, resize it and crop the center area, keeping as much of orig as possible
    '''
    #TODO - implement nonsquare crop
    # done
    if isinstance(input_file_or_np_arr,basestring):
        orig_name = input_file_or_np_arr
        input_file_or_np_arr = cv2.imread(input_file_or_np_arr)
        if input_file_or_np_arr is None:
            logging.debug('trouble reading input file {}'.format(orig_name))
            return
        if 'bbox_' in orig_name and bb is None:
            strs = orig_name.split('bbox_')
            bb_str = strs[1]
            coords = bb_str.split('_')
            bb_x = int(coords[0])
            bb_y = int(coords[1])
            bb_w = int(coords[2])
            bb_h = coords[3].split('.')[0]  #this has .jpg or .bmp at the end
            bb_h = int(bb_h)
            bb=[bb_x,bb_y,bb_w,bb_h]
            print('bb:'+str(bb))
            if bb_h == 0:
                logging.warning('bad height encountered in imutils.resize_and_crop_image for '+str(input_file_or_np_arr))
                return None
            if bb_w == 0:
                logging.warning('bad width encountered in imutils.resize_and_crop_image for '+str(input_file_or_np_arr))
                return None

    height, width, depth = input_file_or_np_arr.shape

    if bb is None:
        bb = [0,0, width,height]
        print('no bbox given, using entire image')


    in_aspect = float(bb[2])/bb[3]
    out_aspect = float(output_w)/output_h
    x1 = bb[0]
    x2 = bb[0] + bb[2]
    y1 = bb[1]
    y2 = bb[1] + bb[3]
    if in_aspect>out_aspect:
        extra_pad_y = int((output_h*bb[2]/output_w - bb[3]) / 2)
        round = (output_h*bb[2]/output_w - bb[3]) % 2
        y1 = max(0,bb[1] - extra_pad_y)
        y2 = min(height,bb[1]+bb[3]+extra_pad_y+round)
        #print('pad y {} y1 {} y2 {}'.format(extra_pad_y,y1,y2))
    elif in_aspect<out_aspect:
        extra_pad_x = int((output_w*bb[3]/output_h - bb[2]) / 2)
        round = (output_w*bb[3]/output_h - bb[2]) % 2
        x1 = max(0,bb[0] - extra_pad_x)
        x2 = min(width,bb[0]+bb[2]+extra_pad_x+round)
        #print('pad x {} x1 {} x2 {}'.format(extra_pad_x,x1,x2))
    #print('x1 {} x2 {} y1 {} y2 {}'.format(x1,x2,y1,y2))
    cropped_img = input_file_or_np_arr[y1:y2,x1:x2,:]

    logging.debug('orig size {}x{} cropped to:{}x{},ar={} desired ar={}'.format(input_file_or_np_arr.shape[0],input_file_or_np_arr.shape[1],cropped_img.shape[0],cropped_img.shape[1],float(cropped_img.shape[1])/cropped_img.shape[0],float(output_w)/output_h))
    scaled_cropped_img = cv2.resize(cropped_img,(output_w,output_h))
#    print('resized to : {}x{}, ar={}, desired ar={}'.format(scaled_cropped_img.shape[0],scaled_cropped_img.shape[1],float(scaled_cropped_img.shape[1])/scaled_cropped_img.shape[0],float(output_w/output_h)))
    if use_visual_output is True:
        cv2.imshow('scaled_cropped', scaled_cropped_img)
        scaled_input = cv2.resize(input_file_or_np_arr,(output_w,output_h))
        cv2.imshow('orig',scaled_input)
        cv2.waitKey(0)
    if output_file is not None:
#        orig_dir = os.path.dirname(orig_name)
  #      orig_name_only = os.path.basename(orig_name)
    #    output_file = os.path.join(orig_dir,output_dir)
        print('writing to:'+output_file)
        retval = cv2.imwrite(output_file, scaled_cropped_img)
        if retval is False:
            logging.warning('retval from imwrite is false (attempt to write file:'+output_file+' has failed :(  )')
    return scaled_cropped_img

def crop_files_in_dir(dirname,save_dir,**arglist):
    '''
    takes a function that has a filename as first arg and maps it onto files in dirname
    :param func: function to map
    :param dirname: dir of files to do function on
    :param arglist: args to func
    :return:
    '''
    Utils.ensure_dir(save_dir)
    logging.debug('cropping files in directory {} with arguments {}'.format(dirname,str(arglist)))
    only_files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    for a_file in only_files:
        input_path = os.path.join(dirname,a_file)
        output_path = os.path.join(save_dir,a_file)
        arglist['output_file']=output_path
        resize_and_crop_image_using_bb(input_path,**arglist)

def crop_files_in_dir_of_dirs(dir_of_dirs,**arglist):
    '''
    takes a function that has a filename as first arg and maps it onto files in directory of directories
    :param func: function to map
    :param dir_of_dirs: dir of dirs to do function on
    :param arglist: args to func
    :return:
    '''
    logging.debug('cropping files in directories under directory {} with arguments {}'.format(dir_of_dirs,str(arglist)))
    only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
    for a_dir in only_dirs:
        fullpath = os.path.join(dir_of_dirs,a_dir)
        save_dir =  os.path.join(dir_of_dirs,'cropped/')
        save_dir =  os.path.join(save_dir,a_dir)
        Utils.ensure_dir(save_dir)
        crop_files_in_dir(fullpath,save_dir,**arglist)


if __name__ == "__main__":
#    test_or_training_textfile('/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/only_train',test_or_train='test')
 #   test_or_training_textfile('/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/only_train',test_or_train='train')
#    Utils.remove_duplicate_files('/media/jr/Transcend/my_stuff/tg/tg_ultimate_image_db/ours/pd_output_brain1/')
#    resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,170],output_w=50,output_h=50)
 #   resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,400],output_w=50,output_h=50)
  #  resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,400],output_w=150,output_h=50)
   # resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,400],output_w=50,output_h=150)
    #resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,170],output_w=1000,output_h=100)
    dir_of_dirs = '/home/jeremy/core/classifier_stuff/caffe_nns/dataset'
#    avg_h,avg_w,avg_d,avg_B,avg_G,avg_R,totfiles = image_stats_from_dir_of_dirs(dir_of_dirs,filter='test')
 #   print('avg h {} avg w {} avgB {} avgG {} avgR {} nfiles {} in dir_of_dirs {}',avg_h,avg_w,avg_d,avg_B,avg_G,avg_R,totfiles,dir_of_dirs)
#    dir_of_dirs = '/home/jr/core/classifier_stuff/caffe_nns/dataset'
#    raw_input('enter to continue')
    crop_files_in_dir_of_dirs(dir_of_dirs,bb=None,output_w =150,output_h =200,use_visual_output=True)