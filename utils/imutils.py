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
from joblib import Parallel,delayed
import multiprocessing
import socket
import copy
from trendi import constants
import matplotlib.pyplot as plt
import shutil

os.environ['REDIS_HOST']='10'
os.environ['MONGO_HOST']='10'
os.environ['REDIS_PORT']='10'
os.environ['MONGO_PORT']='10'
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

def image_chooser_dir_of_dirs(dir_of_dirs,dest_dir,removed_dir=None,filter=None,relabel_dir=None,multiple_dir=None):
    print('running images chooser source:{} dest:{} filter {}'.format(dir_of_dirs,dest_dir,filter))
    only_dirs = [d for d in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs, d))]
    if filter is not None:
        only_dirs = [d for d in only_dirs if filter in d]


    for d in only_dirs:
        actual_source = os.path.join(dir_of_dirs,d)
        actual_dest = os.path.join(dest_dir,d)

        if removed_dir is None:
            removed_dir = os.path.join(actual_source,'removed')
        if relabel_dir is None:
            relabel_dir = os.path.join(actual_source,'mislabelled')
        if multiple_dir is None:
            multiple_dir = os.path.join(actual_source,'multiple_items')

        Utils.ensure_dir(actual_dest)
        Utils.ensure_dir(removed_dir)
        Utils.ensure_dir(relabel_dir)
        Utils.ensure_dir(multiple_dir)
        image_chooser(actual_source,actual_dest,removed_dir=removed_dir,relabel_dir=relabel_dir,multiple_dir=multiple_dir)

def image_chooser(source_dir,dest_dir,removed_dir=None,relabel_dir=None,multiple_dir=None):
    print('starting image chooser source {} dest {}'.format(source_dir,dest_dir))
    if removed_dir is None:
        removed_dir = os.path.join(source_dir,'removed')
    if relabel_dir is None:
        relabel_dir = os.path.join(source_dir,'mislabelled')
    if multiple_dir is None:
        multiple_dir = os.path.join(source_dir,'multiple_items')
    Utils.ensure_dir(removed_dir)
    Utils.ensure_dir(multiple_dir)
    Utils.ensure_dir(relabel_dir)
    Utils.ensure_dir(dest_dir)
    print('choosing:'+str(source_dir)+'\ngood:'+str(dest_dir)+' \nremoved:'+str(removed_dir)+' \nreprocess:'+str(relabel_dir)+'\nmultiple:'+str(multiple_dir))
    only_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    n = len(only_files)
    if n==0:
        print('no files in '+source_dir)
        return
    i = -1
    while i < n-1 : #to allow undo need to manipulate index which doesnt work with iterator
        i = i + 1
        a_file = only_files[i]
        fullname = os.path.join(source_dir,a_file)
        img_arr = cv2.imread(fullname)
        if img_arr is not None:
            shape = img_arr.shape
            resized = img_arr
            h,w = img_arr.shape[0:2]
            if h>200:
                resized = cv2.resize(img_arr,(int((200.0*w)/h),200))
                print('h,w {},{} newh neww {},{}'.format(h,w,resized.shape[0],resized.shape[1]))
            print('img '+str(i)+' of '+str(n)+':'+a_file+' shape:'+str(shape) +' (resized to '+str(resized.shape)+')')
            print('(q)uit (d)elete (k)eep (r)elabel (m)ultiple items (u)ndo')
            cv2.imshow('img',resized)
            while(1):
                k = cv2.waitKey(0)
                    # q to stop
                if k==ord('q'):
                    print('quitting')
                    sys.exit('quitting since you pressed q')
                elif k==ord('d'):  # normally -1 returned,so don't print it
    #                print('removing '+a_file+' to '+removed_dir)
                    print('removing to '+str(removed_dir))
                    dest_fullname = os.path.join(removed_dir,a_file)
                    shutil.move(fullname, dest_fullname)
                    prev_moved_to = dest_fullname
                    prev_moved_from = fullname
                    break
                elif k== ord('k'):
    #                print('keeping '+a_file+' in '+dest_dir)
                    print('keeping in '+str(dest_dir))
                    dest_fullname = os.path.join(dest_dir,a_file)
                    shutil.move(fullname, dest_fullname)
                    prev_moved_to = dest_fullname
                    prev_moved_from = fullname
                    break
                elif k== ord('r'):
    #                print('reprocessing '+a_file+' in '+reprocess_dir)
                    print('relabel-moving to '+str(relabel_dir))
                    dest_fullname = os.path.join(relabel_dir,a_file)
                    shutil.move(fullname, dest_fullname)
                    prev_moved_to = dest_fullname
                    prev_moved_from = fullname
                    break
                elif k== ord('m'):
    #                print('reprocessing '+a_file+' in '+reprocess_dir)
                    print('multiple items-moving to '+str(multiple_dir))
                    dest_fullname = os.path.join(multiple_dir,a_file)
                    shutil.move(fullname, dest_fullname)
                    prev_moved_to = dest_fullname
                    prev_moved_from = fullname
                    break
                elif k== ord('u'):
    #                print('reprocessing '+a_file+' in '+reprocess_dir)
                    print('undo')
                    shutil.move(prev_moved_to,prev_moved_from)
                    i = i - 2
                    break
                else:
                    k = cv2.waitKey(0)
                    print('unident key')
                    #add 'back' option
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
    height_offset = int((new_height - output_side_length) / 2)
    width_offset = int((new_width - output_side_length) / 2)
    cropped_img = resized_img[height_offset:height_offset + output_side_length,
                              width_offset:width_offset + output_side_length]
    if use_visual_output is True:
        cv2.imshow('cropped', cropped_img)
        cv2.imshow('orig',input_file_or_np_arr)
        cv2.waitKey(0)
    if output_file is not None:
        cv2.imwrite(output_file, cropped_img)
    return cropped_img

def resize_to_max_sidelength(img_arr, max_sidelength=250,use_visual_output=True):
    '''
    resizes to a maximum sidelength keeping orig. aspect ratio
    :param img_arr:
    :param max_sidelength:
    :param use_visual_output:
    :return:resized image
    '''
    h,w,c = img_arr.shape
    if h>w:
        if h>max_sidelength:
            new_h = max_sidelength
            new_w = int(w*float(max_sidelength)/h)
            img_arr=cv2.resize(img_arr,(new_w,new_h))
        else:  #longest side is still under limit , show orig without resize
            pass
    else:
        if w>max_sidelength:
            new_w = max_sidelength
            new_h = int(h*float(max_sidelength)/w)
            img_arr=cv2.resize(img_arr,(new_w,new_h))
        else:  #longest side is still under limit , show orig without resize
            pass
    if (use_visual_output):
        cv2.imshow('image',img_arr)
        cv2.waitKey(0)
    return img_arr

def resize_keep_aspect_dir(dir,outdir=None,overwrite=False,output_size=(250,250),use_visual_output=False,filefilter='.jpg',careful_with_the_labels=False):
    files = [ f for f in os.listdir(dir) if filefilter in f]
    print(str(len(files))+' files in '+dir)
    for file in files:
        fullname = os.path.join(dir,file)
        if overwrite:
            newname = fullname
        else:
            if outdir:
                Utils.ensure_dir(outdir)
                newname = os.path.join(outdir,file)
            else:
                newname = file.split(filefilter)[0]+'_resized'+filefilter
                newname = os.path.join(dir,newname)
        print('infile:{} desired size:{} outfile {}'.format(fullname,output_size,newname))
        resize_keep_aspect(fullname, output_file=newname, output_size = output_size,use_visual_output=use_visual_output,careful_with_the_labels=careful_with_the_labels)

def resize_keep_aspect(input_file_or_np_arr, output_file=None, output_size = (300,200),use_visual_output=False,careful_with_the_labels=False):
    '''
    Takes an image name/arr, resize keeping aspect ratio, filling extra areas with edge values
    :param input_file_or_np_arr:
    :param output_file:name for output
    :param output_size:size of output image (height,width)
    :param use_visual_output:
    :return:
    '''

    if isinstance(input_file_or_np_arr,basestring):
        input_file_or_np_arr = cv2.imread(input_file_or_np_arr)

    if input_file_or_np_arr is None:
        logging.warning('got a bad image')
        return
    inheight, inwidth = input_file_or_np_arr.shape[0:2]
    outheight, outwidth = output_size[:]
    out_ar = float(outheight)/outwidth
    in_ar = float(inheight)/inwidth
    if len(input_file_or_np_arr.shape) == 3:
        indepth = input_file_or_np_arr.shape[2]
        output_img = np.ones([outheight,outwidth,indepth],dtype=np.uint8)
    else:
        indepth = 1
        output_img = np.ones([outheight,outwidth],dtype=np.uint8)
    print('input:{}x{}x{}'.format(inheight,inwidth,indepth))
    actual_outheight, actual_outwidth = output_img.shape[0:2]
    print('output:{}x{}'.format(actual_outheight,actual_outwidth))
    if out_ar < in_ar:  #resize height to output height and fill left/right
        factor = float(inheight)/outheight
        new_width = int(float(inwidth) / factor)
        resized_img = cv2.resize(input_file_or_np_arr, (new_width, outheight))
        print('<resize size:'+str(resized_img.shape)+' desired width:'+str(outwidth)+' orig width resized:'+str(new_width))
        width_offset = (outwidth - new_width ) / 2
        output_img[:,width_offset:width_offset+new_width] = resized_img[:,:]
        for n in range(0,width_offset):  #doing this like the below runs into a broadcast problem which could prob be solved by reshaping
#            output_img[:,0:width_offset] = resized_img[:,0]
#            output_img[:,width_offset+new_width:] = resized_img[:,-1]
            output_img[:,n] = resized_img[:,0]
            output_img[:,n+new_width+width_offset] = resized_img[:,-1]
    else:   #resize width to output width and fill top/bottom
        factor = float(inwidth)/outwidth
        new_height = int(float(inheight) / factor)
        resized_img = cv2.resize(input_file_or_np_arr, (outwidth, new_height))
        print('<resize size:'+str(resized_img.shape)+' desired height:'+str(outheight)+' orig height resized:'+str(new_height))
        height_offset = (outheight - new_height) / 2
        output_img[height_offset:height_offset+new_height,:] = resized_img[:,:]
        output_img[0:height_offset,:] = resized_img[0,:]
        output_img[height_offset+new_height:,:] = resized_img[-1,:]

    if careful_with_the_labels:
#        print('uniques in source:'+str(np.unique(input_file_or_np_arr)))
#        print('uniques in dest:'+str(np.unique(output_img)))
        for u in np.unique(output_img):
            if not u in input_file_or_np_arr:
#                print('found new val in target:'+str(u))
                output_img[output_img==u] = 0
#        print('uniques in dest:'+str(np.unique(output_img)))
        # assert((np.unique(output_img)==np.unique(input_file_or_np_arr)).all())   this fails , bool attr has no all()
    if use_visual_output is True:
        cv2.imshow('output', output_img)
        cv2.imshow('orig',input_file_or_np_arr)
#        cv2.imshow('res',resized_img)
        cv2.waitKey(0)
    if output_file is not None:
        cv2.imwrite(output_file, output_img)
    return output_img
#dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)


def resize_and_crop_maintain_bb( input_file_or_np_arr, output_file=None, output_width = 150, output_height = 200,use_visual_output=False,bb=None):
    '''Takes an image name, resize it and crop the center square
    '''
    #TODO - implement nonsquare crop
    #done
    #TODO - implement non-square resize up to maximum deformation e.g. 10% xscale=2 yscale=2.2
    if isinstance(input_file_or_np_arr,basestring):
        print('got image name '+str(input_file_or_np_arr))
        if bb is None:
            if 'bbox_' in input_file_or_np_arr:
                strs = input_file_or_np_arr.split('bbox_')
                bb_str = strs[1]
                coords = bb_str.split('_')
                bb_x = int(coords[0])
                bb_y = int(coords[1])
                bb_w = int(coords[2])
                bb_h = coords[3].split('.')[0]  #this has .jpg or .bmp at the end
                bb_h = int(bb_h)
                bb=[bb_x,bb_y,bb_w,bb_h]
                if bb_h == 0:
                    logging.warning('bad height encountered in imutils.resize_and_crop_image for '+str(input_file_or_np_arr))
                    return None
                if bb_w == 0:
                    logging.warning('bad width encountered in imutils.resize_and_crop_image for '+str(input_file_or_np_arr))
                    return None
        input_file_or_np_arr_name = input_file_or_np_arr
        input_file_or_np_arr = cv2.imread(input_file_or_np_arr)
        if input_file_or_np_arr is None:
            logging.warning('input file {} is none'.format(input_file_or_np_arr_name))
            return None
    img_height, img_width, img_depth = input_file_or_np_arr.shape

    if bb is None:
        bb = [0,0, img_width,img_height]
        print('no bbox given, using entire image')
    print('bb (x,y,w,h) {} {} {} {} image:{}x{} desired:{}x{}'.format(bb[0],bb[1],bb[2],bb[3],img_width,img_height,output_width,output_height))
    if bb[0]<0:
        logging.warning('BB x out of bounds, being reset')
        bb[0]=0
    if bb[1]<0 :
        bb[1]=0
        logging.warning('BB y out of bounds, being reset')
    if bb[0]+bb[2] > img_width:
        logging.warning('BB width out of bounds, being reset')
        bb[2]=img_width-bb[0]
    if bb[1]+bb[3] > img_height:
        logging.warning('BB height out of bounds, being reset')
        bb[3]=img_height - bb[1]

    orig_bb = copy.deepcopy(bb)
    in_aspect = float(img_width)/img_height
    out_aspect = float(output_width)/output_height
    width_out_in_ratio = float(output_width)/img_width
    height_out_in_ratio = float(output_height)/img_height
    if width_out_in_ratio > height_out_in_ratio:  #rescale by smallest amt possible
#    if abs(1-width_out_in_ratio) < abs(1-height_out_in_ratio):  #rescale by smallest amt possible
 #   if output_width >  output_height:  #rescale by smallest amt possible
        #this may be wrong when width_input>1 and height_inout<1 or vice versa
        new_width = int(width_out_in_ratio*img_width)  #should be output_width.  try round instead of int, didnt work
        new_height = int(width_out_in_ratio*img_height)  #may besomething other than output_height
        bb = np.multiply(bb,width_out_in_ratio)
        bb = [int(i) for i in bb]
        print('match width, new w,h:{},{} new bb {},{},{},{}'.format(new_width,new_height,bb[0],bb[1],bb[2],bb[3]))
        scaled_img = cv2.resize(input_file_or_np_arr,(new_width,new_height))
        y1 = bb[1]
        y2 = bb[1] + bb[3]

        height_to_crop = new_height - output_height
        output_extra_margin_over_bb = int(float(new_height-output_height )/2)
        ymin = y1 - output_extra_margin_over_bb

        print('tentative ymin '+str(ymin)+' extra margin '+str(output_extra_margin_over_bb))
        if ymin<0:
            ymin = 0
#            ymax = bb[3]
            ymax = output_height
        else:
            ymax = y2 + output_extra_margin_over_bb
            if ymax>new_height:
                ymax = new_height
#                ymin = ymax - bb[3]
                ymin = new_height-output_height
        print('new ymin,ymax:{},{}'.format(ymin,ymax))
        cropped_img = scaled_img[ymin:ymax,0:output_width,:]   #crop image
        bb[1] = bb[1]-ymin

    else:  #matching output height, width should be more than desired
        new_width = int(height_out_in_ratio*img_width)  #maybe other
        new_height = int(height_out_in_ratio*img_height)  #should be output_height
        bb = np.multiply(bb,height_out_in_ratio)
        bb = [int(i) for i in bb]
        print('match height, new w,h:{},{} new bb {},{},{},{}'.format(new_width,new_height,bb[0],bb[1],bb[2],bb[3]))
        scaled_img = cv2.resize(input_file_or_np_arr,(new_width,new_height))

        x1 = bb[0]
        x2 = bb[0] + bb[2]

        width_to_crop = new_width - output_width
        output_extra_margin_over_bb = int(float(new_width-output_width)/2)
        bb_center_x
        xmin = x1 - output_extra_margin_over_bb
        print('tentative xmin '+str(xmin)+' extra margin '+str(output_extra_margin_over_bb))
        if xmin<0:
            xmin = 0
#            xmax = bb[2]
            xmax = output_width
        else:
            xmax = x2 + output_extra_margin_over_bb

            if xmax>new_width:
                xmax = new_width
                xmin = new_width-output_width
        print('new xmin,xmax:{},{}'.format(xmin,xmax))
        cropped_img = scaled_img[0:output_height,xmin:xmax,:]
        bb[0] = bb[0]-xmin

    raw_input('enter to continue')

    if use_visual_output is True:
        cropped_copy = copy.deepcopy(cropped_img)
        cv2.rectangle(cropped_copy,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),color=[0,255,0 ])
        cv2.imshow('scaled_cropped', cropped_copy)
        orig_copy = copy.deepcopy(input_file_or_np_arr)
        cv2.rectangle(orig_copy,(orig_bb[0],orig_bb[1]),(orig_bb[0]+orig_bb[2],orig_bb[1]+orig_bb[3]),color=[0,255,0 ])

        cv2.imshow('orig',orig_copy)
        cv2.waitKey(0)
    if output_file is  None:
        if input_file_or_np_arr_name:
            output_file = orig_copy
        print('writing to:'+output_file)
        retval = cv2.imwrite(output_file, cropped_img)
        if retval is False:
             logging.warning('retval from imwrite is false (attempt to write file:'+output_file+' has failed :(  )')
    return cropped_img

def resize_and_crop_image_using_bb( input_file_or_np_arr, bb=None, output_file=None, output_w = 128,output_h = 128,use_visual_output=False):
    '''Takes an image name, resize it and crop the bb area, keeping as much of orig as possible
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
    num_cores = multiprocessing.cpu_count()
    fullpaths = []
    save_dirs = []
    for a_dir in only_dirs:
        fullpath = os.path.join(dir_of_dirs,a_dir)
        save_dir =  os.path.join(dir_of_dirs,'cropped/')
        save_dir =  os.path.join(save_dir,a_dir)
        Utils.ensure_dir(save_dir)
        fullpaths.append(fullpath)
        save_dirs.append(save_dir)
        crop_files_in_dir(fullpath,save_dir,**arglist)

# this will work if i can find how to do [x,y for x in a for y in b] 'zip' style
#     Parallel(n_jobs=num_cores)(delayed(crop_files_in_dir)(the_dir,the_path) for the_dir, the_path  in [fullpaths,save_dirs])

def kill_the_missing(sourcedir, targetdir):
    '''
    this removes anything not in the source , from the target
    :param sourcedir: has files removed relative to target
    :param targetdir: has extra files, we want to remove the extras it has relative to source
    :return:
    '''
    files_in_source = [f for f in os.listdir(sourcedir) if os.path.isfile(os.path.join(sourcedir,f))]
    files_in_target = [f for f in os.listdir(targetdir) if os.path.isfile(os.path.join(targetdir,f))]
    print('{} files in {}, {} files in {}'.format(len(files_in_source),sourcedir,len(files_in_target),targetdir))
    kill_dir = os.path.join(targetdir,'removed')
    Utils.ensure_dir(kill_dir)
    n_matched = 0
    n_killed = 0
    for a_file in files_in_target:
        if a_file in files_in_source:
            print('file {} in both dirs'.format(a_file))
            n_matched += 1
        else:
            print('file {} not matched, moving to {}'.format(a_file,kill_dir))
            shutil.move(os.path.join(targetdir,a_file), os.path.join(kill_dir,a_file))
            n_killed += 1
    print('n matched {} n killed {}'.format(n_matched,n_killed))
    files_in_source = [f for f in os.listdir(sourcedir) if os.path.isfile(os.path.join(sourcedir,f))]
    files_in_target = [f for f in os.listdir(targetdir) if os.path.isfile(os.path.join(targetdir,f))]
    print('{} files in {}, {} files in {}'.format(len(files_in_source),sourcedir,len(files_in_target),targetdir))


def find_the_common(sourcedir, targetdir):
    '''
    this removes anything not in the source , from the target
    :param sourcedir: has files removed relative to target
    :param targetdir: has extra files, we want to remove the extras it has relative to source
    :return:
    '''
    files_in_source = [f for f in os.listdir(sourcedir) if os.path.isfile(os.path.join(sourcedir,f))]
    files_in_target = [f for f in os.listdir(targetdir) if os.path.isfile(os.path.join(targetdir,f))]
    print('{} files in {}, {} files in {}'.format(len(files_in_source),sourcedir,len(files_in_target),targetdir))
    n_matched = 0
    n_not_matched = 0
    for a_file in files_in_target:
        if a_file in files_in_source:
            print('file {} in both dirs'.format(a_file))
            n_matched += 1
        else:
            print('file {} not matched'.format(a_file))
            n_not_matched += 1
    print('n matched {} n not matched {}'.format(n_matched,n_not_matched))
    files_in_source = [f for f in os.listdir(sourcedir) if os.path.isfile(os.path.join(sourcedir,f))]
    files_in_target = [f for f in os.listdir(targetdir) if os.path.isfile(os.path.join(targetdir,f))]

def oversegment(img_arr):
    image_height,image_width,image_channels = img_arr.shape
    num_superpixels = 100
    num_levels = 20
    cv2.SuperpixelSEEDS.createSuperpixelSEEDS(image_width, image_height, image_channels, num_superpixels, num_levels, use_prior = 2, histogram_bins=5, double_step = False)

def defenestrate_labels(mask,kplist):
    matches = np.ones_like(mask)
    for i in range(0,len(kplist)):
        index = kplist[i]
        nv = np.multiply(mask == index,i)
        print(nv.shape)
        matches = np.add(matches,nv)
    return matches

def defenestrate_directory(indir,outdir,filter='.png',keep_these_cats=[1,55,56,57],labels=constants.fashionista_categories_augmented):
    masklist = [f for f in os.listdir(indir) if filter in f]
#    print('masks:'+str(masklist))
#    labels = constants.pascal_context_labels
    final_labels = ['','null','hair','skin','face']
    final_labels = [labels[ind] for ind in keep_these_cats]
    final_labels[:0] = [''] #prepend
    print('final labels:'+str(final_labels))
    for mask in masklist:
        fullname = os.path.join(indir,mask)
        print('name:'+mask+' full:'+fullname)
 #       show_mask_with_labels(fullname,labels)
        mask_img = cv2.imread(fullname)
        if len(mask_img.shape)==3:
            print('fixing multichan mask')
            mask_img = mask_img[:,:,0]
        new_mask = defenestrate_labels(mask_img,keep_these_cats)
        outname = os.path.join(outdir,mask)
        cv2.imwrite(outname,new_mask)
        print('outname:'+outname+', uniques '+str(np.unique(new_mask)))
  #      show_mask_with_labels(outname,final_labels)

def concatenate_labels(mask,kplist):
    matches = np.ones_like(mask)
    first = kplist[0]
    for i in range(0,len(kplist)):
        index = kplist[i]
        nv = np.multiply(mask == index,first)
        print(nv.shape)
        matches = np.add(matches,nv)
    return matches


def resize_and_crop_maintain_bb_on_dir(dir, output_width = 150, output_height = 200,use_visual_output=True):
    only_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f))]
    print('doing resize/crop in dir '+dir)
#    print(only_files)
    for a_file in only_files:
        print('file '+a_file)
        fullfile = os.path.join(dir,a_file)
        retval = resize_and_crop_maintain_bb(fullfile, output_width = 150, output_height = 200,use_visual_output=True,bb=None)

def show_mask_with_labels_dir(dir,labels,filter=None,original_images_dir=None,original_images_dir_alt=None,cut_the_crap=False,save_images=False,visual_output=False):
    '''

    :param dir:
    :param filter:  take only images with this substring in name
    :param labels: list of test labels for categories
    :param original_images_dir: dir of image (not labels)
    :param original_images_dir_alt: alternate dir of images (to deal with test/train directories)
    :param cut_the_crap: sort images to keepers and tossers
    :return:
    '''
    if filter:
        print('using filter:'+filter)
        files = [f for f in os.listdir(dir) if filter in f]
    else:
        files = [f for f in os.listdir(dir) ]
    print(str(len(files))+ ' files to process in '+dir)
    fullpaths = [os.path.join(dir,f) for f in files]
    totfrac = 0
    fraclist=[]
    n=0
    if original_images_dir:
        original_images = ['.'.join(f.split('.')[:-1])+'.jpg' for f in files]
#        original_images = [f.split('.')[-2]+'.jpg' for f in files]
        original_fullpaths = [os.path.join(original_images_dir,f) for f in original_images]
        if original_images_dir_alt:
            original_altfullpaths = [os.path.join(original_images_dir_alt,f) for f in original_images]
        for x in range(0,len(files)):
            if os.path.exists(original_fullpaths[x]):
                 show_mask_with_labels(fullpaths[x],labels,original_image=original_fullpaths[x],cut_the_crap=cut_the_crap,save_images=save_images,visual_output=visual_output)
#                if frac is not None:
#                    fraclist.append(frac)
#                    totfrac = totfrac + frac
#                    n=n+1
            elif original_images_dir_alt and os.path.exists(original_altfullpaths[x]):
                show_mask_with_labels(fullpaths[x],labels,original_image=original_altfullpaths[x],cut_the_crap=cut_the_crap,save_images=save_images,visual_output=visual_output)
 #               if frac is not None:
 #                   fraclist.append(frac)
 ##                   totfrac = totfrac + frac
   #                 n=n+1
            else:
                logging.warning(' does not exist:'+original_fullpaths[x])
                continue

    else:
        for f in fullpaths:
            show_mask_with_labels(f,labels,cut_the_crap=cut_the_crap,save_images=save_images,visual_output=visual_output)
#            if frac is not None:
#                fraclist.append(frac)
#                totfrac = totfrac + frac
#                n=n+1
#    print('avg frac of image w nonzero pixels:'+str(totfrac/n))
    hist, bins = np.histogram(fraclist, bins=30)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width,label='nonzero pixelcount')
    plt.show()
    plt.legend()
    plt.savefig('outhist.jpg')
#    print('fraction histogram:'+str(np.histogram(fraclist,bins=20)))


def show_mask_with_labels(mask_filename,labels,original_image=None,cut_the_crap=False,save_images=False,visual_output=False):
    colormap = cv2.COLORMAP_JET
    img_arr = Utils.get_cv2_img_array(mask_filename,cv2.IMREAD_GRAYSCALE)
    if img_arr is None:
        logging.warning('trouble with file '+mask_filename)
        return
    print('file:'+mask_filename+',size'+str(img_arr.shape))
    if len(img_arr.shape) != 2:
        logging.warning('got a multichannel image, using chan 0')
        img_arr = img_arr[:,:,0]
    histo = np.histogram(img_arr,bins=len(labels)-1)
#    print('hist'+str(histo[0]))
    h,w = img_arr.shape[0:2]
    n_nonzero = np.count_nonzero(img_arr)
    n_tot = h*w
    frac = float(n_nonzero)/n_tot
    uniques = np.unique(img_arr)
    print('number of unique mask values:'+str(len(uniques))+' frac nonzero:'+str(frac) +' hxw:'+str(h)+','+str(w))
    if len(uniques)>len(labels):
        logging.warning('number of unique mask values > number of labels!!!')
        return
    # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img_array)
    maxVal = len(labels)
    max_huelevel = 160.0
    satlevel = 200
    vallevel = 200
    scaled = np.uint8(np.multiply(img_arr, max_huelevel / maxVal))
#        dest = cv2.applyColorMap(scaled,colormap)
    dest = np.zeros([h,w,3])
    dest[:,:,0] = scaled  #hue
    dest[:,:,1] = satlevel   #saturation
    dest[:,:,2] = vallevel   #value
 #   print('type:'+str(type(dest)))
    dest = dest.astype(np.uint8)
    dest = cv2.cvtColor(dest,cv2.COLOR_HSV2BGR)

    bar_height = int(float(h)/len(uniques))
    bar_width = 120
    colorbar = np.zeros([h,bar_width])
    i = 0
    print('len labels:'+str(len(labels)))
    for unique in uniques:
        if unique > len(labels):
            logging.warning('pixel value out of label range')
            continue
        colorbar[i*bar_height:i*bar_height+bar_height,:] = unique

#        cv2.putText(colorbar,labels[unique],(5,i*bar_height+bar_height/2-10),cv2.FONT_HERSHEY_PLAIN,1,[i*255/len(uniques),i*255/len(uniques),100],thickness=2)
#        cv2.putText(colorbar,labels[unique],(5,i*bar_height+bar_height/2-5),cv2.FONT_HERSHEY_PLAIN,1,[0,10,50],thickness=2)
        i=i+1

    scaled_colorbar = np.uint8(np.multiply(colorbar, max_huelevel / maxVal))
    h_colorbar,w_colorbar = scaled_colorbar.shape[0:2]
    dest_colorbar = np.zeros([h_colorbar,w_colorbar,3])
    dest_colorbar[:,:,0] = scaled_colorbar  #hue
    dest_colorbar[:,:,1] = satlevel   #saturation
    dest_colorbar[:,:,2] = vallevel  #value
    dest_colorbar = dest_colorbar.astype(np.uint8)
    dest_colorbar = cv2.cvtColor(dest_colorbar,cv2.COLOR_HSV2BGR)
 #   print('size of colrbar:'+str(dest_colorbar.shape))
 #have to do labels here to get black
    i = 0
    totpixels = h*w
    for unique in uniques:
        if unique >= len(labels):
            logging.warning('pixel value out of label range')
            continue
        pixelcount = len(img_arr[img_arr==unique])
        print('unique:'+str(unique)+':'+labels[unique]+' pixcount:'+str(pixelcount)+' fraction'+str(float(pixelcount)/totpixels))
        frac_string='{:.4f}'.format(float(pixelcount)/totpixels)
        cv2.putText(dest_colorbar,labels[unique]+' '+str(frac_string),(5,int(i*bar_height+float(bar_height)/2+5)),cv2.FONT_HERSHEY_PLAIN,1,[0,10,50],thickness=1)
        i=i+1

    #dest_colorbar = cv2.applyColorMap(scaled_colorbar, colormap)
    combined = np.zeros([h,w+w_colorbar,3],dtype=np.uint8)
    combined[:,0:w_colorbar]=dest_colorbar
    combined[:,w_colorbar:w_colorbar+w]=dest
    if original_image is not None:
        orig_arr = cv2.imread(original_image)
        if orig_arr is not None:
            print('original image:'+str(original_image))
            height, width = orig_arr.shape[:2]
            maxheight=600
            minheight=300
            if height>maxheight:  # or height < minheight:
                print('got a big one (hxw {}x{}) resizing'.format(height,width))
                newheight=(height>maxheight)*maxheight   #+(height<minheight)*minheight

                factor = float(newheight)/height
                orig_arr = cv2.resize(orig_arr,(int(round(width*factor)),int(round(height*factor))))
#                print('factor {} newsize {}'.format(factor,orig_arr.shape) )

                colorbar_h,colorbar_w = dest_colorbar.shape[0:2]
                factor = float(newheight)/colorbar_h
                dest_colorbar = cv2.resize(dest_colorbar,(int(round(colorbar_w*factor)),int(round(colorbar_h*factor))))
#                print('cbarfactor {} newsize {}'.format(factor,dest_colorbar.shape) )

                dest_h,dest_w = dest.shape[0:2]
                factor = float(newheight)/dest_h
                dest = cv2.resize(dest,(int(round(dest_w*factor)),int(round(dest_h*factor))))
#                print('maskfactor {} newsize {}'.format(factor,dest.shape) )

        #    cv2.imshow('original',orig_arr)
            colorbar_h,colorbar_w = dest_colorbar.shape[0:2]
            print('dest colorbar w {} h {} shape {}'.format(colorbar_w,colorbar_h,dest_colorbar.shape))
            dest_h,dest_w = dest.shape[0:2]
            print('dest w {} h {} shape {}'.format(dest_w,dest_h,dest.shape))
            orig_h,orig_w = orig_arr.shape[0:2]
            print('orig w {} h {} shape {}'.format(orig_w,orig_h,orig_arr.shape))
#            print('colobar size {} masksize {} imsize {}'.format(dest_colorbar.shape,dest.shape,orig_arr.shape))
            combined = np.zeros([dest_h,dest_w+orig_w+colorbar_w,3],dtype=np.uint8)
            combined[:,0:colorbar_w]=dest_colorbar
            combined[:,colorbar_w:colorbar_w+dest_w]=dest
            combined[:,colorbar_w+dest_w:]=orig_arr
            combined_h,combined_w = combined.shape[0:2]
            print('comb w {} h {} shape {}'.format(combined_w,combined_h,combined.shape))
            if combined_h<minheight:
                factor = float(minheight)/combined_h
                combined = cv2.resize(combined,(int(round(combined_w*factor)),minheight))

        else:
            logging.warning('could not get image '+original_image)
 #   cv2.imshow('map',dest)
 #   cv2.imshow('colorbar',dest_colorbar)
    relative_name = os.path.basename(mask_filename)
    if visual_output:
        cv2.imshow(relative_name,combined)
        k = cv2.waitKey(0)
    if save_images:
        outname=relative_name[:-4]  #strip '.png' or 'bmp' from name
        outname=outname+'_legend.jpg'
        full_outname=os.path.join(os.path.dirname(mask_filename),outname)
        print(full_outname)
        cv2.imwrite(full_outname,combined)

    if cut_the_crap:  #move selected to dir_removed, move rest to dir_kept
        print('(d)elete (c)lose anything else keeps')
        indir = os.path.dirname(mask_filename)
        parentdir = os.path.abspath(os.path.join(indir, os.pardir))
        curdir = os.path.split(indir)[1]
        print('in {} parent {} cur {}'.format(indir,parentdir,curdir))
        if k == ord('d'):
            newdir = curdir+'_removed'
            dest_dir = os.path.join(parentdir,newdir)
            Utils.ensure_dir(dest_dir)
            print('REMOVING moving {} to {}'.format(mask_filename,dest_dir))
            shutil.move(mask_filename,dest_dir)

        elif k == ord('c'):
            newdir = curdir+'_needwork'
            dest_dir = os.path.join(parentdir,newdir)
            Utils.ensure_dir(dest_dir)
            print('CLOSE so moving {} to {}'.format(mask_filename,dest_dir))
            shutil.move(mask_filename,dest_dir)

        else:
            newdir = curdir+'_kept'
            dest_dir = os.path.join(parentdir,newdir)
            Utils.ensure_dir(dest_dir)
            print('KEEPING moving {} to {}'.format(mask_filename,dest_dir))
            shutil.move(mask_filename,dest_dir)

    cv2.destroyAllWindows()

    return combined,frac
#        return dest

def show_mask_with_labels_from_img_arr(mask,labels):
    colormap = cv2.COLORMAP_JET
    img_arr = mask
    s = img_arr.shape
    print(s)
    if len(s) != 2:
        logging.warning('got a multichannel image, using chan 0')
        img_arr = img_arr[:,:,0]
    h,w = img_arr.shape[0:2]
    uniques = np.unique(img_arr)
    print('number of unique mask values:'+str(len(uniques)))
    if len(uniques)>len(labels):
        logging.warning('number of unique mask values > number of labels!!!')
        return
    if img_arr is not None:
        # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img_array)
        maxVal = len(labels)
        scaled = np.uint8(np.multiply(img_arr, 255.0 / maxVal))
        dest = cv2.applyColorMap(scaled,colormap)
        bar_height = int(float(h)/len(uniques))
        bar_width = 100
        colorbar = np.zeros([bar_height*len(uniques),bar_width])
        i = 0
   #     print('len labels:'+str(len(labels)))
        for unique in uniques:
            if unique > len(labels):
                logging.warning('pixel value out of label range')
                continue
            print('unique:'+str(unique)+':'+labels[unique])
            colorbar[i*bar_height:i*bar_height+bar_height,:] = unique

#        cv2.putText(colorbar,labels[unique],(5,i*bar_height+bar_height/2-10),cv2.FONT_HERSHEY_PLAIN,1,[i*255/len(uniques),i*255/len(uniques),100],thickness=2)
            cv2.putText(colorbar,labels[unique],(5,i*bar_height+bar_height/2-10),cv2.FONT_HERSHEY_PLAIN,1,[100,100,100],thickness=2)
            i=i+1

        scaled_colorbar = np.uint8(np.multiply(colorbar, 255.0 / maxVal))
        dest_colorbar = cv2.applyColorMap(scaled_colorbar, colormap)
        cv2.imshow('map',dest)
        cv2.imshow('colorbar',dest_colorbar)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def resize_dir(dir,out_dir,factor=4,filter='.jpg'):
    imfiles = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f)) and filter in f]
    for f in imfiles:
        infile = os.path.join(dir,f)
        img_arr = cv2.imread(infile)
        if filter == '.png' or filter=='.bmp' or filter == 'png' or filter == 'bmp':  #png mask is read as x*y*3 , prob. bmp too
            img_arr = img_arr[:,:,0]
        h, w = img_arr.shape[0:2]
        new_h = int(h/factor)
        new_w = int(w/factor)
        output_arr = cv2.resize(img_arr,(new_w,new_h))
        actualh,actualw = output_arr.shape[0:2]
        outfile = os.path.join(out_dir,f)
        cv2.imwrite(outfile,output_arr)
        print('orig w,h {},{} new {},{} '.format(w,h,actualw,actualh))
        print('infile {} outfile {}'.format(infile,outfile))


def nms_detections(dets, overlap=0.3):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detection.

    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    dets: ndarray
        each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
    overlap: float
        minimum overlap ratio (0.3 default)

    Output
    ------
    dets: ndarray
        remaining after suppression.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    ind = np.argsort(dets[:, 4])

    w = x2 - x1
    h = y2 - y1
    area = (w * h).astype(float)

    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]

        xx1 = np.maximum(x1[i], x1[ind])
        yy1 = np.maximum(y1[i], y1[ind])
        xx2 = np.minimum(x2[i], x2[ind])
        yy2 = np.minimum(y2[i], y2[ind])

        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)

        wh = w * h
        o = wh / (area[i] + area[ind] - wh)

        ind = ind[np.nonzero(o <= overlap)[0]]

    return dets[pick, :]

def img_dir_to_html(img_dir,filter='.jpg',htmlname=None):
    imglist = [i for i in os.listdir(img_dir) if filter in i]
    line_no=0
    lines=[]

    if htmlname is None:
        parentdir = os.path.abspath(os.path.join(img_dir, os.pardir))
        htmlname=parentdir+'.html'
        htmlname=img_dir.replace('/','_')+'.html'
        htmlname=img_dir.replace('/','')+'.html'
    with open(htmlname,'w') as f:
        lines.append('<HTML><HEAD><TITLE>results '+img_dir+' </TITLE></HEAD>\n')
        for img in imglist:
            f.write('<br>\n')
            link = '"'+os.path.join(img_dir,img)+'"'
            f.write('<img src='+link+'>')
            #f.write('<a href='+link+'>'+img+'</a>\n')
        f.write('</HTML>\n')
        f.close()



host = socket.gethostname()
print('host:'+str(host))

if __name__ == "__main__":
#    test_or_training_textfile('/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/only_train',test_or_train='test')
 #   test_or_training_textfile('/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/only_train',test_or_train='train')
#    Utils.remove_duplicate_files('/media/jr/Transcend/my_stuff/tg/tg_ultimate_image_db/ours/pd_output_brain1/')
#    resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,170],output_w=50,output_h=50)
 #   resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,400],output_w=50,output_h=50)
  #  resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,400],output_w=150,output_h=50)
   # resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,400],output_w=50,output_h=150)
    #resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,170],output_w=1000,output_h=100)
#    avg_h,avg_w,avg_d,avg_B,avg_G,avg_R,totfiles = image_stats_from_dir_of_dirs(dir_of_dirs,filter='test')
 #   print('avg h {} avg w {} avgB {} avgG {} avgR {} nfiles {} in dir_of_dirs {}',avg_h,avg_w,avg_d,avg_B,avg_G,avg_R,totfiles,dir_of_dirs)
#    dir_of_dirs = '/home/jr/core/classifier_stuff/caffe_nns/dataset'
#    raw_input('enter to continue')
  #  image_chooser_dir_of_dirs(dir_of_dirs,output_dir)
#    image_chooser(dir_of_dirs,output_dir)
#    crop_files_in_dir_of_dirs(dir_of_dirs,bb=None,output_w =150,output_h =200,use_visual_output=True)
#        dir = '/home/jeremy/projects/core/images'
#        resize_and_crop_maintain_bb_on_dir(dir, output_width = 448, output_height = 448,use_visual_output=True)
    dir = '/home/jeremy/tg/pd_output'
    dir = '/root'
    indir = '/home/jeremy/image_dbs/fashionista-v0.2.1'
    outdir = '/home/jeremy/image_dbs/fashionista-v0.2.1/reduced_cats'

    indir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels_200x150'
    outdir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels_200x150/reduced_cats'
#    defenestrate_directory(indir,outdir,filter='.png',keep_these_cats=[1,55,56,57],labels=constants.fashionista_categories_augmented)

    if host == 'jr-ThinkPad-X1-Carbon' or host == 'jr':
        dir_of_dirs = '/home/jeremy/tg/train_pairs_dresses'
        output_dir = '/home/jeremy/tg/curated_train_pairs_dresses'
        sourcedir = '/home/jeremy/projects/core/d1'
        targetdir = '/home/jeremy/projects/core/d2'
        infile =  '/home/jeremy/projects/core/images/female1.jpg'
    else:
        dir_of_dirs = '/home/jeremy/core/classifier_stuff/caffe_nns/dataset/cropped'
        output_dir = '/home/jeremy/core/classifier_stuff/caffe_nns/curated_dataset'

 #   kill_the_missing(sourcedir, targetdir)
    resize_keep_aspect(infile, output_file=None, output_size = (300,200),use_visual_output=True)



#nonlinear xforms , stolen from:
#https://www.kaggle.com/bguberfain/ultrasound-nerve-segmentation/elastic-transform-for-data-augmentation/comments
'''
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
In [3]:
# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))

# Load images
im = cv2.imread("../input/train/10_1.tif", -1)
im_mask = cv2.imread("../input/train/10_1_mask.tif", -1)

# Draw grid lines
draw_grid(im, 50)
draw_grid(im_mask, 50)

# Merge images into separete channels (shape will be (cols, rols, 2))
im_merge = np.concatenate((im[...,None], im_mask[...,None]), axis=2)
In [4]:
# First sample...

%matplotlib inline

# Apply transformation on image
im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)

# Split image and mask
im_t = im_merge_t[...,0]
im_mask_t = im_merge_t[...,1]

# Display result
plt.figure(figsize = (16,14))
plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')


# Second sample (heavyer transform)...

%matplotlib inline

# Apply transformation on image
im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3, im_merge.shape[1] * 0.07, im_merge.shape[1] * 0.09)

# Split image and mask
im_t = im_merge_t[...,0]
im_mask_t = im_merge_t[...,1]

# Display result
plt.figure(figsize = (16,14))
plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')

 managed to get about 4x improvement by using:

# include 4 standard deviations in the kernel (the default for ndimage.gaussian_filter)
# OpenCV also requires an odd size for the kernel hence the "| 1" part
blur_size = int(4*sigma) | 1
cv2.GaussianBlur(image, ksize=(blur_size, blur_size), sigmaX=sigma)
instead of ndimage.gaussian_filter(image, sigma)

and cv2.remap(image, dx, dy, interpolation=cv2.INTER_LINEAR) instead of ndimage.map_coordinates(image, (dx, dy), order=1)

    resize_keep_aspect(infile, output_file=None, output_size = (300,200),use_visual_output=True)
'''
