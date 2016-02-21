__author__ = 'jeremy'

import os
import cv2
import hashlib
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
from trendi import Utils


def image_stats_from_dir_of_dirs(dir_of_dirs):
    only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
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
        results = image_stats_from_dir(fulldir)
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
    avg_h = np.mean(hlist)
    avg_w = np.mean(wlist)
    avg_d = np.mean(dlist)
    avg_B = np.mean(Blist)
    avg_G = np.mean(Glist)
    avg_R = np.mean(Rlist)
    print('dir:{} avg of {} images: h:{} w{} d{} B {} G {} R {}'.format(dirname,n,avg_h,avg_w,avg_d,avg_B,avg_G,avg_R))
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

def resize_and_crop_image( input_file_or_np_arr, output_file=None, output_side_length = 256):
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
    resized_img = cv2.resize(img, (new_width, new_height))
    height_offset = (new_height - output_side_length) / 2
    width_offset = (new_width - output_side_length) / 2
    cropped_img = resized_img[height_offset:height_offset + output_side_length,
                              width_offset:width_offset + output_side_length]
    if output_file is not None:
        cv2.imwrite(output_file, cropped_img)
    return cropped_img


if __name__ == "__main__":
#    test_or_training_textfile('/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/only_train',test_or_train='test')
 #   test_or_training_textfile('/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/only_train',test_or_train='train')
#    Utils.remove_duplicate_files('/media/jr/Transcend/my_stuff/tg/tg_ultimate_image_db/ours/pd_output_brain1/')
    image_stats_from_dir_of_dirs(dir_of_dirs)