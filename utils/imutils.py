__author__ = 'jeremy'

import os
import cv2
import hashlib
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
from trendi import Utils

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
        print(results)
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
    print('averages of {} images: h:{} w{} d{} B {} G {} R {}'.format(n,avg_h,avg_w,avg_d,avg_B,avg_G,avg_R))
    return(n,avg_h,avg_w,avg_d,avg_B,avg_G,avg_R)

def image_stats(filename):
    img_arr = cv2.imread(filename)
    if img_arr is not None:
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



if __name__ == "__main__":
    Utils.remove_duplicate_files('/media/jr/Transcend/my_stuff/tg/tg_ultimate_image_db/ours/pd_output_brain1/')
#    image_stats_from_dir('/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/dataset/train_pairs_belts/')