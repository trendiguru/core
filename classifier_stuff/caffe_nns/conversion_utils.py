__author__ = 'jeremy'

import os
import cv2
import numpy as np

from trendi import constants


def convert_pd_output(dir,converter=constants.fashionista_aug_zerobased_to_pixlevel_categories_v2,
                      input_suffix='.bmp',output_suffix='_pixlevelv2.png',for_webtool=True,
                      inlabels=constants.fashionista_categories_augmented_zero_based,
                      outlabels=constants.pixlevel_categories_v2):
    '''
    convert e..g from paperdoll to ultimate21 or pixlevel_categories_v2 .
    Optionally only convert R channel for use with webtool. Don't forget to convert back to all chans after done w webtool
    :param dir:
    :param converter:
    :param input_suffix:
    :param for_webtool:
    :return:
    '''
    files = [os.path.join(dir,f) for f in os.listdir(dir) if input_suffix in f]
    print('converting '+str(len(files))+' files in '+dir)
    for f in files:
        img_arr = cv2.imread(f)
        h,w = img_arr.shape[0:2]
        out_arr = np.zeros((h,w,3))
        for u in np.unique(img_arr):
            newindex= converter[u]
            if newindex == None:
                newindex = 0
            print('converting {} {} to {} {}'.format(u,inlabels[u],newindex,outlabels[newindex]))
            out_arr[img_arr==u] = newindex  #B it would seem this can be replaced by out_arr[:,:,:]=img_arr, maybe :: is used here
        if for_webtool:
            out_arr[:,:,0:2] = 0
        newname = f.replace(input_suffix,output_suffix)
        print('outname '+str(newname))
        cv2.imwrite(newname,out_arr)

