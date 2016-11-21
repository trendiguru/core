__author__ = 'jeremy'

import os
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from trendi import constants


def convert_pd_output_dir(dir,converter=constants.fashionista_aug_zerobased_to_pixlevel_categories_v2,
                      input_suffix='.bmp',output_suffix='_pixlevelv2.bmp',for_webtool=True,
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
        converted_arr = convert_pd_output(f,converter=converter,input_suffix=input_suffix,output_suffix=output_suffix,for_webtool=for_webtool,
                          inlabels=inlabels,outlabels=outlabels)
        newname = os.path.join(dir,os.path.basename(f).replace(suffix_to_convert,suffix_to_convert_to))
        print('outname '+str(newname))
        cv2.imwrite(newname,converted_arr)


def convert_pd_output(filename_or_img_array,converter=constants.fashionista_aug_zerobased_to_pixlevel_categories_v2,
                      input_suffix='.bmp',output_suffix='_pixlevelv2.bmp',for_webtool=True,
                      inlabels=constants.fashionista_categories_augmented_zero_based,
                      outlabels=constants.pixlevel_categories_v2):
    '''
    convert e..g from paperdoll to ultimate21 or pixlevel_categories_v2 .
    Optionally only convert R channel for use with webtool. Don't forget to convert back to all chans after done w webtool
    :param converter:
    :param input_suffix:
    :param for_webtool:
    :return:
    '''
    if isinstance(filename_or_img_array,basestring):
        img_arr = cv2.imread(filename_or_img_array)
    else:
        img_arr = filename_or_img_array
    if img_arr is None:
        logging.warning('got null image in conversion_utils.convert_pd_output')
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


def count_values(mask,labels=None):
    image_size = mask.shape[0]*mask.shape[1]
    uniques = np.unique(mask)
    pixelcounts = {}
    mask =
    for unique in uniques:
        pixelcount = len(mask[mask==unique])
        ratio = float(pixelcount)/image_size
        if labels is not None:
            print('class {} {} count {} ratio {}'.format(unique,labels[unique],pixelcount,ratio))
        else:
            print('class {} count {} ratio {}'.format(unique,pixelcount,ratio))
        pixelcounts[unique]=pixelcount
    return pixelcounts

