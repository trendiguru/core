# -*- coding: utf-8 -*-
__author__ = 'jeremy'
import os
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from trendi import constants
from trendi import Utils
from trendi.utils import imutils

def color_fashion_to_fashionista():
    i=0
    for cat in constants.colorful_fashion_parsing_categories:
        newcat=constants.colorful_fashion_to_fashionista[cat]
        cfrp_index = constants.colorful_fashion_parsing_categories.index(cat)
     #   print('oldcat {} newcat {} '.format(cat,newcat))
        if newcat in constants.fashionista_categories:
            fashionista_index = constants.fashionista_categories.index(newcat)
        #    print('oldcat {} newcat {} cfrpix {} fashionistadx {}'.format(cat,newcat,cfrp_index,fashionista_index))
            print('({},{})'.format(cfrp_index,fashionista_index))
        else:
            print('unhandled category:'+str(cat)+' fashionista index:'+str(cfrp_index))
        i=i+1

def tamara_berg_improved_to_ultimate_21(dir_of_masks,overwrite=False,outdir=None,filter=None):
    for ind in range(0,len(constants.tamara_berg_improved_categories)):
        tup = constants.tamara_berg_improved_to_ultimate_21_index_conversion[ind]
        ultimate21_index=tup[1]
        if ultimate21_index <0:
            print('unhandled category:'+str(ind)+' berg label:'+str(constants.tamara_berg_improved_categories[ind]))
        else:
            print('oldcat {} {} newcat {} {} '.format(ind,constants.tamara_berg_improved_categories[ind],ultimate21_index,constants.ultimate_21[ultimate21_index]))
    replace_labels_dir(dir_of_masks,constants.tamara_berg_improved_to_ultimate_21_index_conversion,overwrite=overwrite,outdir=outdir,filter=filter)

def replace_labels_dir(dir_of_masks,index_conversion,overwrite=False,outdir=None,filter=None):
    '''
    convert masks in dir from tamara_berg_improved to ultimate_21 labels
    #21 cats for direct replacement of VOC systems
    #lose the necklace,
    #combine tights and leggings
    :param dir_of_masks:
    :return:
    '''
    if filter:
        masks = [os.path.join(dir_of_masks,f) for f in os.listdir(dir_of_masks) if filter in f]
    else:
        masks = [os.path.join(dir_of_masks,f) for f in os.listdir(dir_of_masks)]
    for f in masks:
        mask_arr=replace_labels(f,index_conversion)
        if mask_arr is None:
            logging.warning('got no file')
            continue
        if len(mask_arr.shape)==3:
            logging.warning('multichannel mask, taking chan 0')
            mask_arr=mask_arr[:,:,0]
        if overwrite:
            outname = f
        else:
            indir = os.path.dirname(f)
            parentdir = os.path.abspath(os.path.join(indir, os.pardir))
            if outdir==None:
                outdir = os.path.join(parentdir,'_relabelled')
                Utils.ensure_dir(outdir)
            inname = os.path.basename(f)
            outname = os.path.join(outdir,f)
        print('in {} out {}'.format(f,outname))
        cv2.imwrite(mask_arr)

def fashionista_to_ultimate_21_dir(dir):
    masks = [os.path.join(dir,m) for m in os.listdir(dir) if 'png' in m or 'bmp' in m]
    print(str(len(masks))+' masks in '+dir)
    for maskfile in masks:
        newmask = fashionista_to_ultimate_21(maskfile)
        newname = maskfile[:-4]+'_u21'+maskfile[-4:]
        cv2.imwrite(newname,newmask)
        print('new maskname:'+newname)
#        imutils.show_mask_with_labels(newname,constants.ultimate_21,visual_output=True)

def fashionista_to_ultimate_21(img_arr_or_url_or_file):
    ##########warning not finished #################3

    ultimate_21 = ['bgnd','bag','belt','blazer','coat','dress','eyewear','face','hair','hat',
                   'jeans','legging','pants','shoe','shorts','skin','skirt','stocking','suit','sweater',
                   'top']
#tossed,'bodysuit', 'socks','bra'
#tossed​, 'accessories', 'ring', 'necklace', 'bracelet', 'wallet', 'tie', 'earrings', 'gloves', 'watch']
#scarf aded to shirt since it mostly shows up there

# ## fashionista classes:
    fashionista_categories_augmented = ['','null','tights','shorts','blazer','t-shirt','bag','shoes','coat','skirt','purse',
                                    'boots','blouse','jacket','bra','dress','pants','sweater','shirt','jeans','leggings',
                                    'scarf','hat','top','cardigan','accessories','vest','sunglasses','belt','socks','glasses',
                                    'intimate','stockings','necklace','cape','jumper','sweatshirt','suit','bracelet','heels','wedges',
                                    'ring','flats','tie','romper','sandals','earrings','gloves','sneakers','clogs','watch',
                                    'pumps','wallet','bodysuit','loafers','hair','skin','face']  #0='',1='null'(bgnd) 57='face'

    #CONVERSION FROM FASH 57 TO ULTIMATE21
    conversion_dictionary_strings = {'bgnd': ['null'],
                                    'bag': ['bag', 'purse'],
                                    'belt': ['belt'],
                                    'blazer': ['blazer', 'jacket', 'vest'],
                                    'top': ['t-shirt', 'shirt','blouse', 'top', 'sweatshirt', 'scarf'],
                                    'coat': ['coat', 'cape'],
                                    'dress': ['dress',  'romper'],
                                    'suit': ['suit'],
                                    'face': ['face'],
                                    'hair': ['hair'],
                                    'hat': ['hat'],
                                    'jeans': ['jeans'],
                                    'legging': ['tights', 'leggings'],
                                    'pants': ['pants'],
                                    'shoe': ['shoes', 'boots', 'heels', 'wedges', 'pumps', 'loafers', 'flats', 'sandals', 'sneakers', 'clogs'],
                                    'shorts': ['shorts'],
                                    'skin': ['skin'],
                                    'skirt': ['skirt'],
                                    'stocking': ['intimate', 'stockings'],
                                    'eyewear': ['sunglasses', 'glasses'],
                                    'sweater': ['sweater', 'cardigan', 'jumper']
                                     }


    index_conversion = [-666 for i in range(len(fashionista_categories_augmented))]
    for k,v in conversion_dictionary_strings.iteritems():
        ultimate_21_index = ultimate_21.index(k)
        for fash_cat in v:
            fash_index = fashionista_categories_augmented.index(fash_cat)
#            fash_index = constants.fashionista_categories_augmented_zero_based.index(fash_cat)
 #           print('ultimate index {} cat {} fasjh index {} cat {}'.format(ultimate_21_index,ultimate_21[ultimate_21_index],fash_index,fashionista_categories_augmented[fash_index]))
            index_conversion[fash_index] = ultimate_21_index

    print(index_conversion)
#    for i in range(len(index_conversion)):
#        if index_conversion[i] == -666:
#            print('unmapped fashcat:'+str(i)+fashionista_categories_augmented[i])

    if isinstance(img_arr_or_url_or_file,basestring):
        mask = Utils.get_cv2_img_array(img_arr_or_url_or_file)
    #todo - check why get_cv2_img_array doesnt like getting a  mask
    else:
        mask = img_arr_or_url_or_file
     #   mask=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    if mask is None:
        if isinstance(img_arr_or_url_or_file,basestring):
            logging.warning('could not get filename/url:'+str(img_arr_or_url_or_file))
        else:
            logging.warning('could not get file/url')
        return None
    if len(mask.shape)==3:
        logging.warning('multichannel mask, taking chan 0')
        mask=mask[:,:,0]
    uniques = np.unique(mask)
#    print('uniques:'+str(uniques))
    for u in uniques:
        newval = index_conversion[u] #find indice(s) of vals matching unique
        if newval<0:  #if you want to 'throw away' a cat just map it to background, otherwise
            newval = 0
        print('replacing index {} with newindex {}'.format(u,newval))
        mask[mask==u] = newval
    return mask


if __name__=="__main__":

    file = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels/93586_var95.png'
    dir  = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels'
    fashionista_to_ultimate_21_dir(dir)

#    imutils.show_mask_with_labels(file,constants.fashionista_categories_augmented_zero_based,visual_output=True)
#    newmask = fashionista_to_ultimate_21(file)
#    cv2.imwrite('testnewmask.bmp',newmask)
#   imutils.show_mask_with_labels('testnewmask.bmp',constants.ultimate_21,visual_output=True)


#    tamara_berg_improved_to_ultimate_21()