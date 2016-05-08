__author__ = 'jeremy'
import os
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from trendi import constants
from trendi import Utils

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

def fashionista_to_ultimate_21(file,index_conversion):
    ##########warning not finished #################3
    mask=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logging.warning('could not get file:'+file)
        return None
    if len(mask.shape)==3:
        logging.warning('multichannel mask, taking chan 0')
        mask=mask[:,:,0]
    uniques = np.unique(file)
    for u in uniques:
        newvals = [v[1] for v in index_conversion if v[0]==u] #find indice(s) of vals matching unique
        if len(newvals)!=1:
            logging.warning('found multiple or no target index for input index '+str(u))
            continue
        if newval<0:  #if you want to 'throw away' a cat just map it to background, otherwise
            logging.warning('found multiple or no target index for input index '+str(u))
            continue
        newindex = newvals[0]
        newval = index_conversion[newindex]
        print('replacing index {} with newindex {}'.format(u,newval))
        mask[mask==u] = newval

    _categories =['bk', 'T-shirt', 'bag', 'belt', 'blazer', 'shirt', 'coat', 'dress', 'face',
                  'hair', 'hat', 'jeans', 'legging', 'pants', 'scarf', 'shoe', 'shorts', 'skin',
                  'skirt', 'socks', 'stocking', 'sunglass', 'sweater']
    ## fashionista classes:
    fashionista_categories = ['null','tights','shorts','blazer','t-shirt','bag','shoes','coat','skirt','purse',
                            'boots','blouse','jacket','bra','dress','pants','sweater','shirt','jeans','leggings',
                            'scarf','hat','top','cardigan','accessories','vest','sunglasses','belt','socks','glasses',
                            'intimate','stockings','necklace','cape','jumper','sweatshirt','suit','bracelet','heels','wedges',
                            'ring','flats','tie','romper','sandals','earrings','gloves','sneakers','clogs','watch',
                            'pumps','wallet','bodysuit','loafers','hair','skin','face']
    conversion_dictionary_strings = {'background': ['null'],
                                    'bag': ['bag', 'purse'],
                                    'belt': ['belt'],
                                    'blazer': ['blazer', 'jacket', 'vest'],
                                    'top': ['t-shirt', 'shirt','blouse', 'top', 'sweatshirt'],
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
                                    'sweater': ['sweater', 'cardigan', 'jumper']}
#tossed'socks': ['socks'],
#tossed,'bodysuit'
#tossed​, 'accessories', 'ring', 'necklace', 'bracelet', 'wallet', 'tie', 'earrings', 'gloves', 'watch']
#tossed                                    'scarf': ['scarf']
    return mask


if __name__=="__main__":
    tamara_berg_improved_to_ultimate_21()