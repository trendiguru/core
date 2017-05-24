# -*- coding: utf-8 -*-
__author__ = 'jeremy'
import os
import cv2
import numpy as np
import logging
import subprocess
import sys
import json
from trendi.classifier_stuff.caffe_nns import conversion_utils
import pdb

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
    #also this is for images saved using the fashionista_categories_augmented labels, raw d output is with a dictionary of arbitrary labels
    #so use  get_pd_results_on_db_for_webtool.convert_and_save_results
    #actually this is a better place for that so now its copied here

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
        for fash_cat in v: #no need to check if the fashcat is in v since all the values here are in fashionista....
            fash_index = fashionista_categories_augmented.index(fash_cat)
#            fash_index = constants.fashionista_categories_augmented_zero_based.index(fash_cat)
            logging.debug('ultimate index {} cat {} fash index {} cat {}'.format(ultimate_21_index,ultimate_21[ultimate_21_index],fash_index,fashionista_categories_augmented[fash_index]))
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

def convert_pd_output(mask, label_names, new_labels=constants.fashionista_categories_augmented):
    '''
    This saves the mask using the labelling fashionista_categories_augmented_zero_based
    :param mask:
    :param label_names: dictionary of 'labelname':index
    :param new_labels: list of labels - map pd labels to the indices of these labels. guaranteed to work for constants.fashionista_aug_zerobased
    :return:
     '''
    h,w = mask.shape[0:2]
    new_mask=np.ones((h,w),dtype=np.uint8)*255  # anything left with 255 wasn't dealt with in the following conversion code
    print('new mask size:'+str(new_mask.shape))
    success = True #assume innocence until proven guilty
    print('attempting convert and save, shapes:'+str(mask.shape)+' new:'+str(new_mask.shape))
    for label in label_names:
        pd_index = label_names[label]
        if label in new_labels:
            if pd_index in mask:
                new_index = new_labels.index(label) + 0  # number by  0=null, 55=skin  , not 1=null,56=skin
                if new_index is None:
                    new_index = 0  #map unused categories (used in fashionista but not pixlevel v2)  to background
                print('pd index '+str(pd_index)+' for '+str(label)+': gets new index:'+str(new_index)+' '+str(new_labels[new_index]))
                new_mask[mask==pd_index] = new_index
        else:
            print('label '+str(label)+' not found in regular cats')
            success=False
    if 255 in new_mask:
        print('didnt fully convert mask')
        return
    conversion_utils.count_values(new_mask,new_labels)
  #  print('bincount:'+str(np.bincount(new_mask.flatten())))
    return new_mask


def hydra_results_to_fashionista(hydra_results,new_labels=constants.fashionista_categories_augmented):
    '''
    warning tested only on new_labels = fashionista_categories_augmented
    :param hydra_results:
    :param output_labels:
    :return:
    '''
    converted_results = np.zeros(len(new_labels))
    # ['','null','tights','shorts','blazer','t-shirt','bag','shoes','coat','skirt','purse',
    #                                 'boots','blouse','jacket','bra','dress','pants','sweater','shirt','jeans','leggings',
    #                                 'scarf','hat','top','cardigan','accessories','vest','sunglasses','belt','socks','glasses',
    #                                 'intimate','stockings','necklace','cape','jumper','sweatshirt','suit','bracelet','heels','wedges',
    #                                 'ring','flats','tie','romper','sandals','earrings','gloves','sneakers','clogs','watch',
    #                                 'pumps','wallet','bodysuit','loafers','hair','skin','face']
    for item in hydra_results:
        n_matched = 0
        logging.debug('item '+str(item))
        for label in new_labels:
            logging.debug('label '+str(label))
            if label == '':
                continue

            if (label in item or \
                (label=='shoes' and 'footwear' in item) or \
                (label=='tights' and 'leggings' in item) or \
                (label=='purse' and 'bag' in item) or \
                (label=='bra' and 'lingerie' in item) or \
                (label=='shirt' and 'top' in item) or \
                (label=='intimate' and 'lingerie' in item) or \
                (label=='purse' and 'bag' in item)) and not \
                ((label=='shirt' and 't-shirt' in item) or \
                (label=='suit' and 'tracksuit' in item) or \
                 (label=='shirt' and 'sweatshirt' in item)) :
              #  pdb.set_trace()
                n_matched += 1
#                i = [m.start() for m in re.finditer(label, item)]
                i = new_labels.index(label)
                converted_results[i] = hydra_results[item]
                print('using {} for {}, i {} newresult {} n_matched {} '.format(label,item,i,converted_results[i],n_matched))


        if n_matched == 0 :
            logging.warning('didnt get match for {}'.format(item))

        elif n_matched > 1 :
            logging.warning('got several matches for {}'.format(item))

    print('converted results:'+str(converted_results))
    for i in range(len(converted_results)):
        print('result {}:{} cat {}'.format(i,converted_results[i],new_labels[i]))
    return converted_results

def hydra_to_u21(hydra_results):
    # 'bgnd','bag','belt','blazer','coat','dress','eyewear','face','hair','hat',
    #            'jeans','leggings','pants','shoe','shorts','skin','skirt','stockings','suit','sweater',
    #            'top'
    new_labels = constants.ultimate_21
    converted_results = np.zeros(len(new_labels))
    for item in hydra_results:
        n_matched = 0
        logging.debug('item '+str(item))
        for label in new_labels:
            logging.debug('label '+str(label))
            if label == '':
                continue

            if (label in item or (label=='shoe' and 'footwear' in item)):
              #  pdb.set_trace()
                n_matched += 1
#                i = [m.start() for m in re.finditer(label, item)]
                i = new_labels.index(label)
                converted_results[i] = hydra_results[item]
                print('using {} for {}, i {} newresult {} n_matched {} '.format(label,item,i,converted_results[i],n_matched))

        if n_matched == 0 :
            logging.warning('didnt get match for {}'.format(item))

        elif n_matched > 1 :
            logging.warning('got several matches for {}'.format(item))

    print('converted results:'+str(converted_results))
    for i in range(len(converted_results)):
        print('result {}:{} cat {}'.format(i,converted_results[i],new_labels[i]))
    return converted_results

def hydra_to_pixlevel_v3(hydra_results):
    #pixlevelv3 are groups like whole_body, so the translation here will be into arrays
    #e.g. if we got dress=0.9 and suit=0.8 and shirt=0.5 from hydra, translate it to [[0.9,0.8],.....[0.5..]...]

# pixlevel_categories_v3 = ['bgnd','whole_body_items', 'whole_body_tight_items','undie_items','upper_under_items',
#                           'upper_cover_items','lower_cover_long_items','lower_cover_short_items','footwear_items','wraparound_items',
#                           'bag','belt','eyewear','hat','tie','skin']
# whole_body_group = ['dress','suit','overalls','tracksuit', 'sarong','robe','pajamas','jumpsuit']
# swimwear_group = ['womens_swimwear_nonbikini','bikini','mens_swimwear']
# undies_group = ['bra','panties','babydoll','lingerie','mens_underwear']
# upper_cover_group = ['coat', 'jacket']
# upper_middle_group = ['blazer','sweatshirt', 'hoodie', 'sweater', 'vest', 'poncho'] #cardigan is not here on purpose since it often goes w dress but sweater does not - check this....
# #put cardigan back and just use all the others specifically for dress, cuz this way cardigan has to be added to everything else
# upper_under_group = ['top','shirt','t-shirt', 'button-down', 'blouse', 'polo', 'henley', 'tube', 'tanktop']
# lower_long_group = ['jeans', 'pants','tights']
# lower_short_group = ['skirt','shorts']
# sock_group = ['leggings','stockings','socks']
# accessories_group = ['bag','belt','eyewear']
# footwear_group = ['footwear','boots','shoes','sandals']
# wrappy_things_group = ['shawl','scarf']
# eyewear_group = ['eyewear','glasses','sunglasses','shades']
   # pdb.set_trace()
    print('incoming dict:'+str(hydra_results))
    results_dict = hydra_results['data']
    new_labels = constants.pixlevel_categories_v3
    converted_results = [[] for i in new_labels]  #list of empty lists to populate
    for item in results_dict:
        n_matched = 0
        logging.debug('item '+str(item))
        try:
        #elif here so that each item gets match to only one group
        #do specific ones first so that: 'sweatshirt'  matches upper cover instead of upper_under(which includes 'shirt' that would match 'sweatshirt)
            if 'blazer' in item or 'sweatshirt' in item or 'sweater' in item :
                i = new_labels.index('upper_cover_items')
                converted_results[i].append(results_dict[item])
                n_matched += 1
                print('matched {} into upper_cover {}'.format(item,results_dict[item]))
            elif any([i in item for i in constants.whole_body_group]):
                i = new_labels.index('whole_body_items')
                converted_results[i].append(results_dict[item])
                n_matched += 1
                print('matched {} into whole_body {}'.format(item,results_dict[item]))
            elif any([i in item for i in constants.swimwear_group]):
                i = new_labels.index('whole_body_tight_items')
                converted_results[i].append(results_dict[item])
                n_matched += 1
                print('matched {} into whole_body_tight'.format(item))
            elif any([i in item for i in constants.undies_group]):
                i = new_labels.index('undie_items')
                converted_results[i].append(results_dict[item])
                n_matched += 1
                print('matched {} into undies {}'.format(item,results_dict[item]))
            elif any([i in item for i in constants.upper_under_group]):
                i = new_labels.index('upper_under_items')
                converted_results[i].append(results_dict[item])
                n_matched += 1
                print('matched {} into upper_under {}'.format(item,results_dict[item]))
            elif any([i in item for i in constants.upper_cover_group]):
                i = new_labels.index('upper_cover_items')
                converted_results[i].append(results_dict[item])
                n_matched += 1
                print('matched {} into upper_cover {}'.format(item,results_dict[item]))
            elif any([i in item for i in constants.lower_long_group]):
                i = new_labels.index('lower_cover_long_items')
                converted_results[i].append(results_dict[item])
                n_matched += 1
                print('matched {} into lower_cover_long {}'.format(item,results_dict[item]))
            elif any([i in item for i in constants.lower_short_group]):
                i = new_labels.index('lower_cover_short_items')
                converted_results[i].append(results_dict[item])
                n_matched += 1
                print('matched {} into lower_cover_short {}'.format(item,results_dict[item]))
            elif any([i in item for i in constants.footwear_group]):
                i = new_labels.index('footwear_items')
                converted_results[i].append(results_dict[item])
                n_matched += 1
                print('matched {} into footwear {}'.format(item,results_dict[item]))
            elif any([i in item for i in constants.wrappy_things_group]):
                i = new_labels.index('wraparound_items')
                converted_results[i].append(results_dict[item])
                n_matched += 1
                print('matched {} into wraparound {}'.format(item,results_dict[item]))
            elif any([i in item for i in constants.accessories_group]):
                i = new_labels.index('bag')
                converted_results[i].append(results_dict[item])
                n_matched += 1
                print('matched {} into bag {}'.format(item,results_dict[item]))
            elif 'cardigan' in item:
                i = new_labels.index('upper_cover_items')
                converted_results[i].append(results_dict[item])
                n_matched += 1
                print('matched {} into upper_cover {}'.format(item,results_dict[item]))

        except:
            e = sys.exc_info()[0]
            print( "<p>Error: %s</p>" % e )

        if n_matched == 0 :
            logging.warning('didnt get match for {}'.format(item))

        elif n_matched > 1 :
            logging.warning('got several matches for {}'.format(item))

    print('converted results:'+str(converted_results))
    for i in range(len(converted_results)):
        print('result {}:{} cat {}'.format(i,converted_results[i],new_labels[i]))
    return converted_results

if __name__=="__main__":

    file = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels/93586_var95.png'
    dir  = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels'
#    fashionista_to_ultimate_21_dir(dir)

    multilabel={u'data':
    {u't-shirt_45000': 0.022, u'shorts_binary_h_iter_50000': 0.006, u'stockings_30000': 0.004, u'top_hydra_iter_25000': 0.533, u'pants_hydra_iter_10000': 0.175, u'blazer_h_iter_15000': 0.0, u'dress_hydra_iter_50000': 0.598, u'belt_95000': 0.015, u'cardigan_binary_h_iter_25000': 0.511, u'jeans_binary_h_iter_10000': 0.102, u'tracksuit_80000': 0.0, u'coat_binary_h_iter_50000': 0.008, u'lingerie_binary_h_iter_50000': 0.0, u'blouse_45000': 0.284, u'hat_hydra_iter_60000': 0.011, u'sweatshirt_binary_h_iter_16000': 0.001, u'backpack_hydra_iter_2000': 0.001, u'shorts_binary_h_iter_30000_charles': 0.003, u'relevant_irrelevant_iter_10000': 0.638, u'skirt_binary_h_iter_50000': 0.987, u'leggings_80000': 0.232, u'url': u'http://s4.thisnext.com/media/largest_dimension/6DA7E812.jpg', u'sweater_binary_h_iter_50000': 0.116, u'jacket_binary_h_iter_50000': 0.007, u'footwear_50000': 0.954, u'suit_65000': 0.143, u'bag_55000': 0.011}}
    hydra_to_pixlevel_v3(multilabel)



#    imutils.show_mask_with_labels(file,constants.fashionista_categories_augmented_zero_based,visual_output=True)
#    newmask = fashionista_to_ultimate_21(file)
#    cv2.imwrite('testnewmask.bmp',newmask)
#   imutils.show_mask_with_labels('testnewmask.bmp',constants.ultimate_21,visual_output=True)


#    tamara_berg_improved_to_ultimate_21()