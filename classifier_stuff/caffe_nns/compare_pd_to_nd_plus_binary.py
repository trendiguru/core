__author__ = 'jeremy'

import cv2
import pymongo
import subprocess
import os
from time import sleep
import numpy as np

from trendi.paperdoll import pd_falcon_client
from trendi import constants#
from trendi import Utils
from trendi.utils import imutils
from trendi.paperdoll import hydra_tg_falcon_client
from trendi.paperdoll import neurodoll_falcon_client
from trendi.utils import imutils
from trendi import pipeline
from trendi.downloaders import label_conversions
#from trendi import neurodoll

def get_live_pd_results(image_file,save_dir='/data/jeremy/image_dbs/tg/pixlevel/pixlevel_fullsize_test_pd_results'):
    #use the api - so first get the image onto the web , then aim the api at it
    copycmd = 'scp '+image_file+' root@104.155.22.95:/var/www/results/pd_test/'+os.path.basename(image_file)
    subprocess.call(copycmd,shell=True)
    sleep(1) #give time for file to get to extremeli - maybe unecessary (if subprocess is synchronous)
    url = 'http://extremeli.trendi.guru/demo/results/pd_test/'+os.path.basename(image_file)
    resp = pd_falcon_client.pd(url)
    print('resp:'+str(resp))
    label_dict = resp['label_dict']
    mask = resp['mask']
    #label_dict = {fashionista_categories_augmented_zero_based[i]:i for i in range(len(fashionista_categories_augmented_zero_based))}
    print label_dict
    if len(mask.shape) == 3:
        mask = mask[:,:,0]

    u21_mask = label_conversions.fashionista_to_ultimate_21(mask)
    print('u21 mask:'+str())

#could also have used
    #   get_pd_results_on_db_for_webtool.convert_and_save_results


    #make a legend of original mask
    print('save dir:'+save_dir)
    image_base = os.path.basename(image_file)
    before_pd_conclusions_name = os.path.join(save_dir,image_base[:-4]+'_pd.bmp')
    res=cv2.imwrite(before_pd_conclusions_name,u21_mask)
    print('save result '+str(res)+ ' for file '+before_pd_conclusions_name)
    imutils.show_mask_with_labels(before_pd_conclusions_name,constants.fashionista_categories_augmented,save_images=True)

    #make a legend of mask after pd conclusions
    after_mask = pipeline.after_pd_conclusions(u21_mask, label_dict)
    after_pd_conclusions_name = os.path.join(save_dir,image_base[:-4]+'_after_pd_conclusions.bmp')
    res = cv2.imwrite(after_pd_conclusions_name,after_mask)
    print('save result '+str(res)+' for file '+after_pd_conclusions_name)

    imutils.show_mask_with_labels(after_pd_conclusions_name,constants.fashionista_categories_augmented,save_images=True)

    #send legends to extremeli
    copycmd = 'scp '+before_pd_conclusions_name+' root@104.155.22.95:/var/www/results/pd_test/'+os.path.basename(before_pd_conclusions_name)
    subprocess.call(copycmd,shell=True)
    copycmd = 'scp '+after_pd_conclusions_name+' root@104.155.22.95:/var/www/results/pd_test/'+os.path.basename(after_pd_conclusions_name)
    subprocess.call(copycmd,shell=True)

    #pose also available , resp['pose']
    #make list of labels in ultimate_21 format

def get_saved_pd_results(mask_file):
    img_arr = cv2.imread(mask_file)
    if not img_arr:
        print('couldnt open '+mask_file)
        return None
    vals = np.unique(img_arr)
    #make sure vals with almost no pixels are tossed out
    multilabel_result = [(i in vals) for i in range(len(constants.ultimate_21))]
    print('result from {} has uniques {}, list looks like {}'.format(mask_file,vals,multilabel_result))

def get_hydra_nd_results(url):
    hydra_result = hydra_tg_falcon_client.hydra_tg(url)
#map hydra result to something equivalent to paperdoll output , return

    nd_result = neurodoll_falcon_client.nd(url,category_index=None,get_combined_results=True,multilabel_results=hydra_result)
    #todo - allow nd to accept multilabel results as input
    #convert multilabel results to ultimate_21 classes
    # nd_result = neurodoll_falcon_client.nd(url,category_index=None,get_multilabel_results=None,
    #                                        get_combined_results=None,get_layer_output=None,
    #                                        get_all_graylevels=None,threshold=None)

    # nd_result = neurodoll.combine_neurodoll_and_multilabel(url_or_np_array,multilabel_threshold=0.7,median_factor=1.0,
    #                                  multilabel_to_ultimate21_conversion=constants.binary_classifier_categories_to_ultimate_21,
    #                                  multilabel_labels=constants.binary_classifier_categories, face=None,
    #                                  output_layer = 'pixlevel_sigmoid_output',required_image_size=(224,224),
    #                                  do_graylevel_zeroing=True)

def image_to_name(url_or_filename_or_img_arr):
    if isinstance(url_or_filename_or_img_arr,basestring):
        name = basestring.replace('https://','').replace('http://','').replace('/','_')
    elif isinstance(url_or_filename_or_img_arr,np.ndarray):
        name = hash(str(url_or_filename_or_img_arr))
    print('name:'+name)
    return name

def get_groundtruth_for_tamaraberg_multilabel(labelfile='/data/jeremy/image_dbs/labels/labelfiles_tb/tb_cats_from_webtool_round2_train.txt',
                                              label_cats=constants.web_tool_categories_v2):
    with open(labelfile,'r') as fp:
        lines = fp.readlines()
    imgs_and_labels = [(line.split()[0],[int(i) for i in line.split()[1:]]) for line in lines]
    print(str(len(imgs_and_labels))+' images described in file '+labelfile)
    print('data looks like '+str(imgs_and_labels[0]))
    return imgs_and_labels

def do_imagelevel_comparison():
    imgs_and_labels = get_groundtruth_for_tamaraberg_multilabel()
    for image_file,label in imgs_and_labels:
        pd_result = get_pd_results(image_file)
        hydra_nd_result = get_hydra_nd_results(image_file)


def dl_images(source_domain='stylebook.de',text_filter='',dl_dir='/data/jeremy/image_dbs/golden/',in_docker=True,visual_output=False):
    '''
    dl everything in the images db, on the assumption that these are the  most relevant to test our answers to
    :return:
    '''

    if in_docker:
        db = pymongo.MongoClient('localhost',port=27017).mydb
    else:
        db = constants.db

    all = db.images.find({'domain':source_domain})
    for doc in all:
        url=doc['image_urls'][0]
        if text_filter in url[0]:
            print url
            Utils.get_cv2_img_array(url,convert_url_to_local_filename=True,download=True,download_directory=dl_dir)
        else:
            print('skipping '+url)

    #move the images with more than one person
    imutils.do_for_all_files_in_dir(imutils.one_person_per_image,dl_dir,visual_output=False)

if __name__ == "__main__":
    Utils.map_function_on_dir(get_live_pd_results,'/data/jeremy/image_dbs/tg/pixlevel/pixlevel_fullsize_test/')