__author__ = 'jeremy'
import os
import cv2
import numpy as np

from trendi import pipeline
from trendi.utils import imutils
from trendi.constants import fashionista_categories_augmented_zero_based
from trendi.constants import fashionista_categories_augmented

def show_pd_results(dir):
    ########WARNING NOT FINISHED
    filenames = [os.path.join]
    for f in filenames:
        print('sending img '+f)
        im = cv2.imread(f)
#        mask, labels = retval.result[:2]
        parse_name = f.split('.jpg')[0]+'_parse.png'
        cv2.imwrite(parse_name,mask)
 #       print('labels:'+str(labels))
  #      sorted_labels=sorted(labels.items(),key=operator.itemgetter(1))
     #   print('sorted labels :'+str(sorted_labels))
     #   labs_only = [i[0] for i in sorted_labels]
      #  print('labsonly '+str(labs_only))
      #  imutils.show_mask_with_labels(parse_name,labs_only,save_images=True)

       # aftermask = pipeline.after_pd_conclusions(mask, labels, face=None)
        after_pd_conclusions_name = parse_name.split('_parse.png')[0]+'_after_pd_conclusions.png'
        #cv2.imwrite(after_pd_conclusions_name,aftermask)
        imutils.show_mask_with_labels(after_pd_conclusions_name,labs_only,save_images=True)

   #     if retval is not None:
   #         print('retval:' + str(retval.result)+' time:'+str(dt))
    #    else:
    #        print('no return val (None)')

def generate_groundtruth_legends(imgdir,labeldir):
    files = [os.path.join(imgdir,f) for f in os.listdir(imgdir) if '.jpg' in f]
    print('{} files in {}'.format(len(files),imgdir))
    for f in files:
        corresponding_label = os.path.basename(f).split('.jpg')[0]+'.png'
        full_label_path = os.path.join(labeldir,corresponding_label)
        imutils.show_mask_with_labels(full_label_path,fashionista_categories_augmented,save_images=True)

def after_nn_processdir(indir,outdir):
    masks = [f for f in os.listdir(dir) if '.bmp' in f]
    print('found {} files in {}'.format(len(masks),dir))
    label_dict = {fashionista_categories_augmented_zero_based[i]:i for i in range(len(fashionista_categories_augmented_zero_based))}
    print label_dict

    for f in masks:
        print f
        fullname = os.path.join(dir,f)
        mask = cv2.imread(fullname)  #have to worry abt 3chan masks?
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        after_mask = pipeline.after_nn_conclusions(mask, label_dict)
        after_nn_conclusions_name = os.path.join(outdir,f[:-4]+'_nnconclusions.bmp'
        cv2.imwrite(after_nn_conclusions_name,after_mask)
        imutils.show_mask_with_labels(after_nn_conclusions_name,fashionista_categories_augmented_zero_based,save_images=True)

if __name__ =="__main__":
#    generate_groundtruth_legends('/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/test','/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels')
    indir ='/home/jeremy/image_dbs/colorful_fashion_parsing_data/600x400_nn1_output_010516'
    outdir = indir+'_afterconclusions'
    after_nn_processdir(indir,outdir)

    indir ='/home/jeremy/image_dbs/colorful_fashion_parsing_data/150x100_nn2_output_010516'
    outdir = indir+'_afterconclusions'
    after_nn_processdir(indir,outdir)
