__author__ = 'Nadav Paz'

import random
import numpy as np
import cv2
import os
import logging
logging.basicConfig(level=logging.INFO)


from trendi.paperdoll import paperdoll_parse_enqueue
import Utils
import background_removal
from trendi import constants
from trendi.classifier_stuff.caffe_nns import jrinfer
from trendi import pipeline
from trendi.utils import imutils
from trendi.paperdoll import pd

def color_paperdoll_mask(paperdoll_mask):
    items_list = np.unique(paperdoll_mask)
    l = list(np.shape(paperdoll_mask))
    l.append(3)
    color_mask = np.zeros(tuple(l), np.uint8)
    for item in items_list:
        for k in range(0, 3):
            color_mask[:, :, k] = color_mask[:, :, k] + random.randint(0, 255) * np.array(
                paperdoll_mask == item)
    return color_mask


# TEST/SHOWCASE:
# input: image_url from the web
# displays the source image and paperdoll mask
# then - loops over items in the mask and display each items on image by paperdoll and by grabcut
# the grabcut's input is the item's mask and it expends and tries to grt to full accurate shape
# in addition, the label is written as the window's name


def pd_test(image_url):
    '''not sure if this is currenltly kosher 10.12.16'''
    image = Utils.get_cv2_img_array(image_url)
    mask, labels, pose = paperdoll.paperdoll_parse_enqueue.paperdoll_enqueue(image_url, async=False)
    cv2.imshow('image', image)
    cv2.imshow('color_mask', color_paperdoll_mask(mask))
    bgnd_mask = []
    for num in np.unique(mask):
        # convert numbers to labelsC
        category = list(labels.keys())[list(labels.values()).index(num)]
        item_mask = 255 * np.array(mask == num, dtype=np.uint8)
        if category == 'null':
            bgnd_mask = 255 - item_mask
        if cv2.countNonZero(item_mask) > 2000:
            item_image = background_removal.get_masked_image(image, item_mask)
            after_gc = create_gc_mask(image, item_mask, bgnd_mask)
            cv2.imshow(category + "'s image (" + str(num) + ')', item_image)
            cv2.imshow(category + "'s gc image",
                       background_removal.get_masked_image(image, background_removal.get_masked_image(image, after_gc)))
            # cv2.imshow(category + "'s mask", 255 * item_mask / num)
            cv2.waitKey(0)
            cv2.destroyWindow(category + "'s image (" + str(num) + ')')
            cv2.destroyWindow(category + "'s gc image")
    cv2.destroyAllWindows()

def pd_test_iou_and_cats(images_file='/home/jeremy/image_dbs/pixlevel/pixlevel_fullsize_test_labels_faz.txt',
                         n_channels=len(constants.fashionista_categories_augmented),labels=constants.fashionista_categories_augmented):
    if not(os._exists(images_file)):
        logging.warning('file {} does not exist, exiting'.format(images_file))
    with open(images_file,'r') as fp:
        lines = fp.readlines()
        imgfiles = [line.split[0] for line in lines]
        labelfiles = [line.split[0] for line in lines]
    hist = np.zeros((n_channels, n_channels))

    for image_file,labelfile in zip(imgfiles,labelfiles):
        image_arr = Utils.get_cv2_img_array(image_file)
        gt_arr = cv2.imread(labelfile)
        print('gt size {} img size {}'.format(gt_arr.shape,image_arr.shape))
        mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(image_arr, async=False)
        converted_mask = pd.convert_and_save_results(mask,labels)
        print('mask uniques {} gt uniques {}'.format(np.unique(final_mask),np.unique(gt_arr)))
        final_mask = pipeline.after_pd_conclusions(mask,labels)
        converted_final_mask = pd.convert_and_save_results(final_mask,labels)
        print('final mask uniques {} gt uniques {}'.format(np.unique(converted_final_mask),np.unique(gt_arr)))
    #before conclusions
        savename = os.path.basename(image_file).replace('.jpg','_legend.jpg')
        imutils.show_mask_with_labels(converted_mask,labels=constants.fashionista_categories_augmented_zero_based,original_image=image_file,visual_output=True,savename=savename)
    #after conclusions
        savename_finalmask = os.path.basename(image_file).replace('.jpg','_afterpdconclusions_legend.jpg')
        imutils.show_mask_with_labels(converted_final_mask,labels=constants.fashionista_categories_augmented_zero_based,original_image=image_file,visual_output=True,savename=savename_finalmask)
    #ground truth
        gtsavename = os.path.basename(image_file).replace('.jpg','_gt_legend.jpg')
        imutils.show_mask_with_labels(gt_arr,labels=constants.fashionista_categories_augmented_zero_based,original_image=image_file,visual_output=True,savename=gtsavename)
    #maks (after conclusions)
        bmpname = os.path.basename(image_file).replace('.jpg','pd.bmp')
        cv2.imwrite(bmpname,converted_final_mask)
        print('saving naive legend to '+savename+' afterconclusions legend to '+savename_finalmask+' gt legend to '+gtsavename+', mask to '+bmpname)

        hist += jrinfer.fast_hist(gt_arr,final_mask,n_channels)

    jrinfer.results_from_hist(hist,labels=labels)


def create_gc_mask(image, pd_mask, bgnd_mask):
    item_gc_mask = np.where(pd_mask == 255, 1, 2).astype('uint8')  # (2, 3) mask
    after_gc_mask = background_removal.simple_mask_grabcut(image, item_gc_mask)  # (255, 0) mask
    final_mask = cv2.bitwise_and(bgnd_mask, after_gc_mask)
    return final_mask  # (255, 0) mask