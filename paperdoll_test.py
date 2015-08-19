__author__ = 'Nadav Paz'

import random

import numpy as np
import cv2

import paperdoll.paperdoll_parse_enqueue
import Utils
import background_removal


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
            item_mask_gc = 2 * np.ones(np.shape(mask), np.uint8) - 1 * np.array(mask == num, dtype=np.uint8)
            after_gc = background_removal.simple_mask_grabcut(image, item_mask_gc)
            final_mask = np.bitwise_and(bgnd_mask, after_gc)
            cv2.imshow(category + "'s image (" + str(num) + ')', item_image)
            cv2.imshow(category + "'s gc image", background_removal.get_masked_image(image, final_mask))
            # cv2.imshow(category + "'s mask", 255 * item_mask / num)
            cv2.waitKey(0)
            cv2.destroyWindow(category + "'s image (" + str(num) + ')')
            cv2.destroyWindow(category + "'s gc image")
    cv2.destroyAllWindows()
