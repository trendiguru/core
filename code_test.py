__author__ = 'Nadav Paz'

import cv2
import numpy as np
from bson import json_util

import background_removal
import find_similar_mongo


def gc2mask_test(image, bb):
    small_image, resize_ratio = background_removal.standard_resize(image, 400)    # shrink image for faster process
    bb = np.array(bb)/resize_ratio
    bb = bb.astype(np.uint16)                                                     # shrink bb in the same ratio
    # bb = [int(b) for b in (np.array(bb)/resize_ratio)]
    x, y, w, h = bb
    cv2.rectangle(small_image, (x, y), (x+w, y+h), [0, 255, 0], 2)
    cv2.imshow('1', small_image)
    cv2.waitKey(0)
    fg_mask = background_removal.get_fg_mask(small_image, bb)                     # returns the grab-cut mask (if bb => PFG-PBG gc, if !bb => face gc)
    cv2.imshow('2', background_removal.get_masked_image(small_image, fg_mask))
    cv2.waitKey(0)
    bb_mask = background_removal.get_binary_bb_mask(small_image, bb)                     # bounding box mask
    cv2.imshow('3', background_removal.get_masked_image(small_image, bb_mask))
    cv2.waitKey(0)
    combined_mask = cv2.bitwise_and(fg_mask, bb_mask)                             # for sending the right mask to the fp
    cv2.imshow('4', background_removal.get_masked_image(small_image, combined_mask))
    cv2.waitKey(0)
    return


inputs = json_util.loads(
    '{"url":"http://msc.wcdn.co.il/w/w-635/1684386-5.jpg","bb":"[137.2972972972973,188.80597014925374,356.97297297297297,319.2537313432836]","keyword":"mens-outerwear","post_id":"552a79359e31f134f0f9c401"}')
find_similar_mongo.got_bb(inputs["url"], inputs["post_id"], json_util.loads(inputs["bb"]), 10, inputs["keyword"])
