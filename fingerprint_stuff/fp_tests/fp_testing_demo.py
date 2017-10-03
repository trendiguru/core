__author__ = 'yonatan'
'''
this is used in the fp testing demo
'''

import logging

import numpy as np
import pymongo
import cv2

import background_removal
from paperdoll import paperdoll_parse_enqueue
import utils_tg
import constants
import find_similar_mongo
from constants import db

fingerprint_length = constants.fingerprint_length
histograms_length = constants.histograms_length
folder = '/home/ubuntu/paperdoll/masks/'



def find_or_create_image(image_url):
    """
    Search in db.images for the image by image url, if not exists - create one and start the process.
    :param image url - this is coming directly from the web interface so it's all we'll ever get.
    :return: image dictionary with svgs
    """
    image = background_removal.standard_resize(utils_tg.get_cv2_img_array(image_url), 400)[0]
    if image is None:
        logging.warning("Bad url!")
        return None
    image_dict = db.images.find_one({"image_urls": image_url})
    if image_dict is None or 'items' not in image_dict.keys():
        if image_dict is None:
            image_id = db.images.insert({"image_urls": [image_url]})
            # TODO - where is the case which we append other url on the same image
        else:
            image_id = image_dict['_id']
        items_dict = from_image_to_svgs(image, image_id)
        image_dict = db.images.find_one_and_update({'image_urls': image_url}, {'$set': items_dict},
                                                   return_document=pymongo.ReturnDocument.AFTER)
    return image_dict


def from_image_to_svgs(image, image_id):
    mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(image, async=False)
    items = []
    bgnd_mask = []
    for num in np.unique(mask):
        # convert numbers to labels
        category = list(labels.keys())[list(labels.values()).index(num)]
        item_mask = 255 * np.array(mask == num, dtype=np.uint8)
        if category == 'null':
            bgnd_mask = 255 - item_mask  # (255, 0) masks list
        if cv2.countNonZero(item_mask) > 2000 and category in constants.paperdoll_shopstyle_women.keys():
            item_gc_mask = create_gc_mask(image, item_mask, bgnd_mask)  # (255, 0) mask
            item_dict = {"category": constants.paperdoll_shopstyle_women[category]}
            mask_name = folder + str(image_id) + '_' + item_dict['category'] + '.png'
            item_dict['mask_name'] = mask_name
            cv2.imwrite(mask_name, item_gc_mask)
            # create svg for each item
            item_dict["svg_name"] = find_similar_mongo.mask2svg(
                item_gc_mask,
                str(image_id) + '_' + item_dict['category'],
                constants.svg_folder)
            item_dict["svg_url"] = constants.svg_url_prefix + item_dict["svg_name"]
            items.append(item_dict)
    image_dict = {"items": items}
    return image_dict


def from_svg_to_similar_results(svg_url, image_url, fp_length=fingerprint_length, bins=histograms_length,
                                collection_name="products",
                                fp_category="fingerprint",
                                distance_func=None):
    projection_dict = {
        'seeMoreUrl': 1,
        'image': 1,
        'clickUrl': 1,
        'retailer': 1,
        'currency': 1,
        'brand': 1,
        'description': 1,
        'price': 1,
        'categories': 1,
        'name': 1,
        'sizes': 1,
        'pageUrl': 1,
        '_id': 0,
        'priceLabel': 1
    }
    if svg_url is None or image_url is None:
        logging.warning("Bad urls!")
        return None
    image_dict = db.images.find_one({'image_urls': image_url})
    if image_dict is None:
        logging.warning("item wasn't found for some reason")
        return None
    for item in image_dict["items"]:
        if item["svg_url"] == svg_url:
            curr_item = item
            item_mask = cv2.imread(curr_item['mask_name'])[:, :, 0]
            image = background_removal.standard_resize(utils_tg.get_cv2_img_array(image_dict['image_urls'][0]), 400)[0]

            curr_item['fp'], curr_item['similar_results'] = \
                find_similar_mongo.find_top_n_results(image, item_mask, 30, curr_item['category'], collection_name,
                                                      fp_category, fp_length, distance_func, bins)

            # top_matches = {"similar_results": [db.products.find_one({"_id": result["_id"]})
            #                          for result in curr_item['similar_results']]}
            #
            # return top_matches

            return db.images.find_one_and_update({'items.svg_url': curr_item["svg_url"]},
                                                 {'$set': {'items.$': curr_item}},
                                                 return_document=pymongo.ReturnDocument.AFTER)


def create_gc_mask(image, pd_mask, bgnd_mask):
    item_bb = bb_from_mask(pd_mask)
    item_gc_mask = background_removal.paperdoll_item_mask(pd_mask, item_bb)
    after_gc_mask = background_removal.simple_mask_grabcut(image, item_gc_mask)  # (255, 0) mask
    final_mask = cv2.bitwise_and(bgnd_mask, after_gc_mask)
    return final_mask  # (255, 0) mask


def bb_from_mask(mask):
    r, c = mask.nonzero()
    x = min(c)
    y = min(r)
    w = max(c) - x + 1
    h = max(r) - y + 1
    return x, y, w, h
