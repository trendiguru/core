
import cv2
import numpy as np
from rq import push_connection
import tldextract
from . import constants
from . import whitelist


db = constants.db
TTL = constants.general_ttl
q1 = constants.q1

push_connection(constants.redis_conn)

# -----------------------------------------------CO-FUNCTIONS-----------------------------------------------------------


def is_in_whitelist(page_url):
    page_domain = tldextract.extract(page_url).registered_domain
    if page_domain not in whitelist.all_white_lists:
        return False
    else:
        return True


def after_pd_conclusions(mask, labels, face=None):
    """
    1. if there's a full-body clothing:
        1.1 add to its' mask - all the rest lower body items' masks.
        1.2 add other upper cover items if they pass the pixel-amount condition/
    2. else -
        2.1 lower-body: decide whether it's a pants, jeans.. or a skirt, and share masks
        2.2 upper-body: decide whether it's a one-part or under & cover
    3. return new mask
    """
    if face:
        ref_area = face[2] * face[3]
        y_split = face[1] + 3 * face[3]
    else:
        ref_area = (np.mean((mask.shape[0], mask.shape[1])) / 10) ** 2
        y_split = np.round(0.4 * mask.shape[0])
    final_mask = mask[:, :]
    mask_sizes = {"upper_cover": [], "upper_under": [], "lower_cover": [], "lower_under": [], "whole_body": []}
    for num in np.unique(mask):
        item_mask = 255 * np.array(mask == num, dtype=np.uint8)
        category = list(labels.keys())[list(labels.values()).index(num)]
        print "W2P: checking {0}".format(category)
        for key, item in constants.paperdoll_categories.iteritems():
            if category in item:
                if float(cv2.countNonZero(item_mask))/mask.size > 0.01:
                    mask_sizes[key].append({num: cv2.countNonZero(item_mask)})
    # 1
    whole_sum = np.sum([item.values()[0] for item in mask_sizes['whole_body']])
    partly_sum = np.sum([item.values()[0] for item in mask_sizes['upper_under']]) +\
                 np.sum([item.values()[0] for item in mask_sizes['lower_cover']])
    if whole_sum > partly_sum:
        max_amount = np.max([item.values()[0] for item in mask_sizes['whole_body']])
        max_item_num = [item.keys()[0] for item in mask_sizes['whole_body'] if item.values()[0] == max_amount][0]
        max_item_cat = list(labels.keys())[list(labels.values()).index(max_item_num)]
        print "W2P: That's a {0}".format(max_item_cat)
        for num in np.unique(mask):
            cat = list(labels.keys())[list(labels.values()).index(num)]
            # 1.1, 1.2
            if cat in constants.paperdoll_categories["lower_cover"] or \
               cat in constants.paperdoll_categories["lower_under"] or \
               cat in constants.paperdoll_categories["upper_under"]:
                final_mask = np.where(mask == num, max_item_num, final_mask)
        return final_mask
    # 2, 2.1
    sections = {"upper_cover": 0, "upper_under": 0, "lower_cover": 0, "lower_under": 0}
    max_item_count = 0
    max_cat = 9
    print "W2P: That's a 2-part clothing item!"
    for section in sections.keys():
        for item in mask_sizes[section]:
            if item.values()[0] > max_item_count:
                max_item_count = item.values()[0]
                max_cat = item.keys()[0]
                sections[section] = max_cat
        # share masks
        if max_item_count > 0:
            for item in mask_sizes[section]:
                cat = list(labels.keys())[list(labels.values()).index(item.keys()[0])]
                # 2.1, 2.2
                if cat in constants.paperdoll_categories[section]:
                    final_mask = np.where(mask == item.keys()[0], max_cat, final_mask)
            max_item_count = 0

    for item in mask_sizes['whole_body']:
        for i in range(0, mask.shape[0]):
            if i <= y_split:
                for j in range(0, mask.shape[1]):
                    if mask[i][j] == item.keys()[0]:
                        final_mask[i][j] = sections["upper_under"] or sections["upper_cover"] or 0
            else:
                for j in range(0, mask.shape[1]):
                    if mask[i][j] == item.keys()[0]:
                        final_mask[i][j] = sections["lower_cover"] or sections["lower_under"] or 0
    return final_mask


def after_nn_conclusions(mask, labels, face=None):
    # Possible to add - remove coat from bottom of body
    # remove holes in items
    """
    0. threshold out small items
    1. if there's a full-body clothing:
        1.1 add to its' mask - all the rest lower body items' masks.
        1.2 add other upper cover items if they pass the pixel-amount condition/
    2. else -
        2.1 lower-body: decide whether it's a pants, jeans.. or a skirt, and share masks
        2.2 upper-body: decide whether it's a one-part or under & cover
    3. return new mask
    """
    if face:
        y_split = face[1] + 3 * face[3]
    else:
        # BETTER TO SEND A FACE
        y_split = np.round(0.4 * mask.shape[0])
    final_mask = mask[:, :]
    mask_sizes = {"upper_cover": [], "upper_under": [], "lower_cover": [], "lower_under": [], "whole_body": []}
    for num in np.unique(mask):
        item_mask = 255 * np.array(mask == num, dtype=np.uint8)
        category = list(labels.keys())[list(labels.values()).index(num)]
        print "W2P: checking {0}".format(category)
        for key, item in constants.nn_categories.iteritems():
            if category in item:
                if float(cv2.countNonZero(item_mask))/mask.size > 0.01:
                    mask_sizes[key].append({num: cv2.countNonZero(item_mask)})
        print mask_sizes
    # 1
    whole_sum = np.sum([item.values()[0] for item in mask_sizes['whole_body']])
    partly_sum = np.sum([item.values()[0] for item in mask_sizes['upper_under']]) +\
                 np.sum([item.values()[0] for item in mask_sizes['lower_cover']])
    if whole_sum > partly_sum:
        max_amount = np.max([item.values()[0] for item in mask_sizes['whole_body']])
        max_item_num = [item.keys()[0] for item in mask_sizes['whole_body'] if item.values()[0] == max_amount][0]
        max_item_cat = list(labels.keys())[list(labels.values()).index(max_item_num)]
        print "W2P: That's a {0}".format(max_item_cat)
        for num in np.unique(mask):
            cat = list(labels.keys())[list(labels.values()).index(num)]
            # 1.1, 1.2
            if cat in constants.nn_categories["lower_cover"] or \
               cat in constants.nn_categories["lower_under"] or \
               cat in constants.nn_categories["upper_under"] or \
               cat in constants.nn_categories["whole_body"]:
                final_mask = np.where(mask == num, max_item_num, final_mask)
        return final_mask
    # 2, 2.1
    sections = {"upper_cover": 0, "upper_under": 0, "lower_cover": 0, "lower_under": 0}
    max_item_count = 0
    max_cat = 9
    print "W2P: That's a 2-part clothing item!"
    for section in sections.keys():
        for item in mask_sizes[section]:
            if item.values()[0] > max_item_count:
                max_item_count = item.values()[0]
                max_cat = item.keys()[0]
                sections[section] = max_cat
        # share masks
        if max_item_count > 0:
            for item in mask_sizes[section]:
                cat = list(labels.keys())[list(labels.values()).index(item.keys()[0])]
                # 2.1, 2.2
                if cat in constants.nn_categories[section]:
                    final_mask = np.where(mask == item.keys()[0], max_cat, final_mask)
            max_item_count = 0

    for item in mask_sizes['whole_body']:
        for i in range(0, mask.shape[0]):
            if i <= y_split:
                for j in range(0, mask.shape[1]):
                    if mask[i][j] == item.keys()[0]:
                        final_mask[i][j] = sections["upper_under"] or sections["upper_cover"] or 0
            else:
                for j in range(0, mask.shape[1]):
                    if mask[i][j] == item.keys()[0]:
                        final_mask[i][j] = sections["lower_cover"] or sections["lower_under"] or 0
    return final_mask


def set_collections(lang):
    if not lang:
        return 'images', 'products'
    else:
        products_collection = 'products_' + lang
        images_collection = 'images_' + lang
        return images_collection, products_collection

