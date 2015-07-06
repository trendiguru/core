__author__ = 'Nadav Paz'

import numpy as np

import constants
import find_similar_mongo


def from_paperdoll_mask_to_results(image, mask, filename):
    # TODO:
    # check mask with image resize

    items_types = np.unique(mask)
    items = []
    for item in items_types:
        if str(item) in constants.RELEVANT_ITEMS.keys():
            item_dict = {}
            item_dict["category"] = constants.RELEVANT_ITEMS[str(item)]
            item_mask = np.zeros(np.shape(mask), np.uint8) + np.array(mask == item)
            # create svg for each item
            item_dict["svg"] = find_similar_mongo.mask2svg(
                item_mask,
                filename,
                constants.svg_folder)
            # fingerprint & find similar results
            item_dict["fp"], item_dict["similar_items"] = find_similar_mongo.find_top_n_results(
                image,
                item_mask,
                10,
                constants.RELEVANT_ITEMS[str(item)])
            items.append(item_dict)
    return items
