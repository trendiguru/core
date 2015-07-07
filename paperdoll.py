__author__ = 'Nadav Paz'

import numpy as np

import Utils
import constants
import find_similar_mongo


def from_image_url_to_svgs(image_url):
    # TODO
    # find the image & parsed mask from a folder
    # decide what will be the svg's filename (id / url)

    image_dict = {'image_url': image_url}
    items = []
    items_types = np.unique(mask)
    for item in items_types:
        if str(item) in constants.RELEVANT_ITEMS.keys():
            item_dict = {"category": constants.RELEVANT_ITEMS[str(item)],
                         "mask": np.zeros(np.shape(mask), np.uint8) + np.array(mask == item)}
            # create svg for each item
            item_dict["svg"] = find_similar_mongo.mask2svg(
                item_dict["mask"],
                filename,
                constants.svg_folder)
            items.append(item_dict)
    image_dict["items"] = items
    return image_dict


def from_svg_to_similar_results(svg, image_dict):
    for item in image_dict["items"]:
        if item["svg"] is svg:
            current_item = item
    image = Utils.get_cv2_img_array(image_dict['image_url'])
    current_item['fp'], current_item['similar_results'] = find_similar_mongo.find_top_n_results(image,
                                                                                                current_item["mask"],
                                                                                                10,
                                                                                                current_item[
                                                                                                    'category'])
    return image_dict

