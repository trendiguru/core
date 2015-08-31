__author__ = 'Nadav Paz'

import numpy as np
from scipy import io
import pymongo

import Utils
import constants
import find_similar_mongo


db = pymongo.MongoClient().mydb


def from_image_url_to_svgs(image_url, image_id):
    mask = find_mask_with_image_url(image_url)
    image_dict = {}
    items = []
    items_types = np.unique(mask)
    for item in items_types:
        if str(item) in constants.RELEVANT_ITEMS.keys():
            item_dict = {"category": constants.RELEVANT_ITEMS[str(item)]}
            item_mask = 255 * (np.zeros(np.shape(mask), np.uint8) + np.array(mask == item))
            # create svg for each item
            item_dict["svg_name"] = find_similar_mongo.mask2svg(
                item_mask,
                str(image_id) + '_' + constants.RELEVANT_ITEMS[str(item)],
                constants.svg_folder)
            item_dict["svg_url"] = constants.svg_url_prefix + item_dict["svg_name"]
            items.append(item_dict)
    image_dict["items"] = items
    return image_dict


def from_svg_to_similar_results(svg_url, image_url):
    if svg_url is None or image_url is None:
        return None
    image_dict = db.images.find_one({'image_urls': image_url})
    # find relevant item
    current_item = {}
    for item in image_dict["items"]:
        if item["svg_url"] == svg_url:
            current_item = item
            break
    # if the item already has similar results, return them
    if "similar_results" in current_item and len(current_item["similar_results"]) > 10:
        return image_dict

    # otherwise, go find similar items
    # get the paperdoll item number
    for item_num, category in constants.RELEVANT_ITEMS.iteritems():
        if category == current_item['category']:
            current_item_num = item_num
            break
    paperdoll_mask = find_mask_with_image_url(image_url)
    item_mask = 255 * (np.zeros(np.shape(paperdoll_mask), np.uint8) + np.array(paperdoll_mask == int(current_item_num)))
    image = Utils.get_cv2_img_array(image_url)
    current_item['fp'], current_item['similar_results'] = \
        find_similar_mongo.find_top_n_results(image, item_mask, 50, current_item['category'])

    # Cache similar items for later
    db.images.update_one({"_id": image_dict["_id"]}, {"$set": image_dict})

    return image_dict


def find_mask_with_image_url(image_url):
    folder = '/home/ubuntu/paperdoll/'
    f = open(folder + 'urls.txt')
    urls_list = f.read().splitlines()
    for i in range(0, len(urls_list)):
        if urls_list[i] == image_url:
            mask = io.loadmat(folder + 'masks' + '/' + str(i + 1) + '.mat')['mask']
    return mask


def find_or_create_image(image_url):
    """
    Search in db.images for the image by image url, if not exists - create one and start the process.
    :param image url - this is coming directly from the web interface so it's all we'll ever get.
    :return: image dictionary with svgs
    """
    if Utils.get_cv2_img_array(image_url) is None:
        return None
    image_dict = db.images.find_one({"image_urls": image_url})
    if image_dict is None or 'items' not in image_dict.keys():
        image_id = db.images.insert({"image_urls": [image_url]})
        image_temp_dict = from_image_url_to_svgs(image_url, image_id)
        image_dict = db.images.find_one_and_update({'_id': image_id}, {'$set': image_temp_dict},
                                                   return_document=pymongo.ReturnDocument.AFTER)
    return image_dict
