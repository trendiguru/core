__author__ = 'Nadav Paz'

import logging

import pymongo
import numpy as np

import background_removal
import Utils


def from_image_url_to_task1(image_url):
    db = pymongo.MongoClient().mydb
    images = db.images
    image = background_removal.standard_resize(Utils.get_cv2_img_array(image_url), 400)[0]
    if image is None:
        logging.warning("There's no image in the url!")
        return None
        # TODO:
        # 1. is there a chance that something will call the function when this image is already in our DB ?
        # image_dict = db.images.find_one({"image_url": image_url})
        # if image_dict['relevant'] and (image_dict is None or image_dict["faces"] is None):
        # do something
    relevance = background_removal.image_is_relevant(image)
    image_id = images.insert({'image_url': image_url,
                              'relevant': relevance.is_relevant,
                              'faces': np.array(relevance.faces)})
    return db.images.find_one({'_id': image_id})