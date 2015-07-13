__author__ = 'Nadav Paz'

import logging

import pymongo
import cv2

import background_removal
import Utils

db = pymongo.MongoClient().mydb


def from_image_url_to_task1(image_url):
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
    image_obj_id = images.insert({'image_url': image_url,
                                  'relevant': relevance.is_relevant,
                                  'people': []})
    image_dict = images.find_one({'_id': image_obj_id})
    for idx, face in enumerate(relevance.faces):
        x, y, w, h = face
        image_dict['people'].append({'face': face.tolist()})
        copy = image.copy()
        cv2.rectangle(copy, (x, y), (x + w, y + h), [0, 255, 0], 2)
        cv2.imwrite('/home/ubuntu/Dev/qcs' + '/' + str(idx) + '.jpg', copy)
    return image_dict


def validate_cats_and_send_to_bb(cats, obj_id):
    for cat in cats:
        if cat is None:
            logging.warning("category is None, check the strings coming back from the QCs")
            # cat = recall to the task1 process
        if cat not in db.categories:
            logging.warning("category isn't in our categories")
            # cat = recall to the task1 process
    return