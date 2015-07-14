__author__ = 'Nadav Paz'

import logging
import os
import binascii

import pymongo
import cv2
from redis import Redis
from rq import Queue

import background_removal
import Utils


db = pymongo.MongoClient().mydb
q = Queue(connection=Redis())


def from_image_url_to_task1(image_url):
    images = db.images
    image_dict = db.images.find_one({"image_url": image_url})
    if not image_dict:
        image = background_removal.standard_resize(Utils.get_cv2_img_array(image_url), 400)[0]
        if image is None:
            logging.warning("There's no image in the url!")
            return None
        relevance = background_removal.image_is_relevant(image)
        if relevance.is_relevant:
            image_dict = {'image_url': image_url,
                          'relevant': relevance.is_relevant,
                          'people': []}
            for face in relevance.faces:
                x, y, w, h = face
                person = {'face': face, 'person_id': binascii.hexlify(os.urandom(32))}
                copy = image.copy()
                cv2.rectangle(copy, (x, y), (x + w, y + h), [0, 255, 0], 2)
                image_s3_url = push_to_bucket(person['person_id'], copy)
                person['url'] = image_s3_url
                image_dict['people'].append(person)
                q.enqueue(send_task1, image_s3_url, image_dict)
            image_obj_id = images.insert(image_dict)
            image_dict = images.find_one({'_id': image_obj_id})
            return image_dict
        else:
            logging.warning("image is not relevant!")
            return image_dict
    else:
        logging.warning('image already exists..')
        return image_dict


# def send_task1(image, image_dict):

def validate_cats_and_send_to_bb(category, image_id, person_id):
    image = db.images.find_one({'_id': image_id})
    if image is None:
        logging.warning('image_id inValid')
        # do something with it
    for person in image['people']:
        if person['person_id'] == person_id:
            current_person = person
    if len(current_person['items']) == 0:
        current_person['items'] = []
    current_person['items'].append({'category': category})
    return image