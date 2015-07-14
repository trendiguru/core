__author__ = 'Nadav Paz'

import logging
import os
import binascii
import boto3
import pymongo
import cv2

import background_removal
import Utils


db = pymongo.MongoClient().mydb


def from_image_url_to_task1(image_url):
    images = db.images
    image_dict = db.images.find_one({"image_url": image_url})
    if not image_dict:
        image = background_removal.standard_resize(Utils.get_cv2_img_array(image_url), 400)[0]
        if image is None:
            logging.warning("There's no image in the url!")
            return None
        relevance = background_removal.image_is_relevant(image)
        image_obj_id = images.insert({'image_url': image_url,
                                      'relevant': relevance.is_relevant,
                                      'people': []})
        image_dict = images.find_one({'_id': image_obj_id})
        for idx, face in enumerate(relevance.faces):
            x, y, w, h = face
            image_dict['people'].append({'face': face.tolist(), 'person_id': binascii.hexlify(os.urandom(32))})
            copy = image.copy()
            cv2.rectangle(copy, (x, y), (x + w, y + h), [0, 255, 0], 2)
            cv2.imwrite('/home/ubuntu/Dev/qcs' + '/' + str(idx) + '.jpg', copy)
        return image_dict
    else:
        if not image_dict['relevant']:
            logging.warning("image already exists and not relevant!")
            return image_dict





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


def upload_image(image, name, bucket_name=IMAGE_BUCKET):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=bucket_name)
    bucket.put_object(Key=name, Body=image)