__author__ = 'Nadav Paz'

import logging
import os
import binascii

import pymongo
import cv2
import redis
from rq import Queue

import boto3

import background_removal
import Utils


images = pymongo.MongoClient().mydb.images
r = redis.Redis()
q = Queue(connection=r)


def from_image_url_to_task1(image_url):
    image_obj = images.find_one({"image_url": image_url})
    if not image_obj:
        image = background_removal.standard_resize(Utils.get_cv2_img_array(image_url), 400)[0]
        if image is None:
            logging.warning("There's no image in the url!")
            return None
        relevance = background_removal.image_is_relevant(image)
        if relevance.is_relevant:
            image_obj = {'image_url': image_url,
                         'relevant': relevance.is_relevant,
                         'people': []}
            for face in relevance.faces:
                x, y, w, h = face
                person = {'face': face, 'person_id': binascii.hexlify(os.urandom(32))}
                copy = image.copy()
                cv2.rectangle(copy, (x, y), (x + w, y + h), [0, 255, 0], 2)
                image_s3_url = upload_image(copy, person['person_id'])
                person['url'] = image_s3_url
                image_obj['people'].append(person)
                # q.enqueue(send_task1, image_s3_url, image_obj)
            image_obj_id = images.insert(image_obj)
            image_obj = images.find_one({'_id': image_obj_id})
            return image_obj
        else:
            logging.warning("image is not relevant!")
            return image_obj
    else:
        logging.warning('image already exists..')
        return image_obj


def validate_cats_and_send_to_bb(category, image_id, person_id):
    image = images.find_one({'_id': image_id})
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


def upload_image(image, name, bucket_name=None):
    image_string = cv2.imencode(".jpg", image)[1].tostring()
    bucket_name = bucket_name or "tg-boxed-faces"
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=bucket_name)
    bucket.put_object(Key=name + {0}.__format__('.jpg'), Body=image_string, ACL='public-read', ContentType="image/jpg")
    return "{0}/{1}/{2}".format("https://s3.eu-central-1.amazonaws.com", bucket_name, name)
