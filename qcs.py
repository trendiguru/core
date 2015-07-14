__author__ = 'Nadav Paz'

import logging
import os
import binascii
import boto3
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
                image_s3_url = upload_image(copy, person['person_id'])
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


def upload_image(image, name, bucket_name=None):
    image_string = cv2.imencode(".jpg", image)[1].tostring()
    bucket_name = bucket_name or "boxed_faces"
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=bucket_name)
    bucket.put_object(Key=name, Body=image_string, ACL='public-read', ContentType="image/jpg")
    return "{0}/{1}/{2}".format("https://s3.eu-central-1.amazonaws.com", bucket_name, name)
