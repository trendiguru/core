__author__ = 'Nadav Paz'

import logging

import pymongo
import cv2
import redis
from rq import Queue
import bson
import boto3

import find_similar_mongo
import background_removal
import Utils


QC_URL = 'http://www.clickworkers.com'
db = pymongo.MongoClient().mydb
images = pymongo.MongoClient().mydb.images
r = redis.Redis()
q2 = Queue('send_to_categorize', connection=r)
q3 = Queue('receive_categories', connection=r)
q4 = Queue('send_to_bb', connection=r)
q5 = Queue('receive_bb', connection=r)
q6 = Queue('send_20s_results', connection=r)
q7 = Queue('receive_20s_results', connection=r)
q8 = Queue('send_last_20', connection=r)
q9 = Queue('receive_final_results', connection=r)


def upload_image(image, name, bucket_name=None):
    image_string = cv2.imencode(".jpg", image)[1].tostring()
    bucket_name = bucket_name or "tg-boxed-faces"
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=bucket_name)
    bucket.put_object(Key="{0}.jpg".format(name), Body=image_string, ACL='public-read', ContentType="image/jpg")
    return "{0}/{1}/{2}.jpg".format("https://s3.eu-central-1.amazonaws.com", bucket_name, name)


def get_person_by_id(person_id):
    image = images.find_one({'people.person_id': person_id})
    for person in image['people']:
        if person['person_id'] == person_id:
            return image, person


def get_item_by_id(item_id):
    image = images.find_one({'people.items.item_id': item_id})
    for person in image['people']:
        try:
            for item in person['items']:
                if item['item_id'] == item_id:
                    return image, {'person': person, 'person_idx': image['people'].index(person)}, \
                           {'item': item, 'item_idx': person['items'].index(item)}
        except:
            logging.warning("No items to this person, continuing..")


# ---------------------------------------------------------------------------------------------------------------------
# q1 - images queue - Web2Py

# FUNCTION 1
def from_image_url_categorization_task(image_url):
    image_obj = images.find_one({"image_urls": {'$elemMatch': {image_url}}})
    if not image_obj:  # new image
        image = background_removal.standard_resize(Utils.get_cv2_img_array(image_url), 400)[0]
        if image is None:
            logging.warning("There's no image in the url!")
            return None
        relevance = background_removal.image_is_relevant(image)
        image_dict = {'image_urls': [], 'relevant': relevance.is_relevant, '_id': bson.ObjectId()}
        image_dict['image_urls'].append(image_url)
        if relevance.is_relevant:
            image_dict['people'] = []
            for face in relevance.faces:
                x, y, w, h = face
                person = {'face': face.tolist(), 'person_id': bson.ObjectId()}
                copy = image.copy()
                cv2.rectangle(copy, (x, y), (x + w, y + h), [0, 255, 0], 2)
                person['url'] = upload_image(copy, str(person['person_id']))
                image_dict['people'].append(person)
                q2.enqueue(send_image_to_qc_categorization, person['url'], str(image_dict['_id']), str(person['id']))
        else:
            logging.warning('image is not relevant, but stored anyway..')
        images.insert(image_dict)
        return
    else:
        if image_url not in image_obj['image_urls']:
            image_obj['image_urls'].append(image_url)
        if image_obj['relevant']:
            logging.warning("Image is in the DB and relevant!")
        else:
            logging.warning("Image is in the DB and not relevant!")
        return image_obj


# END OF FUNCTION 1

# q2

# FUNCTION 2 - Web2py send_image_to_qc_categorization

# q3 - Web2Py


# FUNCTION 3
def from_categories_to_bb_task(person_url, items_list, image_id, person_id):
    if len(items_list) == 0:
        logging.warning("No items in items' list!")
        return None
    # items = determine_final_categories(items_list)
    items_list = []
    for item in items_list:
        item_dict = {'category': item, 'item_id': bson.ObjectId()}
        items_list.append(item_dict)
        q4.enqueue(send_item_to_qc_bb, person_url, image_id, person_id, item_dict)
    images.update_one({'people.person_id': person_id}, {'$set': {'people.$.items': items_list}}, upsert=True)
    return

# END OF FUNCTION 3

# q4

# FUNCTION 4
# Web2Py - send_item_to_qc_bb(person_url, image_id, person_id, item_dict)
# END

# q5 - Web2Py

# FUNCTION 5
# Web2Py - bb_list = get_bb_list_from_qc()
# END


# FUNCTION 6
def from_bb_to_sorting_task(bb, person_id, item_id):
    if len(bb) == 0:
        logging.warning("No bb found")
        return None
    # bb = determine_final_bb(bb_list)  # Yonti's function
    image, person, item = get_item_by_id(item_id)
    fp, results, svg = find_similar_mongo.got_bb(image['image_url'], person_id, item_id, bb, 100, item['category'])
    item['similar_results'] = results
    item['fingerprint'] = fp
    item['svg_url'] = svg
    q6.enqueue(send_initiate_results_to_sorting, results, item_id)
    image['people'][person['person_idx']]['items'][item['item_idx']] = item
    images.replace_one({'people.person': person_id}, image)
    return
# END


"""
# q6

# FUNCTION 6
send_100_results_to_qc_in_20s(copy, results)
# END

# q7

# FUNCTION 7
sorted_results = get_sorted_results_from_qc()
final_20_results = rearrange_results(sorted_results)
# END

# q8

# FUNCTION 8
send_final_20_results_to_qc_in_10s(copy, final_20_results)
# END

# q9

# FUNCTION 9
final_results = get_final_results_from_qc()
insert_final_results(item.id, final_results)
"""
