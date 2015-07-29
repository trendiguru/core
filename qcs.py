__author__ = 'Nadav Paz'

import logging

import requests
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
callback_url = "https://extremeli.trendi.guru/api/nadav/index"
db = pymongo.MongoClient().mydb
images = pymongo.MongoClient().mydb.images
r = redis.Redis()
q1 = Queue('images_queue', connection=r)
q2 = Queue('send_to_categorize', connection=r)
q3 = Queue('send_to_bb', connection=r)
q4 = Queue('send_20s_results', connection=r)
q5 = Queue('send_last_20', connection=r)
q6 = Queue('receive_data_from_qc', connection=r)


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


def decode_task(args, vars, data):  # args(list) = person_id, vars(dict) = task, data(dict) = QC results
    if vars["task_id"] is 'categorization':
        from_categories_to_bb_task(data['items'], args[0])
    elif vars["task_id"] is 'bb':
        from_bb_to_sorting_task(data['bb'], args[0], args[1])
        # elif vars["task_id"] is 'first_sorting':
        # dole_out_work()
        # else:
        # finish_work()
    return


# ---------------------------------------------------------------------------------------------------------------------
# optional data arrangements:
# 1.  only by url: callback url -
# "https://extremeli.trendi.guru/api/nadav/index/image_id/person_id/item_id?task_id=bounding_boxing"
#     in this case we know how many args we have because of the type of the task (e.g item bounding_boxing => 3 args).
# 1.1 maybe more efficient way is: "https://extremeli.trendi.guru/api/nadav/index/item_id?task_id=bounding_boxing"
#     and by task_id we would know that it is an item which we activated by.
# 2.  by task_id: create and send task_id in each kind of task. when we get a post back we search the task id in a tasks
#     table that we built. from the task document we understand what to do with the info we got.
# we will go with 1.1 for now !

# q1 - images queue -  from Web2Py


def from_image_url_to_categorization_task(image_url):
    image_obj = images.find_one({"image_urls": image_url})
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
                q2.enqueue(send_image_to_qc_categorization, person['url'], str(person['id']))
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


def send_image_to_qc_categorization(person_url, person_id):
    data = {"callback_url": callback_url + '/' + person_id + '?task=categorization',
            "person_url": person_url}
    req = requests.post(QC_URL, data)
    return req.status_code


# q6 - decode_task, from Web2Py


def from_categories_to_bb_task(items_list, person_id):
    if len(items_list) == 0:
        logging.warning("No items in items' list!")
        return None
    # items = category_tree.CatNode.determine_final_categories(items_list) # sergey's function
    image, person = get_person_by_id(person_id)
    person_url = person['url']
    for item in items_list:
        item_dict = {'category': item, 'item_id': bson.ObjectId()}
        items_list.append(item_dict)
        q3.enqueue(send_item_to_qc_bb, person_url, person_id, item)
    images.update_one({'people.person_id': person_id}, {'$set': {'people.$.items': items_list}}, upsert=True)
    return


def send_item_to_qc_bb(person_url, person_id, item_dict):
    data = {"callback_url": callback_url + '/' + person_id + '/' + item_dict['item_id'] + '?task=bb',
            "person_url": person_url}
    req = requests.post(QC_URL, data)
    return req.status_code


# q6 - decode_task, from Web2Py


def from_bb_to_sorting_task(bb, person_id, item_id):
    if len(bb) == 0:
        logging.warning("No bb found")
        return None
    # bb = determine_final_bb(bb_list)  # Yonti's function
    image, person, item = get_item_by_id(item_id)
    fp, results, svg = find_similar_mongo.got_bb(image['image_urls'][0], person_id, item_id, bb, 100, item['category'])
    item['similar_results'] = results
    item['fingerprint'] = fp
    item['svg_url'] = svg
    q4.enqueue(send_initiate_results_to_sorting, results, person_id, item_id)
    image['people'][person['person_idx']]['items'][item['item_idx']] = item
    images.replace_one({'people.person': person_id}, image)
    return


"""
# FUNCTION 6
send_100_results_to_qc_in_20s(copy, results)
# END

# q6 - decode_task, from Web2Py

# FUNCTION 7
from 20's to 5's(results, item_id)
q5.enqueue(send_final_20_20_results_to_qc_in_10s, copy, final_20_results)
# END

# FUNCTION 8
send_final_20_results_to_qc_in_10s(copy, final_20_results)
# END

# q6 - decode_task, from Web2Py

# FUNCTION 9
update result for the last time
"""
