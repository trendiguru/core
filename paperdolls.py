__author__ = 'Nadav Paz'

import logging
import datetime
import sys

import numpy as np
import pymongo
import cv2
import redis
from rq import Queue
import bson
from rq.job import Job

import page_results
from .paperdoll import paperdoll_parse_enqueue
import boto3
import find_similar_mongo
import background_removal
import Utils
import constants


folder = '/home/ubuntu/paperdoll/masks/'
QC_URL = 'https://extremeli.trendi.guru/api/fake_qc/index'
callback_url = "https://extremeli.trendi.guru/api/nadav/index"
db = pymongo.MongoClient().mydb
images = pymongo.MongoClient().mydb.images
iip = pymongo.MongoClient().mydb.iip
r = redis.Redis()
q1 = Queue('find_similar', connection=r)
sys.stdout = sys.stderr


# ----------------------------------------------CO-FUNCTIONS------------------------------------------------------------


def upload_image(image, name, bucket_name=None):
    image_string = cv2.imencode(".jpg", image)[1].tostring()
    bucket_name = bucket_name or "tg-boxed-faces"
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=bucket_name)
    bucket.put_object(Key="{0}.jpg".format(name), Body=image_string, ACL='public-read', ContentType="image/jpg")
    return "{0}/{1}/{2}.jpg".format("https://s3.eu-central-1.amazonaws.com", bucket_name, name)


def get_person_by_id(person_id, collection=iip):
    image = collection.find_one({'people.person_id': person_id})
    if image:
        for person in image['people']:
            if person['person_id'] == person_id:
                return image, person
    else:
        return None, None


def get_item_by_id(item_id, collection=iip):
    image = collection.find_one({'people.items.item_id': item_id})
    for person in image['people']:
        try:
            for item in person['items']:
                if item['item_id'] == item_id:
                    person["person_idx"] = image['people'].index(person)
                    item["item_idx"] = person['items'].index(item)
                    return image, person, item
        except:
            logging.warning("No items to this person, continuing..")
            return None, None, None


def after_pd_conclusions(mask, labels, face):
    """
    1. if there's a full-body clothing:
        1.1 add to its' mask - all the rest lower body items' masks.
        1.2 add other upper cover items if they pass the pixel-amount condition/
    2. else -
        2.1 lower-body: decide whether it's a pants, jeans.. or a skirt, and share masks
        2.2 upper-body: decide whether it's a one-part or under & cover
    3. return new mask
    """
    if type(mask) is list:
        mask = np.array(mask)
    final_mask = mask.copy()
    mask_sizes = {"upper_cover": [], "upper_under": [], "lower_cover": [], "lower_under": [], "whole_body": []}
    for num in np.unique(mask):
        item_mask = 255 * np.array(mask == num, dtype=np.uint8)
        category = list(labels.keys())[list(labels.values()).index(num)]
        print "W2P: checking {0}".format(category)
        for key, item in constants.paperdoll_categories.iteritems():
            if category in item:
                mask_sizes[key].append({num: cv2.countNonZero(item_mask)})
    # 1
    for item in mask_sizes["whole_body"]:
        if (float(item.values()[0]) / (face[2] * face[3]) > 2) or \
                (len(mask_sizes["upper_cover"]) == 0 and len(mask_sizes["upper_under"]) == 0) or \
                (len(mask_sizes["lower_cover"]) == 0 and len(mask_sizes["lower_under"]) == 0):
            print "W2P: That's a {0}".format(list(labels.keys())[list(labels.values()).index((item.keys()[0]))])
            item_num = item.keys()[0]
            for num in np.unique(mask):
                cat = list(labels.keys())[list(labels.values()).index(num)]
                # 1.1, 1.2
                if cat in constants.paperdoll_categories["lower_cover"] or \
                                cat in constants.paperdoll_categories["lower_under"] or \
                                cat in constants.paperdoll_categories["upper_under"]:
                    final_mask = np.where(mask == num, item_num, final_mask)
            return final_mask
    # 2, 2.1
    sections = {"upper_cover": 0, "upper_under": 0, "lower_cover": 0, "lower_under": 0}
    max_item_count = 0
    max_cat = 9
    print "W2P: That's a 2-part clothing item!"
    for section in sections.keys():
        for item in mask_sizes[section]:
            if item.values()[0] > max_item_count:
                max_item_count = item.values()[0]
                max_cat = item.keys()[0]
                sections[section] = max_cat
        # share masks
        if max_item_count > 0:
            for item in mask_sizes[section]:
                cat = list(labels.keys())[list(labels.values()).index(item.keys()[0])]
                # 2.1, 2.2
                if cat in constants.paperdoll_categories[section]:
                    final_mask = np.where(mask == item.keys()[0], max_cat, final_mask)
            max_item_count = 0
    y_split = face[1] + 3 * face[3]
    for item in mask_sizes['whole_body']:
        for i in range(0, mask.shape[0]):
            if i <= y_split:
                for j in range(0, mask.shape[1]):
                    if mask[i][j] == item.keys()[0]:
                        final_mask[i][j] = sections["upper_under"] or sections["upper_cover"] or 0
            else:
                for j in range(0, mask.shape[1]):
                    if mask[i][j] == item.keys()[0]:
                        final_mask[i][j] = sections["lower_cover"] or sections["lower_under"] or 0
    return final_mask


def person_isolation(image, face):
    x, y, w, h = face
    x_back = np.max([x - 1.5 * w, 0])
    x_ahead = np.min([x + 2.5 * w, image.shape[1] - 2])
    back_mat = np.zeros((image.shape[0], x_back, 3), dtype=np.uint8)
    ahead_mat = np.zeros((image.shape[0], image.shape[1] - x_ahead, 3), dtype=np.uint8)
    image_copy = np.concatenate((back_mat, image[:, x_back:x_ahead, :], ahead_mat), 1)
    return image_copy


def create_gc_mask(image, pd_mask, bgnd_mask, skin_mask):
    item_bb = bb_from_mask(pd_mask)
    item_gc_mask = background_removal.paperdoll_item_mask(pd_mask, item_bb)
    after_gc_mask = background_removal.simple_mask_grabcut(image, item_gc_mask)  # (255, 0) mask
    final_mask = cv2.bitwise_and(bgnd_mask, after_gc_mask)
    final_mask = cv2.bitwise_and(skin_mask, final_mask)
    return final_mask  # (255, 0) mask


def bb_from_mask(mask):
    r, c = mask.nonzero()
    x = min(c)
    y = min(r)
    w = max(c) - x + 1
    h = max(r) - y + 1
    return x, y, w, h


def search_existing_images(page_url):
    ex_list = []
    query = images.find({'page_urls': page_url}, {'relevant': 1, 'image_urls': 1})
    for doc in query:
        ex_list.append(doc)
    return ex_list


def clear_collection(collection):
    print "before: " + str(collection.count()) + " docs"
    for doc in collection.find():
        if doc['relevant'] is True and len(doc['people'][0]['items']) == 0:
            collection.delete_one({'_id': doc['_id']})
    print "after: " + str(collection.count()) + " docs"


def job_result_from_id(job_id, job_class=Job, conn=None):
    conn = conn or constants.redis_conn
    job = job_class.fetch(job_id, connection=conn)
    return job.result


# ----------------------------------------------MAIN-FUNCTIONS----------------------------------------------------------

def start_process(page_url, image_url):
    # IF URL HAS NO IMAGE IN IT
    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        return

    # IF IMAGE EXISTS IN IMAGES BY URL
    images_obj_url = images.find_one({"image_urls": image_url})
    if images_obj_url:
        return

    # IF IMAGE EXISTS IN IMAGES BY HASH (WITH ANOTHER URL)
    image_hash = page_results.get_hash_of_image_from_url(image_url)
    images_obj_hash = images.find_one_and_update({"image_hash": image_hash}, {'$push': {'image_urls': image_url}})
    if images_obj_hash:
        return

    # IF IMAGE IN PROCESS BY URL/HASH
    iip_obj = iip.find_one({"image_urls": image_url}) or iip.find_one({"image_hash": image_hash})
    if iip_obj:
        return

    # NEW_IMAGE !!
    image = background_removal.standard_resize(image, 400)[0]
    relevance = background_removal.image_is_relevant(image)
    image_dict = {'image_urls': [image_url], 'relevant': relevance.is_relevant,
                  'image_hash': image_hash, 'page_urls': [page_url]}
    if relevance.is_relevant:
        image_dict['people'] = []
        relevant_faces = relevance.faces.tolist()
        idx = 0
        for face in relevant_faces:
            person = {'face': face, 'person_id': str(bson.ObjectId()), 'person_idx': idx,
                      'items': []}
            image_copy = person_isolation(image, face)
            person['url'] = upload_image(image_copy, str(person['person_id']))
            image_dict['people'].append(person)
            paper_job = paperdoll_parse_enqueue.paperdoll_enqueue(image_url)
            q1.enqueue(from_paperdoll_to_similar_results, person['person_id'], paper_job.id, depends_on=paper_job)
            idx += 1
    else:  # if not relevant
        logging.warning('image is not relevant, but stored anyway..')
        images.insert(image_dict)
        return
    iip.insert(image_dict)


def from_paperdoll_to_similar_results(person_id, paper_job_id, num_of_matches=100):
    paper_job_results = job_result_from_id(paper_job_id)
    mask, labels = paper_job_results[:2]
    image_obj, person = get_person_by_id(person_id, iip)
    final_mask = after_pd_conclusions(mask, labels, person['face'])
    image = Utils.get_cv2_img_array(person['url'])
    items = []
    idx = 0
    for num in np.unique(final_mask):
        # convert numbers to labels
        category = list(labels.keys())[list(labels.values()).index(num)]
        if category in constants.paperdoll_shopstyle_women.keys():
            item_mask = 255 * np.array(mask == num, dtype=np.uint8)
            shopstyle_cat = constants.paperdoll_shopstyle_women[category]
            item_dict = {"category": shopstyle_cat, 'item_id': str(bson.ObjectId()), 'item_idx': idx,
                         'saved_date': datetime.datetime.now()}
            svg_name = find_similar_mongo.mask2svg(
                item_mask,
                str(image_obj['_id']) + '_' + person['person_id'] + '_' + item_dict['category'],
                constants.svg_folder)
            item_dict["svg_url"] = constants.svg_url_prefix + svg_name
            item_dict['fp'], item_dict['similar_results'] = find_similar_mongo.find_top_n_results(image, item_mask,
                                                                                                  num_of_matches,
                                                                                                  item_dict['category'])
            items.append(item_dict)
            idx += 1
    image_obj = iip.find_one_and_update({'people.person_id': person_id}, {'$set': {'people.$.items': items}},
                                        return_document=pymongo.ReturnDocument.AFTER)
    if person['person_idx'] == len(image_obj['people']) - 1:
        images.insert(image_obj)
        logging.warning("Done! image was successfully inserted to the DB images!")

