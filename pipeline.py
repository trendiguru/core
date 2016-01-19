__author__ = 'Nadav Paz'

import time
import logging

import cv2

import numpy as np

import bson

import tldextract
from . import constants
from . import whitelist
from . import background_removal
from . import Utils
from . import page_results
from .paperdoll import paperdoll_parse_enqueue


db = constants.db
TTL = constants.general_ttl
q1 = constants.q1
q2 = constants.q2
q3 = constants.q3

# -----------------------------------------------CO-FUNCTIONS-----------------------------------------------------------


def is_in_whitelist(page_url):
    page_domain = tldextract.extract(page_url).registered_domain
    if page_domain not in whitelist.all_white_lists:
        logging.debug("Domain not in whitelist: {0}. Page: {1}".format(page_domain, page_url))
        return False
    else:
        return True


def person_isolation(image, face):
    x, y, w, h = face
    image_copy = np.zeros(image.shape, dtype=np.uint8)
    x_back = np.max([x - 1.5 * w, 0])
    x_ahead = np.min([x + 2.5 * w, image.shape[1] - 2])
    image_copy[:, int(x_back):int(x_ahead), :] = image[:, int(x_back):int(x_ahead), :]
    return image_copy


def after_pd_conclusions(mask, labels, face=None):
    """
    1. if there's a full-body clothing:
        1.1 add to its' mask - all the rest lower body items' masks.
        1.2 add other upper cover items if they pass the pixel-amount condition/
    2. else -
        2.1 lower-body: decide whether it's a pants, jeans.. or a skirt, and share masks
        2.2 upper-body: decide whether it's a one-part or under & cover
    3. return new mask
    """
    if face:
        ref_area = face[2] * face[3]
        y_split = face[1] + 3 * face[3]
    else:
        ref_area = (np.mean((mask.shape[0], mask.shape[1])) / 10) ** 2
        y_split = np.round(0.4 * mask.shape[0])
    final_mask = mask[:, :]
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
        if (float(item.values()[0]) / (ref_area) > 2) or \
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


# -----------------------------------------------MERGE-FUNCTIONS--------------------------------------------------------


def merge_people_and_insert(image_obj):
    # all people are done, now merge all people (unless job is failed, and then pop it out from people)
    for person in image_obj['people']:
        ready_person = db.people.find_one({'id': person['person_id']})
        person['items'] = ready_person['items']
    db.images.insert_one(image_obj)


def merge_items_and_return_person_job(jobs, items, person_id):
    done = all([job.is_finished for job in jobs.values()])
    # POLLING
    while not done:
        time.sleep(0.2)
        done = all([job.is_finished or job.is_failed for job in jobs.values()])

    # all items are done, now merge all items (unless job is failed, and then pop it out from items)
    for idx, job in jobs.iteritems():
        cur_item = next((item for item in items if item['item_idx'] == idx), None)
        if job.is_failed:
            items[:] = [item for item in items if item['item_idx'] != cur_item['item_idx']]
        else:
            cur_item['fp'], cur_item['similar_results'] = job.result

    # update iip object and return. after this return - the current person_job has to get it!
    db.iip.update_one({'people.$.person_id': person_id}, {'$set': {'people.$.items': items}})
    return


# -----------------------------------------------MAIN-FUNCTIONS---------------------------------------------------------


def start_pipeline(page_url, image_url, lang):
    if not lang:
        products_collection = 'products'
        coll_name = 'images'
        images_collection = db[coll_name]
    else:
        products_collection = 'products_' + lang
        coll_name = 'images_' + lang
        images_collection = db[coll_name]

    if not is_in_whitelist(page_url):
        return

    images_obj_url = db.irrelevant_images.find_one({"image_urls": image_url})
    if images_obj_url:
        return

    image_hash = page_results.get_hash_of_image_from_url(image_url)
    images_obj_hash = images_collection.find_one_and_update({"image_hash": image_hash},
                                                            {'$push': {'image_urls': image_url}})
    if images_obj_hash:
        return

    iip_obj = db.iip.find_one({"image_urls": image_url}) or db.iip.find_one({"image_hash": image_hash})
    if iip_obj:
        return

    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        return

    relevance = background_removal.image_is_relevant(image, use_caffe=False, image_url=image_url)
    image_dict = {'image_urls': [image_url], 'relevant': relevance.is_relevant,
                  'image_hash': image_hash, 'page_urls': [page_url], 'people': []}
    if relevance.is_relevant:
        # There are faces
        people_jobs = []
        idx = 0
        for face in relevance.relevant_faces:
            x, y, w, h = face
            person_bb = [int(round(max(0, x - 1.5 * w))), str(y), int(round(min(image.shape[1], x + 2.5 * w))),
                         min(image.shape[0], 8 * h)]
            person = {'face': face, 'person_id': str(bson.ObjectId()), 'person_idx': idx, 'items': [],
                      'person_bb': person_bb}
            image_copy = person_isolation(image, face)
            people_jobs.append(q1.enqueue_call(func=person_job, args=(person['person_id'], image_copy,
                                                                      products_collection, images_collection),
                                               ttl=TTL, result_ttl=TTL, timeout=TTL))
            if db.iip.insert_one(person).acknowledged:
                image_dict['people'].append(person)
                idx += 1
        q3.enqueue_call(func=merge_people_and_insert, args=(image_dict, ), depends_on=people_jobs, ttl=TTL,
                        result_ttl=TTL, timeout=TTL)
    else:
        db.irrelevant_image.insert_one(image_dict)


def person_job(person_id, image, products_coll, images_coll):
    paper_job = paperdoll_parse_enqueue.paperdoll_enqueue(image, person_id, async=False)
    mask, labels = paper_job.result[:2]
    mask = after_pd_conclusions(mask)
    item_jobs = {}
    for item in items:
        item_jobs['idx'] = q2.enqueue_call(func=item_job, args=(nyet1, nyet2)))
        return merge_items_and_return_person_job(item_jobs, items, person_id)