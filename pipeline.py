__author__ = 'Nadav Paz'

import logging
import datetime
import time

import cv2
import numpy as np
from rq.job import Job
from rq import push_connection
import bson
import tldextract

from . import constants
from . import whitelist
from . import background_removal
from . import Utils
from . import page_results
from . import find_similar_mongo
from .paperdoll import paperdoll_parse_enqueue


db = constants.db
TTL = constants.general_ttl
q1 = constants.q1
q2 = constants.q2
q3 = constants.q3
q4 = constants.q4
q5 = constants.q5
q6 = constants.q6
push_connection(constants.redis_conn)

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


def set_collections(lang):
    if not lang:
        return 'images', 'products'
    else:
        products_collection = 'products_' + lang
        images_collection = 'images_' + lang
        return images_collection, products_collection


# -----------------------------------------------MERGE-FUNCTIONS--------------------------------------------------------


def merge_people_and_insert(jobs_ids, image_dict):
    image_dict["people"] = [Job.fetch(job_id).result for job_id in jobs_ids]

    if all(person is None for person in image_dict["people"]):
        raise RuntimeError("Trying to insert an image, but people is None!")
    insert_result = db.images.insert_one(image_dict)
    if not insert_result.acknowledged:
        raise IOError("Insert failed")


def merge_items_into_person(jobs_ids, person_dict):
    person_dict["items"] = [Job.fetch(job_id).result for job_id in jobs_ids]
    if all(item is None for item in person_dict["items"]):
        raise RuntimeError("All items in person are None!")
    return person_dict


# -----------------------------------------------MAIN-FUNCTIONS---------------------------------------------------------


def start_pipeline(page_url, image_url, lang):
    images_coll, products_coll = set_collections(lang)

    if not is_in_whitelist(page_url):
        return

    images_by_url = db.images.find_one({"image_urls": image_url})
    if images_by_url:
        return

    images_obj_url = db.irrelevant_images.find_one({"image_urls": image_url})
    if images_obj_url:
        return

    image_hash = page_results.get_hash_of_image_from_url(image_url)
    images_obj_hash = db[images_coll].find_one_and_update({"image_hash": image_hash},
                                                          {'$addToSet': {'image_urls': image_url}})
    if images_obj_hash:
        return

    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        raise IOError("'get_cv2_img_array' has failed. Bad image!")

    relevance = background_removal.image_is_relevant(image, use_caffe=False, image_url=image_url)
    image_dict = {'image_urls': [image_url], 'relevant': relevance.is_relevant, 'views': 1,
                  'saved_date': datetime.datetime.utcnow(), 'image_hash': image_hash, 'page_urls': [page_url],
                  'people': []}
    if relevance.is_relevant:
        # There are faces
        people_job_id_jobs = []
        for face in relevance.faces:
            x, y, w, h = face
            person_bb = [int(round(max(0, x - 1.5 * w))), str(y), int(round(min(image.shape[1], x + 2.5 * w))),
                         min(image.shape[0], 8 * h)]
            # These are job whose result is the id of the person job
            people_job_id_jobs.append(q2.enqueue_call(func=get_person_job_id, args=(face.tolist(), person_bb,
                                                                                    products_coll, image_url),
                                                      ttl=TTL, result_ttl=TTL, timeout=TTL))
        q6.enqueue_call(func=wait_for_person_ids, args=([job.id for job in people_job_id_jobs], image_dict),
                        depends_on=people_job_id_jobs, ttl=TTL, result_ttl=TTL, timeout=TTL)

    else:
        db.irrelevant_images.insert_one(image_dict)


def wait_for_person_ids(ids_jobs, image_dict):
    people_jobs = [Job.fetch(job_id) for job_id in ids_jobs]
    q5.enqueue_call(func=merge_people_and_insert, args=([job.id for job in people_jobs], image_dict),
                    depends_on=people_jobs, ttl=TTL, result_ttl=TTL, timeout=TTL)


def get_person_job_id(face, person_bb, products_coll, image_url):
    person = {'face': face, 'person_bb': person_bb}
    image = person_isolation(Utils.get_cv2_img_array(image_url), face)
    paper_job = paperdoll_parse_enqueue.paperdoll_enqueue(image, str(bson.ObjectId()))
    while not paper_job.is_finished or paper_job.is_failed:
        time.sleep(0.5)
    if paper_job.is_failed:
        raise SystemError("Paper-job has failed!")
    elif not paper_job.result:
        raise SystemError("Paperdoll has returned empty results!")
    mask, labels = paper_job.result[:2]
    final_mask = after_pd_conclusions(mask, labels)
    item_jobs = []
    for num in np.unique(final_mask):
        # convert numbers to labels
        category = list(labels.keys())[list(labels.values()).index(num)]
        if category in constants.paperdoll_shopstyle_women.keys():
            item_mask = 255 * np.array(final_mask == num, dtype=np.uint8)

            # These are jobs whose result is an item
            item_jobs.append(q3.enqueue_call(func=create_item, args=(image, category, item_mask, products_coll),
                                             ttl=TTL, result_ttl=TTL, timeout=TTL))

    # The result of this job is a person dict
    merge_person_job = q4.enqueue_call(func=merge_items_into_person, args=([job.id for job in item_jobs],
                                                                           person), depends_on=item_jobs,
                                       ttl=TTL, result_ttl=TTL, timeout=TTL)
    return merge_person_job.id


def create_item(image, category, item_mask, products_coll):
    item = {'category': category}
    try:
        fp, similar_results = find_similar_mongo.find_top_n_results(image, item_mask, 100, category,
                                                                    products_coll)
        item['fp'], item['similar_results'] = fp, similar_results
    except Exception as e:
        print e.message, e.args
    return item
