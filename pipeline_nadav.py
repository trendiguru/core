__author__ = 'Nadav Paz'

import time

import bson

from . import background_removal
from . import Utils
from . import page_results
from .paperdoll import paperdoll_parse_enqueue
from .constants import db
from .constants import q1
from .constants import q2
from .constants import general_ttl as TTL


def merge_people_and_insert(jobs, people, image_id):
    done = all([job.is_finished for job in jobs.values()])
    # POLLING
    while not done:
        time.sleep(0.2)
        done = all([job.is_finished or job.is_failed for job in jobs.values()])

    image_obj = db.iip.find_one({'_id': image_id})
    # all people are done, now merge all people (unless job is failed, and then pop it out from people)
    for job in jobs:
        cur_person = next((person for person in people if person['person_idx'] == idx), None)
        if job.is_finished:
            image_obj['people'].append(person)

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


------------------------------------------------------------------------------------------------------------------------


def start_pipeline(page_url, image_url, lang):
    if not lang:
        products_collection = 'products'
        coll_name = 'images'
        images_collection = db[coll_name]
    else:
        products_collection = 'products_' + lang
        coll_name = 'images_' + lang
        images_collection = db[coll_name]
    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        return
    print "Starting pipeline with image: {0}".format(image_url)
    image_hash = page_results.get_hash_of_image_from_url(image_url)
    relevance = background_removal.image_is_relevant(image, use_caffe=False, image_url=image_url)
    image_dict = {'image_urls': [image_url], 'relevant': relevance.is_relevant,
                  'image_hash': image_hash, 'page_urls': [page_url], 'people': []}
    if relevance.is_relevant:
        # There are faces
        people_jobs = {}
        idx = 0
        for face in relevance.relevant_faces:
            x, y, w, h = face
            person_bb = [int(round(max(0, x - 1.5 * w))), y, int(round(min(image.shape[1], x + 2.5 * w))),
                         min(image.shape[0], 8 * h)]
            person = {'face': face, 'person_id': str(bson.ObjectId()), 'person_idx': idx, 'items': [],
                      'person_bb': person_bb}
            image_copy = background_removal.person_isolation(image, face)
            image_dict['people'].append(person)
            people_jobs[idx] = q1.enqueue_call(func=person_job, args=(person['person_id'], image_copy,
                                                                      products_collection, images_collection),
                                               ttl=TTL, result_ttl=TTL, timeout=TTL)
            idx += 1

        image_dict = db.iip.insert_one(image_dict)
        merge_people_and_insert(people_jobs, image_dict['people'], image_dict['_id'])
    else:
        db.irrelevant_image.insert_one(image_dict)


def person_job(person_id, bla2):
    paper_job = paperdoll_parse_enqueue.paperdoll_enqueue(image_copy, person['person_id'])
    mask, labels = paper_job.result[:2]
    mask = after_pd(mask)
    item_jobs = {}
    for item in items:
        item_jobs['idx'] = q2.enqueue_call(func=item_job, args=(nyet1, nyet2)))
        return merge_items_and_return_person_job(item_jobs, items, person_id)
