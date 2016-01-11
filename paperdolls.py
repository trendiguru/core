__author__ = 'Nadav Paz'

import logging
import datetime
import time

import numpy as np
import pymongo
import bson
import cv2
from rq import Queue
from rq.job import Job

import boto3
import page_results
from .paperdoll import paperdoll_parse_enqueue
from . import find_similar_mongo
from . import background_removal
from . import Utils
from . import constants
from .constants import db
from .constants import redis_conn


folder = '/home/ubuntu/paperdoll/masks/'
QC_URL = 'https://extremeli.trendi.guru/api/fake_qc/index'
callback_url = "https://extremeli.trendi.guru/api/nadav/index"
images = db.images
iip = db.iip
q1 = Queue('find_similar', connection=redis_conn)
q2 = Queue('find_top_n', connection=redis_conn)
# sys.stdout = sys.stderr
TTL = constants.general_ttl


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


def person_isolation(image, face):
    x, y, w, h = face
    image_copy = np.zeros(image.shape, dtype=np.uint8)
    x_back = np.max([x - 1.5 * w, 0])
    x_ahead = np.min([x + 2.5 * w, image.shape[1] - 2])
    image_copy[:, int(x_back):int(x_ahead), :] = image[:, int(x_back):int(x_ahead), :]
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


def blacklisted_term_in_url(page_url):
    for term in constants.blacklisted_terms:
        if term in page_url:
            return True
        return False


# ----------------------------------------------MAIN-FUNCTIONS----------------------------------------------------------


def start_process(page_url, image_url, lang=None):
    # db.monitoring.update_one({'queue': 'start_process'}, {'$inc': {'count': 1}})
    if not lang:
        products_collection = 'products'
        coll_name = 'images'
        images_collection = db[coll_name]
    else:
        products_collection = 'products_' + lang
        coll_name = 'images_' + lang
        images_collection = db[coll_name]

    # IF URL IS BLACKLISTED - put in blacklisted_urls
    # {"page_url":"hoetownXXX.com","image_urls":["hoetownXXX.com/yomama1.jpg","yomama2.jpg"]}
    if blacklisted_term_in_url(page_url):
        if db.blacklisted_urls.find_one({"page_url": page_url}):
            db.blacklisted_urls.update_one({"page_url": page_url},
                                           {"$push": {"image_urls": image_url}})
        else:
            db.blacklisted_urls.insert_one({"page_url": page_url,"image_urls":[image_url]})
        return

    # IF IMAGE IN IRRELEVANT_IMAGES
    images_obj_url = db.irrelevant_images.find_one({"image_urls": image_url})
    if images_obj_url:
        return

    # IF IMAGE EXISTS IN IMAGES BY URL
    images_obj_url = images_collection.find_one({"image_urls": image_url})
    if images_obj_url:
        return
    
    # IF URL HAS NO IMAGE IN IT
    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        return
    
    # IF IMAGE EXISTS IN IMAGES BY HASH (WITH ANOTHER URL)
    image_hash = page_results.get_hash_of_image_from_url(image_url)
    images_obj_hash = images_collection.find_one_and_update({"image_hash": image_hash},
                                                            {'$push': {'image_urls': image_url}})
    if images_obj_hash:
        return

    # IF IMAGE IN PROCESS BY URL/HASH
    iip_obj = iip.find_one({"image_urls": image_url}) or iip.find_one({"image_hash": image_hash})
    if iip_obj:
        return

    # NEW_IMAGE !!
    print "Start process image shape: " + str(image.shape)
    relevance = background_removal.image_is_relevant(image, use_caffe=False, image_url=image_url)
    image_dict = {'image_urls': [image_url], 'relevant': relevance.is_relevant,
                  'image_hash': image_hash, 'page_urls': [page_url], 'people': []}
    if relevance.is_relevant:
        if not isinstance(relevance.faces, list):
            relevant_faces = relevance.faces.tolist()
        else:
            relevant_faces = relevance.faces
        if len(relevant_faces) > 0:
            # There are faces
            idx = 0
            for face in relevant_faces:
                x, y, w, h = face
                person_bb = [int(round(max(0, x - 1.5 * w))), y, int(round(min(image.shape[1], x + 2.5 * w))),
                             min(image.shape[0], 8 * h)]
                person = {'face': face, 'person_id': str(bson.ObjectId()), 'person_idx': idx, 'items': [],
                          'person_bb': person_bb}
                image_copy = person_isolation(image, face)
                image_dict['people'].append(person)
                db.monitoring.update_one({'queue': 'pd'}, {'$inc': {'count': 1}})
                paper_job = paperdoll_parse_enqueue.paperdoll_enqueue(image_copy, person['person_id'])
                db.monitoring.update_one({'queue': 'find_similar'}, {'$inc': {'count': 1}})
                q1.enqueue_call(func=from_paperdoll_to_similar_results, args=(person['person_id'], paper_job.id, 100,
                                                                              products_collection, coll_name),
                                depends_on=paper_job, ttl=TTL, result_ttl=TTL, timeout=TTL)
                idx += 1
        else:
            # no faces, only general positive human detection
            person = {'face': [], 'person_id': str(bson.ObjectId()), 'person_idx': 0, 'items': [], 'person_bb': None}
            image_dict['people'].append(person)
            paper_job = paperdoll_parse_enqueue.paperdoll_enqueue(image, person['person_id'])
            q1.enqueue_call(func=from_paperdoll_to_similar_results, args=(person['person_id'], paper_job.id, 100,
                                                                          products_collection, coll_name),
                            depends_on=paper_job, ttl=TTL, result_ttl=TTL, timeout=TTL)
        iip.insert_one(image_dict)
    else:  # if not relevant
        logging.warning('image is not relevant, but stored anyway..')
        db.irrelevant_images.insert_one(image_dict)
        return


def from_paperdoll_to_similar_results(person_id, paper_job_id, num_of_matches=100, products_collection='products',
                                      images_collection='images'):
    start = time.time()
    products_collection = products_collection
    images_collection = db[images_collection]
    paper_job_results = job_result_from_id(paper_job_id)
    if paper_job_results[3] != person_id:
        print
        raise ValueError("paper job refers to another image!!! oy vey !!! filename: {0} & person_id: {1}".format(
            paper_job_results[3], person_id))
    mask, labels = paper_job_results[:2]
    image_obj, person = get_person_by_id(person_id, iip)
    if person is not None and 'face' in person and len(person['face']) > 0:
        final_mask = after_pd_conclusions(mask, labels, person['face'])
    else:
        final_mask = after_pd_conclusions(mask, labels)
    image = Utils.get_cv2_img_array(image_obj['image_urls'][0])
    if image is None:
        iip.delete_one({'_id': image_obj['_id']})
        raise SystemError("image came back empty from Utils.get_cv2..")
    idx = 0
    items = []
    jobs = {}
    for num in np.unique(final_mask):
        # convert numbers to labels
        category = list(labels.keys())[list(labels.values()).index(num)]
        if category in constants.paperdoll_shopstyle_women.keys():
            item_mask = 255 * np.array(final_mask == num, dtype=np.uint8)
            shopstyle_cat_local_name = constants.paperdoll_shopstyle_women_jp_categories[category]['name']
            item_dict = {"category": category, 'item_id': str(bson.ObjectId()), 'item_idx': idx,
                         'saved_date': datetime.datetime.now(), 'category_name': shopstyle_cat_local_name}
            # svg_name = find_similar_mongo.mask2svg(
            # item_mask,
            #     str(image_obj['_id']) + '_' + person['person_id'] + '_' + item_dict['category'],
            #     constants.svg_folder)
            # item_dict["svg_url"] = constants.svg_url_prefix + svg_name
            items.append(item_dict)
            db.monitoring.update_one({'queue': 'find_top_n'}, {'$inc': {'count': 1}})
            jobs[idx] = q2.enqueue_call(func=find_similar_mongo.find_top_n_results, args=(image, item_mask,
                                                                                          num_of_matches,
                                                                                          item_dict['category'],
                                                                                          products_collection),
                                        ttl=TTL, result_ttl=TTL, timeout=TTL)
            idx += 1
    # print "everyone was sent to find_top_n after {0} seconds.".format(time.time() - start)
    done = all([job.is_finished for job in jobs.values()])
    while not done:
        time.sleep(0.2)
        done = all([job.is_finished or job.is_failed for job in jobs.values()])
    # print "all find_top_n is done after {0} seconds".format(time.time() - start)
    for idx, job in jobs.iteritems():
        cur_item = next((item for item in items if item['item_idx'] == idx), None)
        if job.is_failed:
            items[:] = [item for item in items if item['item_idx'] != cur_item['item_idx']]
        else:
            cur_item['fp'], cur_item['similar_results'] = job.result
    new_image_obj = iip.find_one_and_update({'people.person_id': person_id}, {'$set': {'people.$.items': items}},
                                            return_document=pymongo.ReturnDocument.AFTER)
    total_time = 0
    while not new_image_obj:
        if total_time < 30:
            # print "image_obj after update is None!.. waiting for it.. total time is {0}".format(total_time)
            time.sleep(2)
            total_time += 2
            new_image_obj = iip.find_one_and_update({'people.person_id': person_id},
                                                    {'$set': {'people.$.items': items}},
                                                    return_document=pymongo.ReturnDocument.AFTER)
        else:
            print "exceeded.."
            break
    else:
        image_obj = new_image_obj
    if person['person_idx'] == len(image_obj['people']) - 1:
        # print "inserted to db.images after {0} seconds".format(time.time() - start)
        a = images_collection.insert_one(image_obj)
        iip.delete_one({'_id': image_obj['_id']})
        logging.warning("# of images inserted to db.images: {0}".format(a.acknowledged * 1))


def get_results_now(page_url, image_url, collection='products_jp'):
    a = time.time()
    # IF IMAGE EXISTS IN DEMO BY URL
    images_obj_url = db.demo.find_one({"image_urls": image_url})
    if images_obj_url:
        return page_results.merge_items(images_obj_url)

    # IF IMAGE EXISTS IN IMAGES BY URL
    images_obj_url = images.find_one({"image_urls": image_url})
    if images_obj_url:
        return page_results.merge_items(images_obj_url)

    # IF URL HAS NO IMAGE IN IT
    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        return

    # IF IMAGE EXISTS IN IMAGES BY HASH (WITH ANOTHER URL)
    image_hash = page_results.get_hash_of_image_from_url(image_url)
    images_obj_hash = images.find_one_and_update({"image_hash": image_hash}, {'$push': {'image_urls': image_url}})
    if images_obj_hash:
        return page_results.merge_items(images_obj_hash)

    # IF IMAGE IN PROCESS BY URL/HASH
    iip_obj = iip.find_one({"image_urls": image_url}) or iip.find_one({"image_hash": image_hash})
    if iip_obj:
        return

    # NEW_IMAGE !!
    relevance = background_removal.image_is_relevant(image, True, image_url)
    image_dict = {'image_urls': [image_url], 'relevant': relevance.is_relevant,
                  'image_hash': image_hash, 'page_urls': [page_url], 'people': []}
    if relevance.is_relevant:
        idx = 0
        if len(relevance.faces):
            if not isinstance(relevance.faces, list):
                relevant_faces = relevance.faces.tolist()
            else:
                relevant_faces = relevance.faces
            for face in relevant_faces:
                image_copy = person_isolation(image, face)
                person = {'face': face, 'person_id': str(bson.ObjectId()), 'person_idx': idx,
                          'items': []}
                image_dict['people'].append(person)
                print "untill pd: {0}".format(time.time() - a)
                mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(image_copy, async=False).result[:3]
                start = time.time()
                final_mask = after_pd_conclusions(mask, labels, person['face'])
                item_idx = 0
                jobs = {}
                for num in np.unique(final_mask):
                    # convert numbers to labels
                    category = list(labels.keys())[list(labels.values()).index(num)]
                    if category in constants.paperdoll_shopstyle_women.keys():
                        item_mask = 255 * np.array(final_mask == num, dtype=np.uint8)
                        shopstyle_cat = constants.paperdoll_shopstyle_women[category]
                        item_dict = {"category": shopstyle_cat, 'item_id': str(bson.ObjectId()), 'item_idx': item_idx,
                                     'saved_date': datetime.datetime.now()}
                        svg_name = find_similar_mongo.mask2svg(
                            item_mask,
                            str(image_dict['image_hash']) + '_' + person['person_id'] + '_' + item_dict['category'],
                            constants.svg_folder)
                        item_dict["svg_url"] = constants.svg_url_prefix + svg_name
                        jobs[item_idx] = q2.enqueue_call(func=find_similar_mongo.find_top_n_results, args=(image,
                                                                                                           item_mask,
                                                                                                           100,
                                                                                                           item_dict[
                                                                                                               'category'],
                                                                                                           collection),
                                                         ttl=TTL,
                                                         result_ttl=TTL, timeout=TTL)
                        person['items'].append(item_dict)
                        item_idx += 1
                done = all([job.is_finished for job in jobs.values()])
                b = time.time()
                print "done is {0}".format(done)
                while not done:
                    time.sleep(0.2)
                    done = all([job.is_finished for job in jobs.values()])
                print "done is {0} after {1} seconds with {2} items..".format(done, time.time() - b,
                                                                              len(person['items']))
                for idx, job in jobs.iteritems():
                    cur_item = next((item for item in person['items'] if item['item_idx'] == idx), None)
                    cur_item['fp'], cur_item['similar_results'] = job.result
                idx += 1
                image_dict['people'].append(person)
        else:
            print "no faces, went caffe.."
            person = {'face': [], 'person_id': str(bson.ObjectId()), 'person_idx': 0, 'items': []}
            image_dict['people'].append(person)
            mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(image, async=False).result[:3]
            final_mask = after_pd_conclusions(mask, labels)
            item_idx = 0
            jobs = {}
            for num in np.unique(final_mask):
                # convert numbers to labels
                category = list(labels.keys())[list(labels.values()).index(num)]
                if category in constants.paperdoll_shopstyle_women.keys():
                    item_mask = 255 * np.array(final_mask == num, dtype=np.uint8)
                    shopstyle_cat = constants.paperdoll_shopstyle_women[category]
                    item_dict = {"category": shopstyle_cat, 'item_id': str(bson.ObjectId()), 'item_idx': item_idx,
                                 'saved_date': datetime.datetime.now()}
                    svg_name = find_similar_mongo.mask2svg(
                        item_mask,
                        str(image_dict['image_hash']) + '_' + person['person_id'] + '_' + item_dict['category'],
                        constants.svg_folder)
                    item_dict["svg_url"] = constants.svg_url_prefix + svg_name
                    jobs[idx] = q2.enqueue_call(func=find_similar_mongo.find_top_n_results, args=(image,
                                                                                                  item_mask, 100,
                                                                                                  item_dict['category'],
                                                                                                  collection), ttl=TTL,
                                                result_ttl=TTL, timeout=TTL)
                    person['items'].append(item_dict)
                    item_idx += 1
            image_dict['people'].append(person)
        db.demo.insert_one(image_dict)
        print "all took {0} seconds".format(time.time() - start)
        return page_results.merge_items(image_dict)
    else:  # if not relevant
        return