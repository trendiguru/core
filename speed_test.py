__author__ = 'Nadav Paz'

import time

import bson
from rq import Queue

from paperdoll import paperdoll_parse_enqueue

from . import Utils

from .constants import redis_conn
from . import background_removal
from . import page_results
from .constants import db

images = db.final_coll_st
iip = db.iip
mid1 = db.mid1_st
mid2 = db.mid2_st
q1 = Queue('new_images', connection=redis_conn)
q2 = Queue('from_paperdoll_to_similar_results', connection=redis_conn)
q3 = Queue('find_similar', connection=redis_conn)


def speed_test(part, batch):
    if part == 1:
        all = db.dynamic_fp.find().limit(batch)
        i = 0
        start = time.time()
        for doc in all:
            Queue('new_images', connection=redis_conn).enqueue(start_process_st, 'speed_test.fazz',
                                                               doc['images']['XLarge'], lang='st')
            i += 1
            if i % 100 == 0:
                print "start process did {0} items in {1} seconds".format(mid1.count(), time.time() - start)
        first = mid1.count()
        while mid1.count() - first > 0:
            first = mid1.count()
            time.sleep(0.01)
        sumtime = time.time() - start
        print "start process is done. did {0} items in {1} seconds".format(mid1.count(),
                                                                           sumtime)
        return float(mid1.count()) / sumtime
    elif part == 2:
        all = mid1.find().limit(batch)
        i = 0
        start = time.time()
        for doc in all:
            paper_job = paperdoll_parse_enqueue.paperdoll_enqueue(doc['image_urls'][0], doc['people'][0]['person_id'],
                                                                  queue_name='pd')
            doc['people'][0]['job_id'] = paper_job.id
            del doc['_id']
            mid2.insert(doc)
            print "pd did {0} items in {1} seconds".format(mid2.count(), time.time() - start)
        while mid2.count() < batch - 1:
            time.sleep(0.3)
        print "pd is done. did {0} items in {1} seconds".format(mid2.count(), time.time() - start)
        # elif part == 3:


def start_process_st(page_url, image_url, lang=None):
    if not lang:
        coll_name = 'images'
        images_collection = db[coll_name]
    else:
        coll_name = 'images_' + lang
        images_collection = db[coll_name]

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
    relevance = background_removal.image_is_relevant(image, use_caffe=True)
    image_dict = {'image_urls': [image_url], 'relevant': relevance.is_relevant, 'page_urls': [page_url], 'people': []}
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
                # image_copy = paperdolls.person_isolation(image, face)
                image_dict['people'].append(person)
                # paper_job = paperdoll_parse_enqueue.paperdoll_enqueue(image_copy, person['person_id'])
                # q1.enqueue(from_paperdoll_to_similar_results, person['person_id'], paper_job.id,
                # products_collection=products_collection, images_collection=coll_name, depends_on=paper_job)
                idx += 1
        else:
            # no faces, only general positive human detection
            person = {'face': [], 'person_id': str(bson.ObjectId()), 'person_idx': 0, 'items': [], 'person_bb': None}
            image_dict['people'].append(person)
            # paper_job = paperdoll_parse_enqueue.paperdoll_enqueue(image, person['person_id'])
            # q1.enqueue(from_paperdoll_to_similar_results, person['person_id'], paper_job.id,
            # products_collection=products_collection, images_collection=coll_name, depends_on=paper_job)
    # else:  # if not relevant
    # images_collection.insert_one(image_dict)
    #     return
    mid1.insert_one(image_dict)