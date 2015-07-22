__author__ = 'Nadav Paz'

# theirs
import logging
import os
import binascii
import requests

import pymongo
import cv2
import redis
from rq import Queue
import boto3
import numpy as np




# ours
import background_removal
import Utils
import constants
import find_similar_mongo


QC_URL = 'http://www.clickworkers.com'
# Is there only one URL for the whole shebang or different urls for diffrent tasks?
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

#I propose to combine q6 and q8 into one q, and q7 and q9 into another, and they take arguments of how many results to show each worker and how many workers JR
def upload_image(image, name, bucket_name=None):
    image_string = cv2.imencode(".jpg", image)[1].tostring()
    bucket_name = bucket_name or "tg-boxed-faces"
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=bucket_name)
    bucket.put_object(Key="{0}.jpg".format(name), Body=image_string, ACL='public-read', ContentType="image/jpg")
    return "{0}/{1}/{2}.jpg".format("https://s3.eu-central-1.amazonaws.com", bucket_name, name)


# ---------------------------------------------------------------------------------------------------------------------
# q1 - images queue - Web2Py


# FUNCTION 1 - determine if relevant, send to categorize (q2)
def from_image_url_to_task1(image_url):
    image_obj = images.find_one({"image_url": image_url})
    if not image_obj:  # new image
        image = background_removal.standard_resize(Utils.get_cv2_img_array(image_url), 400)[0]
        if image is None:
            logging.warning("There's no image in the url!")
            return None
        relevance = background_removal.image_is_relevant(image)
        image_dict = {'image_url': image_url, 'relevant': relevance.is_relevant}
        if relevance.is_relevant:
            image_dict['people'] = []
            for face in relevance.faces:
                x, y, w, h = face
                person = {'face': face.tolist(), 'person_id': binascii.hexlify(os.urandom(32))}
                copy = image.copy()
                cv2.rectangle(copy, (x, y), (x + w, y + h), [0, 255, 0], 2)
                image_s3_url = upload_image(copy, person['person_id'])
                person['url'] = image_s3_url
                image_dict['people'].append(person)
                q2.enqueue(send_image_to_qc_categorization, image_s3_url, image_dict)
        else:
            logging.warning('image is not relevant, but stored anyway..')
        images.insert(image_dict)
        image_obj = images.find_one({'image_url': image_url})
        return image_obj
    else:
        # TODO - understand which details are already stored and react accordingly
        return image_obj
        # END OF FUNCTION 1


# q2
# FUNCTION 2
def send_image_to_qc_categorization(image_s3_url):
    payload = {'image_url': image_s3_url}
    req = requests.post(QC_URL, data=payload)
    return req.ok
# END OF FUNCTION 2

# q3 - Web2Py
# FUNCTION 3
def get_categorization_from_qc(items_list, person_id):
    person = db.images.find_one({'people.person_id': person_id})
    if person is None:
        logging.warning("Person wasn't found in DataBase!")
        return None
    if len(items_list) == 0:
        logging.warning("No items in items' list!")
        q2.enqueue(send_image_to_qc_categorization, QC_URL, person['url'])
        return None
    items = determine_final_categories(items_list)
    for item in items:
        item_dict = {'category': item, 'item_id': binascii.hexlify(os.urandom(32))}
        person.append(item_dict)


def determine_final_categories(items_list):
    pass


# END OF FUNCTION 3

# q4 - 'send_to_bb'
# FUNCTION 4
def send_bb_task_to_qc(img_url, item_category):
    q4.enqueue(send_bb_to_qc_categorization, img_url, item_category)


# QUEUE FUNCTION FOR FUNCTION 4
def send_bb_to_qc_categorization(image_s3_url, category):
    payload = {'image_url': image_s3_url, 'category': category}
    req = requests.post(QC_URL, data=payload)
    return req.ok


# END OF QUEUE FUNCTION FOR FUNCTION 4
# END OF FUNCTION 4

# q5 - receive_bb
# FUNCTION 5
def receive_bb_from_qc(image_url, image_id, item_id, bb):
    '''
    Get bb from qc, see if there's enough to do voting. If so, call find_similar and send
    those results to the next stage
    :param image_url:
    :param image_id:
    :param item_id:
    :param bb:
    :return:
    '''
    # bb_list = get_bb_list_from_qc()
    n_bbs_received_so_far, bb_list = put_bb_into_db(image_url, bb)
    if n_bbs_received_so_far >= constants.N_bb_votes_required:  # probably no need for >= instead of ==
        final_bb = determine_final_bb(bb_list)
    n_results = constants.N_top_results_to_show[0]
    item_category = None  # how do we get this info....
    fp, results, svg = find_similar_mongo.got_bb(image_url, image_id, item_id, final_bb, n_results, item_category)
    send_100_results_to_qc_in_20s(image_url, results)


def determine_final_bb(bb_list):
    '''
    kick out illegal bbs (too small, maybe beyond img frame later on).
    take average of all remaining. kick out anything with iou < threshold
    :param bb_list:
    :return:
    '''

    good_bblist = []
    for bb in bb_list:
        if Utils.legal_bounding_box(bb):
            good_bblist.append(bb)
    if len(good_bblist) == 1:  # if thees only one bb, return it
        return good_bblist[0]
    if len(good_bblist) == 0:
        logging.warning('no good bbs in list')
        return None
    avg_bb = average_bbs(good_bblist)
    # print('avg bb:'+str(avg_bb))
    # check if any of the boxes are way out
    good_intersection_bblist = []
    for bb in good_bblist:
        if Utils.intersectionOverUnion(bb, avg_bb) >= constants.bb_iou_threshold:
            good_intersection_bblist.append(bb)
    if good_intersection_bblist != []:  # got at least one good bb
        improved_result = average_bbs(good_intersection_bblist)
        return improved_result
    else:
        logging.warning('no good intersections found')
        return None


def average_bbs(bblist):
    avg_box = [0, 0, 0, 0]
    n = 0
    for bb in bblist:
        # print('avg'+str(avg_box))
        # print('bb'+str(bb))
        avg_box = np.add(avg_box, bb)
        # print('avg after'+str(avg_box))
        n = n + 1
    avg_box = np.int_(np.divide(avg_box, n))
    return avg_box


def put_bb_into_db(image_url, bb):
    pass


# END


# q6 - send_20s_resuqlts
# FUNCTION 6
# constants.N_top_results_to_show[i]
# constants.N_workers[j]
# constants.N_pics_per_worker[k]constants
def send_100_results_to_qc_in_20s(original_image_url, results):
    for i in range(0, constants.N_workers[0]):  #divide results into chunks for N workers
        final_image_index = min(i + constants.N_pics_per_worker - 1,
                                len(results) - 1)  #deal with case where there's fewer images for last worker
        chunk_of_results = results[i:final_image_index]
        q6.enqueue(send_many_results_to_qcs, original_image_url, chunk_of_results)


# QUEUE FUNC FOR FUNCTION 6
def send_many_results_to_qcs(original_image, chunk_of_results):
    payload = {'original_image': original_image, 'results_to_sort': chunk_of_results}
    req = requests.post(QC_URL, data=payload)
    return req.ok


# END OF QUEUE FUNC FOR FUNCTION 6
# END  FUNCTION 6

# q7 - receive_20s_results
# FUNCTION 7
# assumption
def get_20s_results(chunk_of_results, ratings):
    #final_20_results = rearrange_results(sorted_results)
    pass


# HELPER FUNCTION FOR FUNCTION 6
def send_many_results_to_qcs(original_image, chunk_of_results):
    payload = {'original_image': original_image, 'results_to_sort': chunk_of_results}
    req = requests.post(QC_URL, data=payload)
    return req.ok

# END OF HELPER FOR FUNCTION 6
# END

"""
# q8 - send_last_20

# FUNCTION 8
send_final_20_results_to_qc_in_10s(copy, final_20_results)
# END

# q9 - receive_final_results

# FUNCTION 9
final_results = get_final_results_from_qc()
insert_final_results(item.id, final_results)
"""


