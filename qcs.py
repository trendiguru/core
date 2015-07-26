__author__ = 'Nadav Paz'

# theirs
import logging

import pymongo
import cv2
import redis
from rq import Queue
import bson
import boto3
import numpy as np

import find_similar_mongo
import background_removal
import Utils
import constants


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
            return person


def get_item_by_id(item_id):
    image = images.find_one({'people.items.item_id': item_id})
    for person in image['people']:
        for item in person['items']:
            if item['item_id'] == item_id:
                return item


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
        image_dict = {'image_url': image_url, 'relevant': relevance.is_relevant, '_id': bson.ObjectId()}
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
        # TODO - understand which details are already stored and react accordingly
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
    items = determine_final_categories(items_list)
    items_list = []
    for item in items:
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
    bb = determine_final_bb(bb_list)  # Yonti's function
    image = images.find_one({'people.items.item_id': item_id}, {})
    category = get_item_by_id(item_id)['category']
    fp, results, svg = find_similar_mongo.got_bb(image['image_url'], person_id, item_id, bb, N_top_results_to_show,
                                                 category)

# q6 - send_20s_resuqlts
# FUNCTION 6
def send_100_results_to_qc_in_20s(original_image_url, results):
    for i in range(0, constants.N_workers[0]):  #divide results into chunks for N workers
        final_image_index = min(i + constants.N_pics_per_worker - 1,
                                len(results) - 1)  # the min deals with case where there's fewer images for last worker
        chunk_of_results = results[i:final_image_index]
        q6.enqueue(send_many_results_to_qcs, original_image_url, chunk_of_results)

# QUEUE FUNC FOR FUNCTION 6
def send_many_results_to_qcs(original_image, chunk_of_results):
    payload = {'original_image': original_image, 'results_to_sort': chunk_of_results}
    req = requests.post(QC_URL, data=payload)
    return req.ok

# END OF QUEUE FUNC FOR FUNCTION 6
# END  FUNCTION 6

def average_bbs(bblist):
    avg_box = [0, 0, 0, 0]
    n = 0
    for bb in bblist:
        # print('avg'+str(avg_box))
        # print('bb'+str(bb))
        avg_box = np.add(avg_box, bb)
        # print('avg after'+str(avg_box))
        n = n + 1
    avg_box = np.int(np.divide(avg_box, n))
    return avg_box


# q7
# q7 - receive_20s_results
# FUNCTION 7
# assumption
def set_voting_stage(N_stage, item_id):
    '''
    this can be replaced by a different persistent storage scheme than storing
    in the image db
    :param N_stage:
    :param item_id:
    :return:
    '''
    # db_image = images.find_one({'people.items.item_id': item_id})
    write_result = images.update({"people.items.item_id": item_id},
                                 {"$set": {"people.items.voting_stage": N_stage}})


def get_voting_stage(item_id):
    image = images.find_one({'people.items.item_id': item_id})
    if 'voting_stage' in image['people']['items']:
        return image['people']['items']['voting_stage']
    else:  # no voting stage set yet,. so set to 0
        set_voting_stage(0, item_id)
        return 0


def receive_votes(similar_items, voting_results):
    got_all_votes, combined_votes = combine_results(similar_items, voting_results)
    if got_all_votes:
        ordered_results = order_results(combined_votes)
        set_voting_stage(get_voting_stage() + 1)


# if persistent_voting_stage == final_stage:
# return top_N (or do whatever else needs to be done when voting is over)
#        otherwise:
#           dole_out_work(top_N,voting_stage=persistent_voting_stage)

def combine_results(similar_items, voting_results):
    final_20_results = None
    image = images.find_one({'people.items.item_id': item_id})
    for person in image['people']:
        for item in person['items']:
            if item['item_id'] == item_id:
                if 'votes' in item:
                    item['votes'].append([chunk_of_results, ratings])
                    n_votes_so_far = len(item['votes']) / 2
                    if n_votes_so_far >= constants.N_workers[0]:  # enough votes rec'd
                        final_20_results = rearrange_results(item['votes'])
                else:
                    item['votes'] = [[chunk_of_results, ratings]]
                # SOMEONE PLS REVIEW THE LINE BELOW - i've never used the mongodot notation before
                write_result = images.update({"people.items.item_id": item_id},
                                             {"$set": {"people.items.votes": item['votes']}})
    return got_all_votes


def persistently_store_votes(votes):
    '''
    a list of persistently stored dictionaries
    we need a list (i think) since results have to be stored for more than one input image at a time
    an alternate way to do this is to store in the image database
    :param list:
    :return:
    '''
    pass


# def combine_results(similar_items, voting_results):
# for similar_item in similar_items:
#        if similar_item in persistent_votes:
#            persistent_votes[similar_item].append(ith voting result)
#        if there are enough votes in persistent_votes
#            persistent_votes[similar_item] = combine_votes(persistent_votes[similar_item])

def rearrange_results(votes):
    '''
    Take a bunch of votes. CHeck for duplicate votes (two or more dudes voting on same item).
    Tote up all the results and send back in order
    :param votes:
    :return:
    '''
    all_results = []
    all_ratings = []
    for results, ratings in votes:
        all_results = all_results.append(results)
        all_ratings = all_ratings.append(ratings)

    # combine multiple votes on same item
    combined_results = [all_results[0]]
    combined_ratings = [all_ratings[0]]
    for i in range(0, len(all_results)):
        for j in range(i + 1, all_results):
            if all_results[i] != all_results[j]:  # different results being voted on by 2 dudes
                combined_results = combined_results.append(all_results[j])
                combined_ratings = combined_ratings.append(all_ratings[j])
            else:  # same result being voted on by 2 dudes
                combined_ratings[i] = combined_ratings[j]

def send_many_results_to_qcs(original_image, chunk_of_results):
    payload = {'original_image': original_image, 'results_to_sort': chunk_of_results}
    req = requests.post(QC_URL, data=payload)
    return req.ok

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
