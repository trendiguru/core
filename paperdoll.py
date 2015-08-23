__author__ = 'Nadav Paz'

import logging
import random
import copy

import requests
import numpy as np
import pymongo
import cv2
import redis
from rq import Queue
import bson

from paperdoll import paperdoll_parse_enqueue
import boto3
import find_similar_mongo
import background_removal
import Utils
import constants


QC_URL = 'https://extremeli.trendi.guru/api/fake_qc/index'
callback_url = "https://extremeli.trendi.guru/api/nadav/index"
db = pymongo.MongoClient().mydb
images = pymongo.MongoClient().mydb.images
r = redis.Redis()
q1 = Queue('images_queue', connection=r)
q2 = Queue('paperdoll', connection=r)
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
                    person["person_idx"] = image['people'].index(person)
                    item["item_idx"] = person['items'].index(item)
                    return image, person, item
        except:
            logging.warning("No items to this person, continuing..")
            return None, None, None


def decode_task(args, vars, data):  # args(list) = person_id, vars(dict) = task, data(dict) = QC results
    if vars["task_id"] == 'categorization':
        from_qc_get_votes(args[1], data['results'], data['votes'], vars['voting_stage'])


def set_voting_stage(n_stage, item_id):
    image, person_dict, item_dict = get_item_by_id(item_id)
    person_idx = person_dict['person_idx']
    item_idx = item_dict['item_idx']
    image['people'][person_idx]['items'][item_idx]['voting_stage'] = n_stage
    image.pop('_id')
    images.replace_one({"people.items.item_id": item_id}, image)


def get_voting_stage(item_id):
    image, person_dict, item_dict = get_item_by_id(item_id)
    person_idx = person_dict['person_idx']
    item_idx = item_dict['item_idx']
    item = image['people'][person_idx]['items'][item_idx]
    if 'voting_stage' in item:
        return item['voting_stage']
    else:  # no voting stage set yet,. so set to 0
        set_voting_stage(0, item_id)
        return 0


def get_paperdoll_data(image):
    image = background_removal.standard_resize(image, 400)
    mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(image, async=False)
    after_paperdoll_work_conclusions(mask, labels)


def person_isolation(image, face):
    x, y, w, h = face
    x_back = np.max([x - 2 * w, 0])
    x_ahead = np.min([x + 3 * w, image.shape[1] - 2])
    back_mat = np.zeros((image.shape[0], x_back, 3), dtype=np.uint8)
    ahead_mat = np.zeros((image.shape[0], image.shape[1] - x_ahead, 3), dtype=np.uint8)
    image_copy = np.concatenate((back_mat, image[:, x_back:x_ahead, :], ahead_mat), 1)
    return image_copy


def create_gc_mask(image, pd_mask, bgnd_mask):
    item_bb = bb_from_mask(pd_mask)
    item_gc_mask = background_removal.paperdoll_item_mask(pd_mask, item_bb)
    after_gc_mask = background_removal.simple_mask_grabcut(image, item_gc_mask)  # (255, 0) mask
    final_mask = cv2.bitwise_and(bgnd_mask, after_gc_mask)
    return final_mask  # (255, 0) mask


def bb_from_mask(mask):
    r, c = mask.nonzero()
    x = min(c)
    y = min(r)
    w = max(c) - x + 1
    h = max(r) - y + 1
    return x, y, w, h

# ---------------------------------------------------------------------------------------------------------------------


def start_process(image_url):
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
                person = {'face': face.tolist(), 'person_id': str(bson.ObjectId())}
                image_copy = person_isolation(image, face)
                person['url'] = upload_image(image_copy, str(person['person_id']))
                image_dict['people'].append(person)
                q2.enqueue(get_paperdoll_data, person['url'], person['person_id'])
        else:
            logging.warning('image is not relevant, but stored anyway..')
        images.insert(image_dict)
    else:
        if image_url not in image_obj['image_urls']:
            image_obj['image_urls'].append(image_url)
        if image_obj['relevant']:
            logging.warning("Image is in the DB and relevant!")
        else:
            logging.warning("Image is in the DB and not relevant!")
        return image_obj


def after_paperdoll_work_conclusions(mask, labels):
    bla
    bla
    for item in final_items:
        bli
        blo


def db_update_to_sorting_task(item_id):
    image, person, item = get_item_by_id(item_id)
    fp, results, svg = find_similar_mongo.got_bb(image['image_urls'][0], person_id, item_id, bb, 100, item['category'])
    item['similar_results'] = results
    item['fingerprint'] = fp
    item['svg_url'] = svg
    image['people'][person['person_idx']]['items'][item['item_idx']] = item
    image.pop('_id')
    images.replace_one({'image_urls': {'$in': image['image_urls']}}, image)
    dole_out_work(item_id)


def dole_out_work(item_id):
    """
    dole out images. this function is the engine of the sorting process.
    :param item_id:
    :return:
    """
    voting_stage = get_voting_stage(item_id)
    # make sure theres at least 1 worker per image
    image, person_dict, item_dict = get_item_by_id(item_id)
    person = person_dict['person']
    person_idx = person_dict['person_idx']
    item = item_dict['item']
    item_idx = item_dict['item_idx']
    results = image['people'][person_idx]['items'][item_idx]['similar_items']
    assert (constants.N_workers[voting_stage] * constants.N_pics_per_worker[voting_stage] /
            len(results) > 1)  # len similar_items instead of constants.N_top_results_to_show

    if results is None:
        logging.warning('oh man no similar items found')
        return None

    results_indexer = []
    for i in range(0, constants.N_pics_per_worker[voting_stage]):
        results_indexer.append(np.linspace(i * constants.N_workers[voting_stage],
                                           (i + 1) * constants.N_workers[voting_stage] - 1,
                                           constants.N_workers[voting_stage], dtype=np.uint8))
    for i in range(0, constants.N_workers[voting_stage]):  # divide results into chunks for N workers
        # I don't know what to do with this stage, seems like it is unnecessary:
        final_image_index = min(i + constants.N_pics_per_worker[voting_stage] - 1,
                                len(results) - 1)  # the min deals with case where there's fewer images for last worker
        indices_filter = []
        for group in results_indexer:
            indices_filter.append(group.tolist().pop(random.randint(0, len(group))))
        chunk_of_results = [results[j]['image']['sizes']['XLarge']['url'] for j in indices_filter]
        q4.enqueue(send_results_chunk_to_qc, person['url'], person['person_id'], item['item_id'], chunk_of_results,
                   voting_stage)


def send_results_chunk_to_qc(person_url, person_id, item_id, chunk, voting_stage):
    data = {"callback_url": callback_url + '/' + person_id + '/' + item_id + '?task_id=sorting?&stage=' + voting_stage,
            "person_url": person_url, 'results': chunk}
    req = requests.post(QC_URL, data)
    return req.status_code


# Here i am assuming I get votes in the form of a list of numbers or 'not relevant',
# the same length as the similar_items
def from_qc_get_votes(item_id, chunk_of_similar_items, chunk_of_votes, voting_stage):
    image, person_dict, item_dict = get_item_by_id(item_id)
    print('image before:' + str(image))
    person_idx = person_dict['person_idx']
    item_idx = item_dict['item_idx']
    item = image['people'][person_idx]['items'][item_idx]
    if 'votes' in item:
        extant_votes = item['votes']
        extant_similar_items = item['similar_items']
    else:
        extant_votes = []
        extant_similar_items = []
    tot_votes, combined_similar_items, combined_votes = \
        add_results(extant_similar_items, extant_votes, chunk_of_similar_items, chunk_of_votes)

    logging.debug('tot votes: ' + str(tot_votes))
    logging.debug('comb.sim.items: ' + str(combined_similar_items))
    logging.debug('comb.votes: ' + str(combined_votes))
    # enough votes done already to take results and move to next stage?
    print('votingStage: ' + str(voting_stage))
    logging.debug('votingStage: ' + str(voting_stage))
    enough_votes = constants.N_pics_per_worker[voting_stage] * constants.N_workers[voting_stage]
    if tot_votes >= enough_votes:
        combined_similar_items, combined_votes = order_results(combined_similar_items, combined_votes)
        set_voting_stage(voting_stage + 1, item_id)

    item['votes'] = combined_votes
    item['similar_items'] = combined_similar_items
    image['people'][person_idx]['items'][item_idx]['votes'] = item[
        'votes']  # maybe unnecessary since item['votes'] prob writes into image
    image['people'][person_idx]['items'][item_idx]['similar_items'] = item[
        'similar_items']  # maybe unnecessary since item['votes'] prob writes into image
    image.pop('_id')
    images.replace_one({"people.items.item_id": item_id}, image)
    print('image written: ' + str(image))


def add_results(extant_similar_items, extant_votes, new_similar_items, new_votes):
    """
    add in new votes to current votes, making sure to check if new votes are on things
    already voted for, if so tack onto end of vote list
    :param extant_similar_items: list of items like [itemA,itemB]
    :param extant_votes:  list of vote lists like [[voteA1,voteA2],[voteB1,voteB2]]
    :param new_similar_items:  like previous list
    :param new_votes: flat list of votes like [voteA3,voteC3]
    :return: new extant_votes list and similar_items list
    """
    assert (len(extant_similar_items) == len(extant_votes))
    assert (len(new_similar_items) == len(new_votes))
    # check if votes are on same items
    modified_extant_similar_items = copy.copy(extant_similar_items)
    modified_extant_votes = copy.copy(extant_votes)

    for i in range(0, len(new_similar_items)):
        match_flag = False
        for j in range(0, len(extant_similar_items)):
            if new_similar_items[i] == extant_similar_items[j]:  # got a vote for already-voted-on item
                modified_extant_votes[j].append(new_votes[i])
                match_flag = True
                break
        if match_flag is False:
            # got a vote on a new item
            modified_extant_similar_items.append(new_similar_items[i])
            modified_extant_votes.append([new_votes[i]])

    assert (len(modified_extant_similar_items) == len(modified_extant_votes))
    tot_votes = 0
    for j in range(0, len(extant_votes)):
        tot_votes += len(extant_votes[j])

    return tot_votes, modified_extant_similar_items, modified_extant_votes


def order_results(combined_similar_items, combined_votes):
    """
    smush multiple votes into a single combined vote
    :param combined_similar_items:
    :param combined_votes:
    :return:
    """
    for j in range(0, len(combined_similar_items)):
        combined_votes[j] = combine_votes(combined_votes[j])
    sorted_votes = sorted(combined_votes)
    # the following works but is dangerous since it sorts by tuples of [vote,item]
    sorted_items = [item for (vote, item) in sorted(zip(combined_votes, combined_similar_items))]
    return sorted_items, sorted_votes


def combine_votes(combined_votes):
    """
    take multiple votes and smush into one
    :return:
    """
    # if all are numbers:
    # return average
    #    if all are 'not relevant'
    #    return 'not relevant'
    # if some are numbers and some are 'not relevant':
    #    if the majority voted not relevant:
    #        return not relevant
    #    otherwise:
    #    return average
    #    vote,
    #    with not relevant guys counted as 0 or -1 or something like that
    not_relevant_score = -1
    n = 0
    sum = 0
    non_int_flag = False
    int_flag = False
    if len(combined_votes) == 0:
        logging.debug('no votes to combine')
        return None
    for vote in combined_votes:
        if type(vote) is int or type(vote) is float:
            n += 1
            sum += vote
            int_flag = True
        else:
            n += 1
            sum += not_relevant_score
            non_int_flag = True
    if not int_flag:  # all non-int/float
        return 'not relevant'
    if not non_int_flag:  # all int/float
        return float(sum) / n
    return float(
        sum) / n  # some int/float and some not-relevant, above 3 lines can be combined, but ths is clearer to me

    # # combine multiple votes on same item
    # combined_results = [all_results[0]]
    # for i in range(0, len(all_results)):
    # for j in range(i + 1, all_results):
    #         if all_results[i] != all_results[j]:  # different results being voted on by 2 dudes
    #             combined_results = combined_results.append(all_results[j])
    #             combined_ratings = combined_ratings.append(all_ratings[j])
    #         else:  # same result being voted on by 2 dudes
    #             combined_ratings[i] = combined_ratings[j]




