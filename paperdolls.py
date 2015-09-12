__author__ = 'Nadav Paz'

import logging
import random
import copy
import datetime
import time
import sys

import requests
import numpy as np
import pymongo
import cv2
import redis
from rq import Queue
import bson

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
q1 = Queue('images_queue', connection=r)
q2 = Queue('paperdoll', connection=r)
q4 = Queue('send_20s_results', connection=r)
q5 = Queue('send_last_20', connection=r)
q6 = Queue('receive_data_from_qc', connection=r)
sys.stdout = sys.stderr


def upload_image(image, name, bucket_name=None):
    image_string = cv2.imencode(".jpg", image)[1].tostring()
    bucket_name = bucket_name or "tg-boxed-faces"
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=bucket_name)
    bucket.put_object(Key="{0}.jpg".format(name), Body=image_string, ACL='public-read', ContentType="image/jpg")
    return "{0}/{1}/{2}.jpg".format("https://s3.eu-central-1.amazonaws.com", bucket_name, name)


def get_person_by_id(person_id, collection=iip):
    image = collection.find_one({'people.person_id': person_id})
    for person in image['people']:
        if person['person_id'] == person_id:
            return image, person


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


def decode_task(args, vars, data):  # args(list) = person_id, vars(dict) = task, data(dict) = QC results
    if vars["task_id"] == 'categorization':
        from_qc_get_votes(args[1], data['results'], data['votes'], vars['voting_stage'])


def set_voting_stage(n_stage, item_id):
    image, person_dict, item_dict = get_item_by_id(item_id, iip)
    person_idx = person_dict['person_idx']
    item_idx = item_dict['item_idx']
    image['people'][person_idx]['items'][item_idx]['voting_stage'] = n_stage
    image.pop('_id')
    iip.replace_one({"people.items.item_id": item_id}, image)


def get_voting_stage(item_id):
    image, person_dict, item_dict = get_item_by_id(item_id, iip)
    person_idx = person_dict['person_idx']
    item_idx = item_dict['item_idx']
    item = image['people'][person_idx]['items'][item_idx]
    if 'voting_stage' in item:
        return item['voting_stage']
    else:  # no voting stage set yet,. so set to 0
        set_voting_stage(0, item_id)
        return 0


def get_paperdoll_data(image, person_id):
    mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(image, async=False)
    final_mask = after_pd_conclusions(mask, labels)
    from_paperdoll_to_similar_results(person_id, final_mask, labels)


def after_pd_conclusions(mask, labels):
    """
    1. if there's a full-body clothing:
        1.1 add to its' mask - all the rest lower body items' masks.
        1.2 add other upper cover items if they pass the pixel-amount condition/
    2. else -
        2.1 lower-body: decide whether it's a pants, jeans.. or a skirt, and share masks
        2.2 upper-body: decide whether it's a one-part or under & cover
    3. return new mask
    """
    # TODO - relations between head-size and masks sizes
    print "W2P: got into after_pd_conclusions!"
    final_mask = mask.copy()
    mask_sizes = {"upper_cover": [], "upper_under": [], "lower_cover": [], "lower_under": [], "whole_body": []}
    for num in np.unique(mask):
        item_mask = 255 * np.array(mask == num, dtype=np.uint8)
        category = list(labels.keys())[list(labels.values()).index(num)]
        for key, item in constants.paperdoll_categories.iteritems():
            if category in item:
                mask_sizes[key].append({num: cv2.countNonZero(item_mask)})
    # 1
    for item in mask_sizes["whole_body"]:
        if item.values()[0] > 30000:
            print "W2P: That's a {0)".format(list(labels.keys())[list(labels.values()).index((item.keys()[0]))])
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
    sections = ["upper_cover", "upper_under", "lower_cover", "lower_under"]
    max_item_count = 0
    max_cat = 9
    print "W2P: That's a 2-part clothing item!"
    for section in sections:
        for item in mask_sizes[section]:
            if item.values()[0] > max_item_count:
                max_item_count = item.values()[0]
                max_cat = item.keys()[0]
        # share masks
        if max_item_count > 0:
            for item in mask_sizes[section]:
                cat = list(labels.keys())[list(labels.values()).index(item.keys()[0])]
                # 2.1, 2.2
                if cat in constants.paperdoll_categories[section]:
                    final_mask = np.where(mask == item.keys()[0], max_cat, final_mask)
            max_item_count = 0
    return final_mask


def person_isolation(image, face):
    x, y, w, h = face
    x_back = np.max([x - 1.5 * w, 0])
    x_ahead = np.min([x + 2.5 * w, image.shape[1] - 2])
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


def search_existing_images(page_url):
    ex_list = []
    query = images.find({'page_urls': page_url}, {'relevant': 1, 'image_urls': 1})
    for doc in query:
        ex_list.append(doc)
    return ex_list


# ---------------------------------------------------------------------------------------------------------------------


def start_process(page_url, image_url, async=False):
    image_obj = images.find_one({"image_urls": image_url})
    if not image_obj:  # new image_url
        image_hash = page_results.get_hash_of_image_from_url(image_url)
        image_obj = images.find_one_and_update({'image_hash': image_hash}, {'$push': {"image_urls": image_url}},
                                               return_document=pymongo.ReturnDocument.AFTER)
        if not image_obj:  # doesn't exists with another url
            image = background_removal.standard_resize(Utils.get_cv2_img_array(image_url), 400)[0]
            if image is None:
                logging.warning("There's no image in the url!")
                return None
            relevance = background_removal.image_is_relevant(image)
            image_dict = {'image_urls': [image_url], 'relevant': relevance.is_relevant,
                          'image_hash': image_hash, 'page_urls': [page_url]}
            if relevance.is_relevant:
                image_dict['people'] = []
                relevant_faces = relevance.faces.tolist()
                idx = 0
                for face in relevant_faces:
                    person = {'face': face, 'person_id': str(bson.ObjectId()), 'person_idx': idx, 'items': []}
                    image_copy = person_isolation(image, face)
                    person['url'] = upload_image(image_copy, str(person['person_id']))
                    image_dict['people'].append(person)
                    q2.enqueue(get_paperdoll_data, person['url'], person['person_id'])
                    idx += 1
            else:  # if not relevant
                logging.warning('image is not relevant, but stored anyway..')
                images.insert(image_dict)
                return
            iip.insert(image_dict)
            if not async:
                while images.find_one({'image_urls': image_url}) is None:
                    time.sleep(0.5)
                return page_results.merge_items(images.find_one({'image_urls': image_url}))
        else:  # if the exact same image was found under other urls
            logging.warning("image_hash was found in other urls:")
            logging.warning("{0}".format(image_obj['image_urls']))
            return page_results.merge_items(image_obj)
    else:  # if image is in the DB
        if image_obj['relevant']:
            logging.warning("Image is in the DB and relevant!")
        else:
            logging.warning("Image is in the DB and not relevant!")
        return page_results.merge_items(image_obj)


def from_paperdoll_to_similar_results(person_id, mask, labels):
    image_obj, person = get_person_by_id(person_id, iip)
    image = Utils.get_cv2_img_array(person['url'])
    items = []
    idx = 0
    for num in np.unique(mask):
        # convert numbers to labels
        category = list(labels.keys())[list(labels.values()).index(num)]
        if category in constants.paperdoll_shopstyle_women.keys():
            item_mask = 255 * np.array(mask == num, dtype=np.uint8)
            # item_gc_mask = create_gc_mask(image, item_mask, bgnd_mask)  # (255, 0) mask
            item_dict = {"category": constants.paperdoll_shopstyle_women[category],
                         'item_id': str(bson.ObjectId()), 'item_idx': idx, 'saved_date': datetime.datetime.now()}
            svg_name = find_similar_mongo.mask2svg(
                item_mask,
                str(image_obj['_id']) + '_' + person['person_id'] + '_' + item_dict['category'],
                constants.svg_folder)
            item_dict["svg_url"] = constants.svg_url_prefix + svg_name
            item_dict['fp'], item_dict['similar_results'] = find_similar_mongo.find_top_n_results(image, item_mask,
                                                                                                  100,
                                                                                                  item_dict['category'])
            items.append(item_dict)
            idx += 1
    image_obj = iip.find_one_and_update({'people.person_id': person_id}, {'$set': {'people.$.items': items}},
                                        return_document=pymongo.ReturnDocument.AFTER)
    if person['person_idx'] == len(image_obj['people']) - 1:
        images.insert(image_obj)
        logging.warning("Done! image was successfully inserted to the DB images!")


def dole_out_work(item_id):
    """
    dole out images. this function is the engine of the sorting process.
    :param item_id:
    :return:
    """
    voting_stage = get_voting_stage(item_id)
    # make sure there's at least 1 worker per image
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




