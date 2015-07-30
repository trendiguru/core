__author__ = 'Nadav Paz'

# theirs
import logging
import requests
import copy

import pymongo
import cv2
import redis
from rq import Queue
import bson
import boto3

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
        try:
            for item in person['items']:
                if item['item_id'] == item_id:
                    return image, {'person': person, 'person_idx': image['people'].index(person)}, \
                           {'item': item, 'item_idx': person['items'].index(item)}
        except:
            logging.warning("No items to this person, continuing..")
            return None, None



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
    avg_bb = Utils.average_bbs(good_bblist)
    # print('avg bb:'+str(avg_bb))
    # check if any of the boxes are way out
    good_intersection_bblist = []
    for bb in good_bblist:
        if Utils.intersectionOverUnion(bb, avg_bb) >= constants.bb_iou_threshold:
            good_intersection_bblist.append(bb)
    if good_intersection_bblist != []:  # got at least one good bb
        improved_result = Utils.average_bbs(good_intersection_bblist)
        return improved_result
    else:
        logging.warning('no good intersections found')
        return None
#    bb = determine_final_bb(bb_list)  # Yonti's function
    # image = images.find_one({'people.items.item_id': item_id}, {})
    #    category = get_item_by_id(item_id)['category']
    #    fp, results, svg = find_similar_mongo.got_bb(image['image_url'], person_id, item_id, bb, N_top_results_to_show,
    #category)


# q6 - send_20s_(actually N) results
# FUNCTION 6
def dole_out_work(item_id):
    '''
    dole out images. Im assuming that i should dole out all the similar items instead of
    doling out constants.N_top_results_to_show
    :param item_id:
    :return:
    '''
    voting_stage = get_voting_stage(item_id)
    # make sure theres at least 1 worker per image
    image, person_dict, item_dict = get_item_by_id(item_id)
    person_idx = person_dict['person_idx']
    item_idx = item_dict['item_idx']
    item = image['people'][person_idx]['items'][item_idx]
    similar_items = item['similar_items']

    assert (constants.N_workers[voting_stage] * constants.N_pics_per_worker[voting_stage] /
            len(similar_items) > 1)  #len similar_items instead of constants.N_top_results_to_show

    image, person_dict, item_dict = get_item_by_id(item_id)
    person_idx = person_dict['person_idx']
    item_idx = item_dict['item_idx']
    similar_items = image['people'][person_idx]['items'][item_idx]['similar_items']
    if similar_items is None:
        logging.warning('oh man no similar items found')
        return None
    for i in range(0, constants.N_workers[voting_stage]):  #divide results into chunks for N workers
        first_image_index = i * constants.N_pics_per_worker
        last_image_index = (i + 1) * constants.N_pics_per_worker
        # the min below deals with case where there's fewer images for last worker
        last_image_index = min(last_image_index, len(similar_items))
        chunk_of_similar_items = similar_items[first_image_index:last_image_index]
        ######CHECK WITH NADAV THAT THIS QUEUEUE IS RIGHT
        q4.enqueue(send_similar_items_to_qc, item_id, chunk_of_similar_items)


# QUEUE FUNC FOR FUNCTION 6
def send_similar_items_to_qc(item_id, chunk_of_similar_items):
    payload = {'item_id': item_id, 'results_to_sort': chunk_of_similar_items}
    req = requests.post(QC_URL, data=payload)
    return req.ok
# END OF QUEUE FUNC FOR FUNCTION 6
# END  FUNCTION 6

def set_voting_stage(N_stage, item_id):
    '''
    this can be replaced by a different persistent storage scheme than storing
    in the image db
    :param N_stage:
    :param item_id:
    :return:
    '''
    image, person_dict, item_dict = get_item_by_id(item_id)
    person_idx = person_dict['person_idx']
    item_idx = item_dict['item_idx']
    image['people'][person_idx]['items'][item_idx]['voting_stage'] = N_stage
    write_result = images.update({"people.items.item_id": item_id}, image)

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
        # image, {'person': person, 'person_idx': image['people'].index(person)}, \
        # {'item': item, 'item_idx': person['items'].index(item)}


###Here i am assuming I get votes in the form of a list of numbers or 'not relevant',
### the same length as the similar_items
def from_qc_get_votes(item_id, chunk_of_similar_items, chunk_of_votes):
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

    logging.debug('tot votes:' + str(tot_votes))
    logging.debug('comb.sim.items:' + str(combined_similar_items))
    logging.debug('comb.votes:' + str(combined_votes))
    # enough votes done already to take results and move to next stage?
    voting_stage = get_voting_stage(item_id)
    print('votingStage:' + str(voting_stage))
    logging.debug('votingStage:' + str(voting_stage))
    enough_votes = constants.N_pics_per_worker[voting_stage] * constants.N_workers[voting_stage]
    if tot_votes >= enough_votes:
        combined_similar_items, combined_votes = order_results(combined_similar_items, combined_votes)
        set_voting_stage(voting_stage + 1, item_id)

    item['votes'] = combined_votes
    item['similar_items'] = combined_similar_items
    image['people'][person_idx]['items'][item_idx]['votes'] = item[
        'votes']  # maybe unecessary since item['votes'] prob writes into image
    image['people'][person_idx]['items'][item_idx]['similar_items'] = item[
        'similar_items']  # maybe unecessary since item['votes'] prob writes into image
    write_result = images.update({"people.items.item_id": item_id}, image)
    print('image written:' + str(image))

# if persistent_voting_stage == final_stage:
# return top_N (or do whatever else needs to be done when voting is over)
#        otherwise:
#           dole_out_work(top_N,voting_stage=persistent_voting_stage)

def add_results(extant_similar_items, extant_votes, new_similar_items, new_votes):
    '''
    add in new votes to current votes, making sure to check if new votes are on things
    already voted for, if so tack onto end of vote list
    :param extant_similar_items: list of itemes like [itemA,itemB]
    :param extant_votes:  list of vote lists like [[voteA1,voteA2],[voteB1,voteB2]]
    :param new_similar_items:  like previous list
    :param new_votes: flat list of votes like [voteA3,voteC3]
    :return: new extant_votes list and similar_items list
    '''
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
        if match_flag == False:
            # got a vote on a new item
            modified_extant_similar_items.append(new_similar_items[i])
            modified_extant_votes.append([new_votes[i]])

    assert (len(modified_extant_similar_items) == len(modified_extant_votes))
    tot_votes = 0
    for j in range(0, len(extant_votes)):
        tot_votes = tot_votes + len(extant_votes[j])

    return tot_votes, modified_extant_similar_items, modified_extant_votes


def order_results(combined_similar_items, combined_votes):
    '''
    smush multiple votes into a single combined vote
    :param combined_similar_items:
    :param combined_votes:
    :return:
    '''
    for j in range(0, len(combined_similar_items)):
        combined_votes[j] = combine_votes(combined_votes[j])
    sorted_votes = sorted(combined_votes)
    # the following works but is dangerous since it sorts by tuples of [vote,item]
    sorted_items = [item for (vote, item) in sorted(zip(combined_votes, combined_similar_items))]
    return sorted_items, sorted_votes

def combine_votes(combined_votes):
    '''
    take multiple votes and smush into one
    :return:
    '''
    # if all are numbers:
    #        return average
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
            n = n + 1
            sum = sum + vote
            int_flag = True
        else:
            n = n + 1
            sum = sum + not_relevant_score
            non_int_flag = True
    if not int_flag:  # all non-int/float
        return 'not relevant'
    if not non_int_flag:  # all int/float
        return float(sum) / n
    return float(
        sum) / n  #some int/float and some not-relevant - above 3 lines can be combined, but ths is clearer to me

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
