from . import constants
from . import find_similar_mongo
from constants import db


# ------------------------------------------------ IMAGE-LEVEL ---------------------------------------------------------

def cancel_image(image_id):
    """
    Robust cancel something function.
    Returns bool of success or fail to cancel.
    :param image_id: str ID of image in db.images ('_id')
    :param person_id: str ID of person in the image obj
    :param item_category: str category of the item
    :return: success boolean
    """
    image_obj = db.test.find_one({'image_id': image_id})
    if not image_obj:
        return False
    # CANCEL IMAGE (INSERT TO IRRELEVANT_IMAGES BEFORE)
    sparse_obj = shrink_image_object(image_obj)
    # db.irrelevant_images.insert_one(sparse_obj)
    db.test.delete_one({'image_id': image_id})
    return True


# ----------------------------------------------- PERSON-LEVEL ---------------------------------------------------------

def cancel_person(image_id, person_id):
    image_obj = db.test.find_one({'_id': image_id})
    if not image_obj:
        return False
    res = db.test.update_one({'_id': image_id}, {'$pull': {'people': {'_id': person_id}}})
    return bool(res.modified_count)


def change_gender_and_rebuild_person(image_id, person_id):
    image_obj = db.images.find_one({'_id': image_id})
    if not image_obj:
        return False

    new_person = [person for person in image_obj['people'] if person['_id'] == person_id][0]
    new_person['gender'] = 'Female'*bool(new_person['gender'] == 'Male') or 'Male'
    for item in new_person['items']:
        if new_person['gender'] == 'Male':
            item['category'] = constants.paperdoll_paperdoll_men[item['category']]
        else:
            item['category'] = constants.paperdoll_paperdoll_men.keys()[constants.paperdoll_paperdoll_men.values().index(item['category'])]
        for res_coll in item['similar_results'].keys():
            res_coll_gen = res_coll + '_' + new_person['gender']
            item['fp'], item['similar_results'][res_coll] = find_similar_mongo.find_top_n_results(number_of_results=100,
                                                                                                  category_id=item['category'],
                                                                                                  fingerprint=item['fp'],
                                                                                                  collection=res_coll_gen)
    res1 = db.images.update_one({'_id': image_id}, {'$pull': {'people': {'_id': person_id}}})
    res2 = db.images.update_one({'_id': image_id}, {'$push': {'people': new_person}})
    return bool(res1.modified_count*res2.modified_count)


# ------------------------------------------------ ITEM-LEVEL ----------------------------------------------------------

def cancel_item(image_id, person_id, item_category):
    image_obj = db.images.find_one({'_id': image_id})
    if not image_obj:
        return False
    for person in image_obj['people']:
        if person['_id'] == person_id:
            for item in person['items']:
                if item['category'] == item_category:
                    person.remove(item)
    res = db.images.replace_one({'_id': image_id}, image_obj)
    return bool(res.modified_count)


def reorder_results(image_id, person_id, item_category, ordered_results, results_collection):
    image_obj = db.images.find_one({'_id': image_id})
    if not image_obj:
        return False
    for person in image_obj['people']:
        if person['_id'] == person_id:
            for item in person['items']:
                if item['category'] == item_category:
                    item['similar_results'][results_collection] = ordered_results
    res = db.images.replace_one({'_id': image_id}, image_obj)
    return bool(res.modified_count)


# ----------------------------------------------- CO-FUNCTIONS ---------------------------------------------------------

def shrink_image_object(image_obj):
    image_obj['relevant'] = False
    image_obj['people'] = []
    image_obj.pop('num_of_people')
    image_obj.pop('image_id')
    return image_obj
