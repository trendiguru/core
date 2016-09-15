import pymongo
import bson
import datetime
import numpy as np
from . import Utils, pipeline, constants, find_similar_mongo
from .page_results import genderize
from .constants import db, q1
from .paperdoll import pd_falcon_client, neurodoll_falcon_client

EDITOR_PROJECTION = {'image_id': 1,
                     'image_urls': 1,
                     'saved_date': 1,
                     'num_of_people': 1,
                     'people.face': 1,
                     'people.person_bb': 1,
                     'people.num_of_items': 1,
                     'people.gender': 1,
                     'people._id': 1,
                     'people.items.category': 1,
                     'people.items.similar_results': 1}


# ------------------------------------------------ IMAGE-LEVEL ---------------------------------------------------------

def get_image_obj_for_editor(image_url, image_id=None):
    query = {"image_id": image_id} if image_id else {'image_urls': image_url}
    sparse = db.images.find_one(query, EDITOR_PROJECTION)
    return sparse


def cancel_image(image_id):
    """
    Robust cancel something function.
    Returns bool of success or fail to cancel.
    :param image_id: str ID of image in db.images ('_id')
    :param person_id: str ID of person in the image obj
    :param item_category: str category of the item
    :return: success boolean
    """
    image_obj = db.images.find_one({'image_id': image_id})
    if not image_obj:
        return False
    # CANCEL IMAGE (INSERT TO IRRELEVANT_IMAGES BEFORE)
    sparse_obj = shrink_image_object(image_obj)
    # db.irrelevant_images.insert_one(sparse_obj)
    db.images.delete_one({'image_id': image_id})
    return True


def get_latest_images(num=10):
    curs = db.images.find({}, {'image_id': 1, 'image_urls': 1}).sort('_id', pymongo.DESCENDING).limit(int(num))
    return list(curs)


# ----------------------------------------------- PERSON-LEVEL ---------------------------------------------------------

def cancel_person(image_id, person_id):
    image_obj = db.images.find_one({'image_id': image_id})
    if not image_obj:
        return False
    res = db.images.update_one({'image_id': image_id}, {'$pull': {'people': {'_id': person_id}}})
    return bool(res.modified_count)


def change_gender_and_rebuild_person(image_id, person_id, new_gender):
    image_obj = db.test.find_one({'image_id': image_id})
    if not image_obj:
        return "image_obj haven't found"

    new_person = [person for person in image_obj['people'] if person['_id'] == person_id][0]
    new_person['gender'] = new_gender
    print "switching gender to {0}".format(new_gender)
    for item in new_person['items']:
        if new_gender == 'Male':
            item['category'] = constants.paperdoll_paperdoll_men[item['category']]
        else:
            item['category'] = constants.paperdoll_paperdoll_men.keys()[constants.paperdoll_paperdoll_men.values().index(item['category'])]
        for res_coll in item['similar_results'].keys():
            res_coll_gen = res_coll + '_' + new_person['gender']
            item['fp'], item['similar_results'][res_coll] = find_similar_mongo.find_top_n_results(number_of_results=100,
                                                                                                  category_id=item['category'],
                                                                                                  fingerprint=item['fp'],
                                                                                                  collection=res_coll_gen)
    res1 = db.test.update_one({'image_id': image_id}, {'$pull': {'people': {'_id': person_id}}})
    res2 = db.test.update_one({'image_id': image_id}, {'$push': {'people': new_person}})
    return bool(res1.modified_count*res2.modified_count)


def add_people_to_image(image_url, page_url, faces, products_collection='ShopStyle', method='pd'):
    for index, face in enumerate(faces):
        if not isinstance(face, list):
            faces[index] = face.tolist()
    image_obj = db.images.find_one({'image_urls': image_url})
    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        return False
    if not image_obj:
        # BUILD A NEW IMAGE WITH THE GIVEN FACES
        image_obj = {'people': [{'person_id': str(bson.ObjectId()), 'face': face,
                                 'gender': genderize(image, face)['gender']} for face in faces],
                     'image_url': image_url,
                     'page_url': page_url}
        db.iip.insert_one({'image_url': image_url, 'insert_time': datetime.datetime.utcnow()})
        db.genderator.insert_one(image_obj)
        db.irrelevant_images.delete_one({'image_urls': image_url})
        q1.enqueue_call(func="", args=(page_url, image_url, products_collection, method),
                        ttl=2000, result_ttl=2000, timeout=2000)
        return True
    else:
        # ADD PEOPLE TO AN EXISTING IMAGE
        people_to_add = [build_new_person(image, face, products_collection, method) for face in faces]
        if len(people_to_add):
            db.test.update_one({'_id': image_obj['_id']}, {'$push': {'image_urls': {'$each': people_to_add}},
                                                           '$set': {'num_of_people': len(people_to_add)}})
            return True
        else:
            return False


# ------------------------------------------------ ITEM-LEVEL ----------------------------------------------------------

def cancel_item(image_id, person_id, item_category):
    image_obj = db.images.find_one({'image_id': image_id})
    if not image_obj:
        return False
    for person in image_obj['people']:
        if person['_id'] == person_id:
            for item in person['items']:
                if item['category'] == item_category:
                    person['items'].remove(item)
    res = db.images.replace_one({'image_id': image_id}, image_obj)
    return bool(res.modified_count)


def reorder_results(image_id, person_id, item_category, collection, new_results):
    print "got into the function!"
    image_obj = db.images.find_one({'image_id': image_id})
    if not image_obj:
        return False
    for person in image_obj['people']:
        if person['_id'] == person_id:
            print "Found person"
            for item in person['items']:
                if item['category'] == item_category:
                    print "Found item, gonna switch results with {0}".format(new_results)
                    item['similar_results'][collection] = new_results
    res = db.images.replace_one({'image_id': image_id}, image_obj)
    return res.modified_count


# ----------------------------------------------- RESULT-LEVEL ---------------------------------------------------------

def cancel_result(image_id, person_id, item_category, results_collection, result_id):
    image_obj = db.images.find_one({'image_id': image_id})
    if not image_obj:
        return False
    ret = False
    for person in image_obj['people']:
        if person['_id'] == person_id:
            ret = 'found person'
            for item in person['items']:
                if item['category'] == item_category:
                    ret = 'found_item'
                    for result in item['similar_results'][results_collection]:
                        if result['id'] == int(result_id):
                            item['similar_results'][results_collection].remove(result)
                            ret = True
    res = db.images.replace_one({'image_id': image_id}, image_obj)
    return ret


# ----------------------------------------------- CO-FUNCTIONS ---------------------------------------------------------

def shrink_image_object(image_obj):
    image_obj['relevant'] = False
    image_obj['people'] = []
    image_obj.pop('num_of_people')
    image_obj.pop('image_id')
    return image_obj


def build_new_person(image, face, products_collection, method):
        x, y, w, h = face
        person_bb = [int(round(max(0, x - 1.5 * w))), str(y), int(round(min(image.shape[1], x + 2.5 * w))),
                     min(image.shape[0], 8 * h)]
        person = {'face': face, 'person_bb': person_bb, 'gender': genderize(image, face)['gender'],
                  'products_collection': products_collection, 'segmentation_method': method, 'items': [],
                  '_id': str(bson.ObjectId())}
        try:
            if person['segmentation_method'] == 'pd':
                seg_res = pd_falcon_client.pd(image)
            else:
                seg_res = neurodoll_falcon_client.pd(image)
        except Exception as e:
            print e
            return
        if 'success' in seg_res and seg_res['success']:
            mask = seg_res['mask']
            if person['segmentation_method'] == 'pd':
                labels = seg_res['label_dict']
            else:
                labels = constants.ultimate_21_dict
        else:
            return
        final_mask = pipeline.after_pd_conclusions(mask, labels, person['face'])
        for num in np.unique(final_mask):
            pd_category = list(labels.keys())[list(labels.values()).index(num)]
            if pd_category in constants.paperdoll_relevant_categories:
                item_mask = 255 * np.array(final_mask == num, dtype=np.uint8)
                if person['gender'] == 'Male':
                    category = constants.paperdoll_paperdoll_men[pd_category]
                else:
                    category = pd_category
            person['items'].append(bulid_new_item(category, item_mask, products_collection, image, person['gender']))
        return person


def bulid_new_item(category, item_mask, collection, image, gender):
    prod = collection + '_' + gender
    item = {'similar_results': {}, 'category': category}
    item['fp'], item['similar_results'][collection] = find_similar_mongo.find_top_n_results(image, item_mask, 100,
                                                                                            category, prod)
    return item
