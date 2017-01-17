import traceback
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
    query = {'image_id': image_id} if image_id else {'image_urls': image_url}
    sparse = db.images.find_one(query, EDITOR_PROJECTION)
    # TODO - what happen if the image is in db.irrelevant
    # if not sparse:

    # for person in sparse['people']:
    #     for item in person['items']:
    #         for prod_coll in item['similar_results'].keys():
    #             for result in item['similar_results'][prod_coll]:
    #                 product = db[prod_coll+'_'+person['gender']].find_one({'id': result['id']})
    #                 result['price'] = product['price']
    #                 result['brand'] = product['brand']
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
    db.irrelevant_images.insert_one(sparse_obj)
    db.images.delete_one({'image_id': image_id})
    return True


def get_latest_images(num=10, user_filter=None):
    # user filter - string that contained in page_url
    if user_filter:
        curs = db.images.find({'page_urls': {'$regex': user_filter}}, {'_id': 0, 'image_id': 1, 'image_urls': 1}).sort('_id', pymongo.DESCENDING).limit(int(num))
    else:
        curs = db.images.find({}, {'_id': 0, 'image_id': 1, 'image_urls': 1}).sort('_id', pymongo.DESCENDING).limit(int(num))
    return list(curs)


# ----------------------------------------------- PERSON-LEVEL ---------------------------------------------------------

def cancel_person(image_id, person_id):
    image_obj = db.images.find_one({'image_id': image_id})
    if not image_obj:
        return False
    # res = db.images.update_one({'image_id': image_id}, {'$pull': {'people': {'_id': person_id}}})
    for person in image_obj['people']:
        if person['_id'] == person_id:
            image_obj['people'].remove(person)
            if not len(image_obj['people']):
                cancel_image(image_id)
                res = 1
            else:
                res = db.images.replace_one({'image_id': image_id}, image_obj).modified_count
    return bool(res)


def change_gender_and_rebuild_person(image_id, person_id, new_gender):
    image_obj = db.images.find_one({'image_id': image_id})
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
    res1 = db.images.update_one({'image_id': image_id}, {'$pull': {'people': {'_id': person_id}}})
    res2 = db.images.update_one({'image_id': image_id}, {'$push': {'people': new_person}})
    return bool(res1.modified_count*res2.modified_count)


def add_people_to_image(image_id, faces, products_collection='amazon_US', method='pd'):
    image_obj = db.images.find_one({'image_id': image_id})
    # MAKE SURE ALL FACES ARE LISTS
    for index, face in enumerate(faces):
        if not isinstance(face, list):
            faces[index] = face.tolist()
    # DOWNLOAD IMAGE AND VERIFY
    # page_url = image_obj['page_urls'][0]
    image_url = image_obj['image_urls'][0]
    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        return False
    # if not image_obj:
    #     # BUILD A NEW IMAGE WITH THE GIVEN FACES
    #     image_obj = {'people': [{'person_id': str(bson.ObjectId()), 'face': face.tolist(),
    #                              'gender': genderize(image, face.tolist())['gender']} for face in faces],
    #                  'image_urls': image_url, 'page_url': page_url, 'insert_time': datetime.datetime.now()}
    #     db.iip.insert_one(image_obj)
    #     q1.enqueue_call(func="", args=(page_url, image_url, products_collection, method),
    #                     ttl=2000, result_ttl=2000, timeout=2000)
    #     db.irrelevant_images.delete_one({'image_urls': image_url})
    #     return True
    # else:
    # ADD PEOPLE TO AN EXISTING IMAGE
    people_to_add = [build_new_person(image, face, products_collection, method) for face in faces]
    if len(people_to_add):
        db.images.update_one({'_id': image_obj['_id']}, {'$push': {'image_urls': {'$each': people_to_add}},
                                                         '$set': {'num_of_people': len(people_to_add)}})
        return True
    else:
        return False


# ------------------------------------------------ ITEM-LEVEL ----------------------------------------------------------

def add_item(image_id, person_id, category, collection):
    # GET IMAGE
    image_obj = db.images.find_one({'image_id': image_id})
    image = Utils.get_cv2_img_array(image_obj['image_urls'][0])
    if image is None:
        return False
    # NEURODOLL WITH CATEGORY
    labels = constants.ultimate_21_dict
    seg_res = neurodoll_falcon_client.pd(image, labels[category])
    if not seg_res['success']:
        return False
    item_mask = 255 * np.array(seg_res['mask'] > np.median(seg_res['mask']), dtype=np.uint8)
    person = [pers for pers in image_obj['people'] if pers['_id'] == person_id][0]
    # BUILD ITEM WITH MASK {fp, similar_results, category}
    # Assume this person has at least one item...
    collections = person["items"][0]["similar_results"].keys()
    new_item = build_new_item(category, item_mask, collections, image, person['gender'])
    # UPDATE THE DB DOCUMENT
    res = db.images.update_one({'people._id': person_id}, {'$push': {'people.$.items': new_item}})
    return bool(res.modified_count)


def cancel_item(image_id, person_id, item_category):
    image_obj = db.images.find_one({'image_id': image_id})
    if not image_obj:
        return False
    for person in image_obj['people']:
        if person['_id'] == person_id:
            for item in person['items']:
                if item['category'] == item_category:
                    person['items'].remove(item)
                    if not len(person['items']):
                        cancel_person(image_id, person_id)
                        res = 1
                    else:
                        res = db.images.replace_one({'image_id': image_id}, image_obj).modified_count
    return bool(res)


def reorder_results(image_id, person_id, item_category, collection, new_results):
    image_obj = db.images.find_one({'image_id': image_id})
    if not image_obj:
        return False
    if not isinstance(new_results, list) or not len(new_results):
        return False
    for person in image_obj['people']:
        if person['_id'] == person_id:
            for item in person['items']:
                if item['category'] == item_category:
                    item['similar_results'][collection] = new_results
    res = db.images.replace_one({'image_id': image_id}, image_obj)
    return res.modified_count


# ----------------------------------------------- RESULT-LEVEL ---------------------------------------------------------

def add_result(image_id, person_id, item_category, results_collection, new_result):
    print "Will add result..."
    image_obj = db.images.find_one({'image_id': image_id})
    if not image_obj:
        return False
    try:
        person = [pers for pers in image_obj['people'] if pers['_id'] == person_id][0]
        item = [item for item in person['items'] if item['category'] == item_category][0]
        results = item['similar_results'][results_collection]
        # new_result['id'] = db[results_collection+'_'+person['gender']].find_one({'images.XLarge': new_result['images']['XLarge']})['id']
        new_result['id'] = bson.ObjectId()
        results.insert(0, new_result)
        db.images.replace_one({'image_id': image_id}, image_obj)
        return True
    except Exception as e:
        print traceback.format_exc()
        return False


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
                        if result['id'] == result_id:
                            item['similar_results'][results_collection].remove(result)
                            ret = True
    db.images.replace_one({'image_id': image_id}, image_obj)
    return ret


# ----------------------------------------------- CO-FUNCTIONS ---------------------------------------------------------

def shrink_image_object(image_obj):
    image_obj['relevant'] = False
    image_obj['people'] = []
    image_obj.pop('num_of_people')
    image_obj.pop('image_id')
    return image_obj


def build_new_person(image, face, products_collection, method):
        # INITIALIZE PERSON STRUCTURE
        x, y, w, h = face
        person_bb = [int(round(max(0, x - 1.5 * w))), str(y), int(round(min(image.shape[1], x + 2.5 * w))),
                     min(image.shape[0], 8 * h)]
        person = {'face': face, 'person_bb': person_bb, 'gender': genderize(image, face)['gender'],
                  'products_collection': products_collection, 'segmentation_method': method, 'items': [],
                  '_id': str(bson.ObjectId())}
        # SEGMENTATION
        try:
            if person['segmentation_method'] == 'pd':
                seg_res = pd_falcon_client.pd(image)
            else:
                seg_res = neurodoll_falcon_client.pd(image)
        except Exception as e:
            print e
            return
        # CATEGORIES CONCLUSIONS
        if 'success' in seg_res and seg_res['success']:
            mask = seg_res['mask']
            if person['segmentation_method'] == 'pd':
                labels = seg_res['label_dict']
                final_mask = pipeline.after_pd_conclusions(mask, labels, person['face'])
            else:
                labels = constants.ultimate_21_dict
                final_mask = pipeline.after_nn_conclusions(mask, labels, person['face'])
        else:
            return
        # BUILD ITEMS IN THE PERSON
        for num in np.unique(final_mask):
            pd_category = list(labels.keys())[list(labels.values()).index(num)]
            if pd_category in constants.paperdoll_relevant_categories:
                item_mask = 255 * np.array(final_mask == num, dtype=np.uint8)
                if person['gender'] == 'Male':
                    category = constants.paperdoll_paperdoll_men[pd_category]
                else:
                    category = pd_category
                person['items'].append(build_new_item(category, item_mask, products_collection, image, person['gender']))
        return person


def build_new_item(category, item_mask, collections, image, gender):
    if isinstance(collections, basestring):
        collections = [collections]
      
    item = {'similar_results': {}, 'category': category}
    fp = None
    for collection in collections:
        prod = collection + '_' + gender
        # fp is initially none, so find_top_n calculates it, then next time when it has a value, it gets used.
        fp, results = find_similar_mongo.find_top_n_results(image, item_mask, 100, category, prod, fingerprint=fp)
        item["similar_results"][collections] = results
        item["fp"] = fp
    
    return item
  
#-------------------------- TESTS ------------------------
                                                           
def test_add_item():
    from trendi import edit_results
    from trendi.constants import db
    from trendi import page_results
    import re
    
    im = "586b7ae5603c0b2d4c6ab3d6"
    per = "586b7b47603c0b2d3e2af6e8"
    cat = "dress"
    pid = db.users.find_one({'email': re.compile(".*stylebook.*")})['pid']
    coll = page_results.get_collection_from_ip_and_pid(None, pid)
    add_item(im, per, cat, coll)
