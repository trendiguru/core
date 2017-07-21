from __future__ import absolute_import, print_function, unicode_literals
import traceback
import tldextract
import datetime
import bson
import time
import rq
import numpy as np
from .. import find_similar_mongo
from .. import constants
from ..constants import db, redis_conn
from .. import whitelist, page_results, Utils, background_removal
from .. import new_image_notifier
from ..paperdoll import neurodoll_falcon_client as nd_client


def process_image(image, image_dict, products_collection, method='nd'): # page_url, image_url, products_collection, method):
    # page_url = _image_dict['page_urls'][0]
    # try:
    #     domain = tldextract.extract(page_url).registered_domain
    # except:
    #    domain = 'not_found'

    domain = "dummy_domain"


    image_hash = page_results.get_hash(image)

    image_dict.update({
        'relevant': True,
        'views': 1,
        'saved_date': str(datetime.datetime.utcnow()),
        'image_hash': image_hash,
        'image_id': str(bson.ObjectId()),
        'domain': domain
    })


    idx = 0
    people_to_emit = []
    for person in image_dict['people']:
        face = person['face']
        isolated_image = background_removal.person_isolation(image, face)

        person_bb = Utils.get_person_bb_from_face(face, image.shape)
        person_args = {'face': face, 'person_bb': person_bb, 'image_id': image_dict['image_id'],
                       'image': isolated_image.tolist(), 'gender': person.get('gender'), 'domain': domain,
                       'products_collection': products_collection, 'segmentation_method': method}
        people_to_emit.append(person_args)
        idx += 1

    image_dict['num_of_people'] = idx
    image_dict['people'] = people_to_emit

    return image_dict


def process_person(image_url, person_dict):
    # image_id = person['image_id']
    person_image = np.array((person_dict['image']), dtype=np.uint8)
    person_dict['_id'] = str(bson.ObjectId())
    person_dict['items'] = []
    # neurodoll_falcon_client.nd(url, get_combined_results=True)
    seg_res = nd_client.nd(image_url, get_combined_results=True)
    # TODO: this is what it should be (when nd can handle it)
    #seg_res = nd_client.nd(person_image, get_combined_results=True)
    if seg_res.get('success'):
        final_mask = seg_res['mask']
        labels = seg_res['label_dict']
    else:
        raise SystemError("segmentation failed with:\n{}".format(traceback.format_exc()))
    # final_mask = pipeline.after_nn_conclusions(mask, labels, person['face'])

    idx = 0
    items = []
    for num in np.unique(final_mask):
        pd_category = list(labels.keys())[list(labels.values()).index(num)]
        if pd_category in constants.paperdoll_relevant_categories:
            item_mask = 255 * np.array(final_mask == num, dtype=np.uint8)
            if person_dict['gender'] == 'Male':
                category = constants.paperdoll_paperdoll_men[pd_category]
            else:
                category = pd_category
            item_args = {'mask': item_mask.tolist(), 'category': category, 'image': person_image.tolist(),
                         'domain': person_dict['domain'], 'gender': person_dict['gender'],
                         'products_collection': person_dict['products_collection']}
            items.append(item_args)
            idx += 1
    person_dict['num_of_items'] = idx
    person_dict['items'] = items
    person_dict.pop('domain') # WHY??


    return person_dict


def process_item(person_id, item):
    item_mask = np.array(item['mask'], dtype=np.uint8)
    person_image = np.array(item['image'], dtype=np.uint8)
    if 'gender' in item.keys():
        gender = item['gender'] or 'Female'
    else:
        gender = "Female"
    out_item = {'similar_results': {}}
    start = time.time()
    coll = item['products_collection']
    prod = coll + '_' + gender
    out_item['fp'], out_item['similar_results'][coll] = find_similar_mongo.find_top_n_results(person_image,
                                                                                              item_mask, 50,
                                                                                              item['category'],
                                                                                              prod)
    for feature in out_item['fp'].keys():
        if isinstance(out_item['fp'][feature], np.ndarray):
            out_item['fp'][feature] = out_item['fp'][feature].tolist()
    print("find_top_n took {0} secs, {1} , {2}: ".format(time.time() - start, prod, item['category']))

    item.update(out_item)

    return item


