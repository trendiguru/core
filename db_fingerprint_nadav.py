__author__ = 'Nadav Paz'

import logging
import multiprocessing
import argparse
import pdb

import pymongo
import numpy as np
import cv2

import fingerprint_core as fp
import background_removal
import Utils

import constants


CLASSIFIER_FOR_CATEGORY = {}
TOTAL_PRODUCTS = 0
CURRENT = Utils.ThreadSafeCounter()
DB = None
FP_VERSION = 0


def get_all_subcategories(category_collection, category_id):
    subcategories = []

    def get_subcategories(c_id):
        subcategories.append(c_id)
        curr_cat = category_collection.find_one({"id": c_id})
        if "childrenIds" in curr_cat.keys():
            for childId in curr_cat["childrenIds"]:
                get_subcategories(childId)

    get_subcategories(category_id)
    return subcategories


def create_classifier_for_category_dict(db):
    """
    Creates a dictionary of category_id: CascasdeClassifier
    :param db:
    :return:
    """
    result_dict = {}
    classifier_dict = {xml: cv2.CascadeClassifier(constants.classifiers_folder + xml)
                       for xml in constants.classifier_to_category_dict.keys()}
    for xml, cats in constants.classifier_to_category_dict.iteritems():
        for cat in cats:
            for sub_cat in get_all_subcategories(db.categories, cat):
                result_dict[sub_cat] = classifier_dict[xml]
    return result_dict


def run_fp(doc):
    # pdb.set_trace()
    # CURRENT.increment()
    # print "Starting {i} of {total}...".format(i=CURRENT.value, total=TOTAL_PRODUCTS)
    image_url = doc["image"]["sizes"]["XLarge"]["url"]
    image = Utils.get_cv2_img_array(image_url)
    small_image, resize_ratio = background_removal.standard_resize(image, 400)
    print "Image URL: {0}".format(image_url)
    # if there is a valid human BB, use it
    if "human_bb" in doc.keys() and doc["human_bb"] != [0, 0, 0, 0] and doc["human_bb"] is not None:
        chosen_bounding_box = doc["human_bb"]
        chosen_bounding_box = [int(b) for b in (np.array(chosen_bounding_box) / resize_ratio)]
        mask = background_removal.get_fg_mask(small_image, chosen_bounding_box)
        logging.debug("Human bb found: {bb} for item: {id}".format(bb=chosen_bounding_box, id=doc["id"]))
    # otherwise use the largest of possibly many classifier bb's
    else:
        if "categories" in doc:
            classifier = CLASSIFIER_FOR_CATEGORY.get(doc["categories"][0]["id"], "")
        else:
            classifier = None

        # first try grabcut with no bb
        mask = background_removal.get_fg_mask(small_image)
        bounding_box_list = []

        if classifier and not classifier.empty():
            # then - try to classify the image (white backgrounded and get a more accurate bb
            white_bckgnd_image = background_removal.image_white_bckgnd(small_image, mask)
            try:
                bounding_box_list = classifier.detectMultiScale(white_bckgnd_image)
            except KeyError:
                logging.warning("Could not classify with {0}".format(classifier))
        max_bb_area = 0
        chosen_bounding_box = None
        # choosing the biggest bounding box
        for possible_bb in bounding_box_list:
            if possible_bb[2] * possible_bb[3] > max_bb_area:
                chosen_bounding_box = possible_bb
                max_bb_area = possible_bb[2] * possible_bb[3]
        if chosen_bounding_box is None:
            logging.warning("No Bounding Box found, using the whole image. "
                            "Document id: {0}, BB_list: {1}".format(doc.get("id"), str(bounding_box_list)))
        else:
            mask = background_removal.get_fg_mask(small_image, chosen_bounding_box)
    try:
        fingerprint = fp.fp(small_image, mask)
        DB.products.update({"id": doc["id"]},
                           {"$set": {"fingerprint": fingerprint.tolist(),
                                     "fp_version": FP_VERSION,
                                     "bounding_box": np.array(chosen_bounding_box).tolist()}
                           })

    except Exception as e:
        logging.warning("Exception caught while fingerprinting, skipping: {0}".format(e))


def fingerprint_db(fp_version, category_id=None, num_processes=None):
    """
    main function - fingerprints items in category_id and its subcategories.
     If category_id is None, then fingerprints entire db.
    :param fp_version: integer to keep track of which items have been already fingerprinted with this version
    :param category_id: category to be fingerprinted
    :return:
    """
    global DB, TOTAL_PRODUCTS, CURRENT, CLASSIFIER_FOR_CATEGORY

    DB = DB or pymongo.MongoClient().mydb

    if category_id is not None:
        query_doc = {"$and": [
            {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(DB.categories, category_id)}}}},
            {"$or": [{"fp_version": {"$lt": fp_version}}, {"fp_version": {"$exists": 0}}]}
        ]}
    else:
        query_doc = {"$or": [{"fp_version": {"$lt": fp_version}}, {"fp_version": {"$exists": 0}}]}

    fields = {"image": 1, "human_bb": 1, "fp_version": 1, "bounding_box": 1, "categories": 1}

    # batch_size required because cursor timed out without it. Could use further investigation
    product_cursor = DB.products.find(query_doc, fields).batch_size(100)
    TOTAL_PRODUCTS = product_cursor.count()
    CLASSIFIER_FOR_CATEGORY = create_classifier_for_category_dict(DB)

    num_processes = num_processes or multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(num_processes)

    pdb.set_trace()
    for doc in product_cursor[0:10]:
        run_fp(doc)
    # pool.map(run_fp, product_cursor[0:10])
    # pool.close()
    # pool.join()

    print "All done!!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fingerprint the DB or part of it')
    parser.add_argument('-c', '--category_id', help='id of category to be fingerprinted', required=False)
    parser.add_argument('-p', '--num_processes', help='number of parallel processes to spawn', required=False)
    parser.add_argument('-v', '--fp_version', help='current fp version', required=True)
    args = vars(parser.parse_args())

    fingerprint_db(int(args['fp_version']), args['category_id'], int(args['num_processes']))




