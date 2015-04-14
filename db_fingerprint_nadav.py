__author__ = 'Nadav Paz'

import logging

import pymongo
import numpy as np

import fingerprint_core as fp
import classify_core as classify
import background_removal
import Utils
import constants


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


def db_fp(fp_version, category_name=None):
    """
    example: python fingerprint_db_params_mongo.py mens-shirts shirtClassifier.xml true
    to run on entire database and fingerprint anything not that doesnt already have a fingerprint, do:
    python fingerprint_db_params_mongo.py undone
    """
    db = pymongo.MongoClient().mydb
    if category_name is not None:
        query_doc = {"$and": [
            {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_name)}}}},
            {"fp_version": {"$lt": fp_version}}]}
    else:
        query_doc = {"fp_version": {"$lt": fp_version}}
    fields = {"image": 1, "human_bb": 1, "fp_version": 1, "bounding_box": 1}
    query = db.products.find(query_doc, fields).batch_size(
        100)  # batch_size required because cursor timed out without it
    total_items = query.count()

    i = 1
    # boxes_found = 0

    for doc in query:
        print "Starting {i} of {total}...".format(i=i, total=total_items)
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
            # search the classifier_xml that fits that category
            for key, value in constants.classifier_to_category_dict.iteritems():
                value_subcategories = set(get_all_subcategories(db.categories, value))
                if not value_subcategories.isdisjoint(set(doc["categories"])):
                    classifier_xml = constants.classifiers_folder + key
                    break
            # first try grabcut with no bb
            mask = background_removal.get_fg_mask(small_image)
            # then - try to classify the image (white backgrounded and get a more accurate bb
            white_bckgnd_image = background_removal.image_white_bckgnd(small_image, mask)
            try:
                bounding_box_list = classify.classify_image_with_classifiers(white_bckgnd_image, classifier_xml)[classifier_xml]
            except KeyError:
                logging.warning("Could not classify with {0}".format(classifier_xml))
                bounding_box_list = []
            max_bb_area = 0
            chosen_bounding_box = None
            # choosing the biggest bounding box
            for possible_bb in bounding_box_list:
                if possible_bb[2] * possible_bb[3] > max_bb_area:
                    chosen_bounding_box = possible_bb
                    max_bb_area = possible_bb[2] * possible_bb[3]
            if chosen_bounding_box is None:
                logging.warning("No Bounding Box found, using the whole image. "
                                "Document id: {0}, BB_list: {1}".format(doc["id"], str(bounding_box_list)))
            else:
                mask = background_removal.get_fg_mask(small_image, chosen_bounding_box)
        try:
            fingerprint = fp.fp(small_image, mask)
            db.products.update({"id": doc["id"]},
                               {"$set": {"fingerprint": fingerprint.tolist(),
                                         "fp_version": fp_version,
                                         "bounding_box": np.array(chosen_bounding_box).tolist()}
                               })
            i += 1

        except Exception as e:
            logging.warning("Exception caught while fingerprinting, skipping: {0}".format(e))


