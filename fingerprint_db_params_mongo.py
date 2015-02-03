__author__ = 'liorsabag'
import sys
from Utils import get_cv2_img_array
import fingerprint_core as fp
import classify_core as classify
from bson import datetime
import logging
import pymongo
import numpy as np


def get_all_subcategories(category_collection, category_id):
    subcategories = []

    def get_subcategories(c_id):
        subcategories.append(c_id)
        curr_cat = category_collection.find_one({"id": c_id})
        if "childrenIds" in curr_cat.keys():
            for childId in curr_cat["childrenIds"]:
                get_subcategories(childId)

    get_subcategories(category_id)
    print(subcategories)
    return subcategories


def main():
    # logging.basicConfig(filename='fingerprint_db.log', level=logging.DEBUG)
    first_run = False

    #example: python fingerprint_db_params_mongo.py mens-shirts shirtClassifier.xml true
    if len(sys.argv) >= 3:
        category_name = sys.argv[1]
        classifier_xml = sys.argv[2]
        if len(sys.argv) == 4 and sys.argv[3].lower() == "true":
            first_run = True
    else:
        print("Missing parameters. Example use: python fingerprint_db_params_mongo.py mens-shirts shirtClassifier.xml true")

    db = pymongo.MongoClient().mydb
    query_doc = {}
    # product_collection = db.products
    # query_doc = {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_name)}}}}

    #during first run we should record date
    if first_run is True:
        db.globals.update({"_id": "FP_DATE"}, {"$set": {category_name: datetime.datetime.now()}})
        query_doc = {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_name)}}}}

    #else find all docs with date earlier than date recorded in first_run
    else:
        fp_start_date = db.globals.find_one({"_id": "FP_DATE"})[category_name]
        query_doc = {"$and": [
            {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_name)}}}},
            {"fp_date": {"$lt": fp_start_date}}]}

    query = db.products.find(query_doc).batch_size(100) # batch_size required because cursor timed out without it

    total_items = query.count()

    i = 1
    # boxes_found = 0

    for doc in query:
        print "Starting {i} of {total}...".format(i=i, total=total_items)
        image_url = doc["image"]["sizes"]["XLarge"]["url"]
        print "Image URL: {0}".format(image_url)

        # if there is a valid human BB, use it
        if "human_bb" in doc.keys() and doc["human_bb"] != [0, 0, 0, 0] and doc["human_bb"] is not None:
            chosen_bounding_box = doc["human_bb"]
            logging.debug("Human bb found: {bb} for item: {id}".format(bb=chosen_bounding_box, id=doc["id"]))
        #otherwise use the largest of possibly many classifier bb's
        else:
            try:
                bounding_box_list = classify.classify_image_with_classifiers(image_url, classifier_xml)[classifier_xml]
            except KeyError:
                logging.warning("Could not classify with {0}".format(classifier_xml))
                bounding_box_list = []

            max_bb_area = 0
            chosen_bounding_box = None
            for possible_bb in bounding_box_list:
                if possible_bb[2] * possible_bb[3] > max_bb_area:
                    chosen_bounding_box = possible_bb
                    max_bb_area = possible_bb[2] * possible_bb[3]
            if chosen_bounding_box is None:
                chosen_bounding_box = [0, 0, 0, 0]
                logging.warning("No Bounding Box found, using [0,0,0,0]. "
                                "Document id: {0}, BB_list: {1}".format(doc["id"], str(bounding_box_list)))

        try:
            fingerprint = fp.fp(get_cv2_img_array(image_url), chosen_bounding_box)

            db.products.update({"id": doc["id"]},
                                      {"$set": {"fingerprint": fingerprint.tolist(),
                                                "fp_date": datetime.datetime.now(),
                                                "bounding_box": np.array(chosen_bounding_box).tolist()}
                                      })
            i += 1

        except Exception as e:
            logging.warning("Exception caught while fingerprinting, skipping: {0}".format(e))


if __name__ == "__main__":
    main()

"""
def print_children(category_id, hierarchy_string):
    curr_cat = category_collection.find_one({"id": category_id})
    print hierarchy_string + curr_cat["name"]
    if "childrenIds" in curr_cat.keys():
        for childId in curr_cat["childrenIds"]:
            printChildren(childId, " - - " + hierarchy_string)
"""


