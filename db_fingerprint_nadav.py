__author__ = 'Nadav Paz'

import sys
import fingerprint_core as fp
import classify_core as classify
from bson import datetime
import logging
import pymongo
import numpy as np
import background_removal
import Utils
import cv2


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


def main():
    """
    example: python fingerprint_db_params_mongo.py mens-shirts shirtClassifier.xml true
    to run on entire database and fingerprint anything not that doesnt already have a fingerprint, do:
    python fingerprint_db_params_mongo.py undone
    """
# logging.basicConfig(filename='fingerprint_db.log', level=logging.DEBUG)
    first_run = False
    print(sys.argv)
    if len(sys.argv) == 2:
        category_name = sys.argv[1]
    if len(sys.argv) >= 3:
        category_name = sys.argv[1]
        classifier_xml = sys.argv[2]
        if len(sys.argv) == 4 and sys.argv[3].lower() == "true":
            first_run = True
    else:
        print "Missing parameters. Example use: python fingerprint" \
              "_db_params_mongo.py mens-shirts shirtClassifier.xml true"

    db = pymongo.MongoClient().mydb
    query_doc = {}
    # during first run we should record date
    if first_run is True:
        db.globals.update({"_id": "FP_DATE"}, {"$set": {category_name: datetime.datetime.now()}})
        query_doc = {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_name)}}}}

    # else find all docs with date earlier than date recorded in first_run
    else:
        fp_start_date = db.globals.find_one({"_id": "FP_DATE"})[category_name]
        query_doc = {"$and": [
            {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_name)}}}},
            {"fp_date": {"$lt": fp_start_date}}]}

    query = db.products.find(query_doc).batch_size(100)  # batch_size required because cursor timed out without it

    total_items = query.count()

    i = 1
    # boxes_found = 0

    for doc in query:
        print "Starting {i} of {total}...".format(i=i, total=total_items)
        image_url = doc["image"]["sizes"]["XLarge"]["url"]
        image = Utils.get_cv2_img_array(image_url)
        print "Image URL: {0}".format(image_url)

        # if there is a valid human BB, use it
        if "human_bb" in doc.keys() and doc["human_bb"] != [0, 0, 0, 0] and doc["human_bb"] is not None:
            chosen_bounding_box = doc["human_bb"]
            mask = background_removal.get_fg_mask(image, chosen_bounding_box)
            logging.debug("Human bb found: {bb} for item: {id}".format(bb=chosen_bounding_box, id=doc["id"]))
        # otherwise use the largest of possibly many classifier bb's
        else:
            mask = background_removal.get_fg_mask(image)
            white_bckgnd_image = background_removal.image_white_bckgnd(image, mask)
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
                chosen_bounding_box = [0, 0, image.shape[1], image.shape[0]]
                logging.warning("No Bounding Box found, using the whole image. "
                                "Document id: {0}, BB_list: {1}".format(doc["id"], str(bounding_box_list)))
                mask = np.ones((np.shape(image)))
            else:
                bb_mask = background_removal.get_bb_mask(image, chosen_bounding_box)
                mask = cv2.bitwise_and(mask, bb_mask)
        try:
            fingerprint = fp.fp(image, mask)
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
