__author__ = 'liorsabag'

import pymongo
import fingerprint as fp
from NNSearch import findNNs
import logging
import background_removal
import Utils
import numpy as np
import classify_core


def get_classifiers():
    default_classifiers = ["/home/www-data/web2py/applications/fingerPrint/modules/shirtClassifier.xml",
                           "/home/www-data/web2py/applications/fingerPrint/modules/pantsClassifier.xml",
                           "/home/www-data/web2py/applications/fingerPrint/modules/dressClassifier.xml"]
    classifiers_dict = {'shirt': '/home/www-data/web2py/applications/fingerPrint/modules/shirtClassifier.xml',
                        'pants': '/home/www-data/web2py/applications/fingerPrint/modules/pantsClassifier.xml',
                        'dress': '/home/www-data/web2py/applications/fingerPrint/modules/dressClassifier.xml'}
    return default_classifiers, classifiers_dict


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


def find_top_n_results(imageURL, number_of_results=10, bb=None, category_id=None):
    # if (bb is None) or (bb == np.array([0, 0, 0, 0])).all():
    #     raise NotImplementedError
        # masked_image = background_removal.get_masked_image(small_image, fg_mask)    # returns small image after GC masking
        # bb_dict = classify_core.classify_image_with_classifiers(masked_image,
                                                           # get_classifiers()[0], get_classifiers()[1])
    db = pymongo.MongoClient().mydb
    product_collection = db.products

    subcategory_id_list = get_all_subcategories(db.categories, category_id)

    # get all items in the subcategory/keyword
    query = product_collection.find({"$and": [{"categories": {"$elemMatch": {"id": {"$in": subcategory_id_list}}}},
                                              {"fingerprint": {"$exists": 1}}]},
                                    {"_id": 0, "id": 1, "categories": 1, "fingerprint": 1, "image": 1,
                                     "clickUrl": 1, "price": 1, "brand": 1})
    db_fingerprint_list = []
    for row in query:
        fp_dict = {}
        fp_dict["id"] = row["id"]
        fp_dict["clothingClass"] = category_id
        fp_dict["fingerPrintVector"] = row["fingerprint"]
        fp_dict["imageURL"] = row["image"]["sizes"]["Large"]["url"]
        fp_dict["buyURL"] = row["clickUrl"]
        db_fingerprint_list.append(fp_dict)

    image = Utils.get_cv2_img_array(imageURL)                                     # turn the URL into a cv2 image
    small_image, resize_ratio = background_removal.standard_resize(image, 400)    # shrink image for faster process
    fg_mask = background_removal.get_fg_mask(small_image, bb)                     # returns the grab-cut mask (if bb => PFG-PBG gc, if !bb => face gc)
    combined_mask = fg_mask + background_removal.get_bb_mask(small_image, bb)     # for sending the right mask to the fp
    color_fp = fp.fp(small_image, combined_mask)
    # Fingerprint the bounded area
    target_dict = {"clothingClass": category_id, "fingerPrintVector": color_fp}
    closest_matches = findNNs(target_dict, db_fingerprint_list, number_of_results)
    return color_fp.tolist(), closest_matches
