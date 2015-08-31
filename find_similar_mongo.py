__author__ = 'liorsabag'

import os
import subprocess

import pymongo
import cv2
import numpy as np

import fingerprint_core as fp
import NNSearch
import background_removal
import Utils
import kassper
import constants


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


def mask2svg(mask, filename, save_in_folder):
    """
    this function takes a binary mask and turns it into a svg file
    :param mask: 0/255 binary 2D image
    :param filename: item id string
    :param save_in_folder: address string
    :return: the path of the svg file
    """
    mask = 255 - mask
    os.chdir(save_in_folder)
    cv2.imwrite(filename + '.bmp', mask)                                # save as a bmp image
    subprocess.call('potrace -s ' + filename + '.bmp'
                    + ' -o ' + filename + '.svg'
                    + ' -t 100', shell=True)  # create the svg
    os.remove(filename + '.bmp')  # remove the bmp mask
    return filename + '.svg'


def find_top_n_results(image, mask, number_of_results=10, category_id=None):
    db = pymongo.MongoClient().mydb
    product_collection = db.products
    subcategory_id_list = get_all_subcategories(db.categories, category_id)

    # get all items in the subcategory/keyword
    potential_matches_cursor = product_collection.find(
        {"$and": [{"categories": {"$elemMatch": {"id": {"$in": subcategory_id_list}}}},
                  {"fingerprint": {"$exists": 1}}]},
        {"_id": 1, "id": 1, "fingerprint": 1, "image.sizes.XLarge.url": 1})

    # db_fingerprint_list = []
    # for row in potential_matches_cursor:
    #     fp_dict = {}
    #     fp_dict["id"] = row["id"]
    #     fp_dict["clothingClass"] = category_id
    #     fp_dict["fingerPrintVector"] = row["fingerprint"]
    #     fp_dict["imageURL"] = row["image"]["sizes"]["Large"]["url"]
    #     fp_dict["buyURL"] = row["clickUrl"]
    #     db_fingerprint_list.append(fp_dict)

    color_fp = fp.fp(image, mask)
    target_dict = {"clothingClass": category_id, "fingerprint": color_fp}
    closest_matches = NNSearch.find_n_nearest_neighbors(target_dict, potential_matches_cursor, number_of_results,
                                                        fp_key="fingerprint")
    # get only the object itself, not the distance
    closest_matches = [match_tuple[0] for match_tuple in closest_matches]

    return color_fp.tolist(), closest_matches


def got_bb(image_url, post_id, item_id, bb=None, number_of_results=10, category_id=None):
    svg_folder = constants.svg_folder
    full_item_id = post_id + "_" + item_id
    image = Utils.get_cv2_img_array(image_url)                                    # turn the URL into a cv2 image
    small_image, resize_ratio = background_removal.standard_resize(image, 400)    # shrink image for faster process
    if bb is not None:
        bb = [int(b) for b in (np.array(bb) / resize_ratio)]  # shrink bb in the same ratio
    fg_mask = background_removal.get_fg_mask(small_image, bb)                     # returns the grab-cut mask (if bb => PFG-PBG gc, if !bb => face gc)
    gc_image = background_removal.get_masked_image(small_image, fg_mask)
    without_skin = kassper.skin_removal(gc_image, small_image)
    crawl_mask = kassper.clutter_removal(without_skin, 400)
    without_clutter = background_removal.get_masked_image(without_skin, crawl_mask)
    fp_mask = kassper.get_mask(without_clutter)
    fp_vector, closest_matches = find_top_n_results(small_image, fp_mask, number_of_results, category_id)
    svg_filename = mask2svg(fp_mask, full_item_id, svg_folder)
    svg_url = constants.svg_url_prefix + svg_filename
    return fp_vector, closest_matches, svg_url





