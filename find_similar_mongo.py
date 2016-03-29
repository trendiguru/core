__author__ = 'liorsabag'

import os
import subprocess

import cv2
import numpy as np

from . import NNSearch
from . import Utils
from . import background_removal
from . import constants
from . import fingerprint_core as fp
from . import kassper

fingerprint_length = constants.fingerprint_length
histograms_length = constants.histograms_length
# fp_weights = constants.fingerprint_weights
FP_KEY = "fingerprint"
db = constants.db


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


def find_top_n_results(image, mask, number_of_results=10, category_id=None, collection="products",
                       fp_category=FP_KEY, fp_len=fingerprint_length, distance_function=None,
                       bins=histograms_length, fingerprint=None):
    '''
    for comparing 2 fp call the function twice, both times with collection_name ='fp_testing' :
      - for the control group leave fp_category as is
      - fot the test group call the function with fp_category="new_fp"
    if the new fingerprint has a new length then make sure that the color_fp length
      is correct by entering the correct fp_len
    if a distance_function other than Bhattacharyya is used then call the function with that distance function's name
    '''
    fp_weights = constants.fingerprint_weights
    collection = db[collection]
    print "query collection name is: {0}".format(str(collection))
    print "number of results to search: {0}".format(number_of_results)
    print "category: {0}".format(category_id)
    # subcategory_id_list = get_all_subcategories(db.categories, category_id)

    # get all items in the subcategory/keyword
    potential_matches_cursor = collection.find(
        {"categories": category_id},
        {"_id": 1, "id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1}).batch_size(100)

    print "amount of docs in cursor: {0}".format(potential_matches_cursor.count())
    if not fingerprint:
        fingerprint = fp.fp(image, bins, fp_len, mask)
    target_dict = {"clothingClass": category_id, "fingerprint": fingerprint}
    print "calling find_n_nearest.."
    closest_matches = NNSearch.find_n_nearest_neighbors(target_dict, potential_matches_cursor, number_of_results,
                                                        fp_weights, bins, fp_category, distance_function)

    print "done with find_n_nearest.. num of closest_matches: {0}".format(len(closest_matches))
    # get only the object itself, not the distance

    return fingerprint.tolist(), closest_matches


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





