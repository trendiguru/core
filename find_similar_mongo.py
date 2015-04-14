__author__ = 'liorsabag'

import os
import subprocess

import pymongo
import cv2
import numpy as np

import fingerprint as fp
import NNSearch
import background_removal
import Utils
import kassper
import constants

# comment


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


def mask2svg(mask, filename, address):
    """
    this function takes a binary mask and turns it into a svg file
    :param mask: 0/255 binary 2D image
    :param filename: item id string
    :param address: address string
    :return: the path of the svg file
    """
    mask = 255 - mask
    os.chdir(address)
    cv2.imwrite(filename + '.bmp', mask)                                # save as a bmp image
    subprocess.call('potrace -s ' + filename + '.bmp' + ' -o ' + filename + '.svg', shell=True)  # create the svg
    os.remove(filename + '.bmp')  # remove the bmp mask
    return filename + '.svg'


def got_bb(image_url, post_id, bb=None, number_of_results=10, category_id=None):
    svg_address = constants.svg_address
    image = Utils.get_cv2_img_array(image_url)                                    # turn the URL into a cv2 image
    small_image, resize_ratio = background_removal.standard_resize(image, 400)    # shrink image for faster process
    if bb is not None:
        bb = [int(b) for b in (np.array(bb) / resize_ratio)]  # shrink bb in the same ratio
    fg_mask = background_removal.get_fg_mask(small_image, bb)                     # returns the grab-cut mask (if bb => PFG-PBG gc, if !bb => face gc)
    # bb_mask = background_removal.get_binary_bb_mask(small_image, bb)            # bounding box mask
    # combined_mask = cv2.bitwise_and(fg_mask, bb_mask)                           # for sending the right mask to the fp
    gc_image = background_removal.get_masked_image(small_image, fg_mask)
    face_rect = background_removal.find_face(small_image)
    if len(face_rect) > 0:
        x, y, w, h = face_rect[0]
        face_image = image[y:y+h, x:x+w, :]
        without_skin = kassper.skin_removal(face_image, gc_image)
        crawl_mask = kassper.clutter_removal(without_skin, 200)
        without_clutter = background_removal.get_masked_image(without_skin, crawl_mask)
        mask = kassper.get_mask(without_clutter)
    else:
        mask = fg_mask
    fp_vector, closest_matches = find_top_n_results(small_image, mask, number_of_results, category_id)
    svg_filename = mask2svg(mask, post_id, svg_address)
    svg_url = constants.svg_url_prefix + svg_filename
    return fp_vector, closest_matches, svg_url


def find_top_n_results(image, mask, number_of_results=10, category_id=None):
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

    color_fp = fp.fp(image, mask)
    target_dict = {"clothingClass": category_id, "fingerPrintVector": color_fp}
    closest_matches = NNSearch.find_n_nearest_neighbors(target_dict, db_fingerprint_list, number_of_results)
    return color_fp.tolist(), closest_matches


def find_top_n_results_using_grabcut(image_url, post_id=None, bb=None, number_of_results=10, category_id=None,
                                     do_svg=True):
    image = Utils.get_cv2_img_array(image_url)  # turn the URL into a cv2 image
    small_image, resize_ratio = background_removal.standard_resize(image, 400)  # shrink image for faster process
    bb = [int(b) for b in (np.array(bb) / resize_ratio)]  # shrink bb in the same ratio

    fg_mask = background_removal.get_fg_mask(small_image,
                                             bb)  # returns the grab-cut mask (if bb => PFG-PBG gc, if !bb => face gc)
    # bb_mask = background_removal.get_binary_bb_mask(small_image, bb)  # bounding box mask
    # combined_mask = cv2.bitwise_and(fg_mask, bb_mask)  # for sending the right mask to the fp
    gc_image = background_removal.get_masked_image(small_image, fg_mask)
    face_rect = background_removal.find_face(small_image)
    if len(face_rect) > 0:
        x, y, w, h = face_rect[0]
        face_image = image[y:y+h, x:x+w, :]
        without_skin = kassper.skin_removal(face_image, gc_image)
        crawl_mask = kassper.clutter_removal(without_skin, 200)
        without_clutter = background_removal.get_masked_image(without_skin, crawl_mask)
        mask = kassper.get_mask(without_clutter)
    else:
        mask = kassper.get_mask(gc_image)

    fp_vector, closest_matches = find_top_n_results(gc_image, mask, number_of_results, category_id)

    if do_svg:
        if post_id is None:
            print('error - svg wanted but no post_id given')
            return None
        svg_address = constants.svg_address
        svg_filename = mask2svg(mask, post_id, svg_address)
        svg_url = constants.svg_url_prefix + svg_filename
        return fp_vector, closest_matches, svg_url
    else:
        return fp_vector, closest_matches

