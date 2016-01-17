__author__ = 'yonatan'
"""
this is an "hard copy" of all most of the relevant functions related to the demo
so it wouldn't get massed up with every change of the main functions

*notice that for simplicity paperdoll returns no more than one person!
"""

import copy

import numpy as np
import bson

from . import paperdolls
from . import Utils
from . import background_removal
from .paperdoll import paperdoll_parse_enqueue
from .paperdolls import after_pd_conclusions
from . import constants
from . import fingerprint_core as fp
from . import NNSearch
from .fingerprint_stuff.fp_tests import mr8_worker
from . import page_results
from . import find_similar_mongo
from . import new_finger_print

fp_weights = constants.fingerprint_weights
bins = constants.histograms_length
fp_len = constants.fingerprint_length
db = constants.db

def person_isolation(image, face):
    x, y, w, h = face
    image_copy = np.zeros(image.shape, dtype=np.uint8)
    x_back = np.max([x - 1.5 * w, 0])
    x_ahead = np.min([x + 2.5 * w, image.shape[1] - 2])
    image_copy[:, int(x_back):int(x_ahead), :] = image[:, int(x_back):int(x_ahead), :]
    return image_copy


def find_top_n_results(fp, mr8, category, number_of_results, collection, wing, weight):
    '''
    for comparing 2 fp call the function twice, both times with collection_name ='fp_testing' :
      - for the control group leave fp_category as is
      - fot the test group call the function with fp_category="new_fp"
    if the new fingerprint has a new length then make sure that the color_fp length
      is correct by entering the correct fp_len
    if a distance_function other than Bhattacharyya is used then call the function with that distance function's name
    '''
    fp_weights = constants.fingerprint_weights
    collection = constants.db[collection]

    # get all items in the category
    potential_matches_cursor = collection.find(
        {"categories": category},
        {"_id": 1, "id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1, "mr8": 1}).batch_size(100)

    print "amount of docs in cursor: {0}".format(potential_matches_cursor.count())
    color_fp = fp
    if wing == "right":
        mr8 = mr8
    else:
        mr8 = []
    target_dict = {"clothingClass": category, "fingerprint": color_fp, "mr8": mr8}
    print "calling find_n_nearest.."
    closest_matches = find_n_nearest_neighbors(target_dict, potential_matches_cursor, number_of_results,
                                               fp_weights, bins, wing, weight)

    print "done with find_n_nearest.."
    # get only the object itself, not the distance
    closest_matches = [match_tuple[0] for match_tuple in closest_matches]

    return closest_matches


def find_top_n_results_nate(fp, method):
    '''
    for comparing 2 fp call the function twice, both times with collection_name ='fp_testing' :
      - for the control group leave fp_category as is
      - fot the test group call the function with fp_category="new_fp"
    if the new fingerprint has a new length then make sure that the color_fp length
      is correct by entering the correct fp_len
    if a distance_function other than Bhattacharyya is used then call the function with that distance function's name
    '''
    collection = constants.db["nate_testing"]

    # get all items in the category
    potential_matches_cursor = collection.find(
        {"categories": "dress"},
        {"_id": 1, "id": 1, "images.XLarge": 1, "clickUrl": 1, method: 1}).batch_size(10000)  # limit(10000)

    print "amount of docs in cursor: {0}".format(potential_matches_cursor.count())

    target_dict = {"categories": "dress", method: fp}

    print "calling find_n_nearest.."
    if method == "fingerprint":
        closest_matches = find_n_nearest_neighbors_nate(method, target_dict, potential_matches_cursor)

    elif method == "specio":
        for i in range(len(fp)):
            fp[i] = np.array(fp[i])
        target_dict["sp_one"] = fp[0]
        target_dict["sp_two"] = fp[1]
        target_dict["specio"] = fp

        potential_matches_cursor = collection.find(
            {"categories": "dress"},
            {"_id": 1, "id": 1, "sp_one": 1}).batch_size(15000)  # limit
        closest_matches = find_n_nearest_neighbors_nate(method, target_dict, potential_matches_cursor,
                                                        rank=1, b_size=1000)
        id_list = [item[0]["id"] for item in closest_matches]
        query2 = collection.find({"id": {"$in": id_list}}, {"_id": 1, "id": 1, "sp_two": 1}).batch_size(1000)
        closest_matches = find_n_nearest_neighbors_nate(method, target_dict, query2,
                                                        rank=2, b_size=100)
        id_list2 = [item[0]["id"] for item in closest_matches]
        query3 = collection.find({"id": {"$in": id_list2}},
                                 {"_id": 1, "id": 1, "specio": 1, "images.XLarge": 1, "clickUrl": 1}).batch_size(100)
        closest_matches = find_n_nearest_neighbors_nate(method, target_dict, query3,
                                                        rank=3, b_size=20)
    else:
        return []
    print "done with find_n_nearest.."
    # get only the object itself, not the distance
    closest_matches = [match_tuple[0] for match_tuple in closest_matches]

    return closest_matches

def trim_mr8(mr8, shift):
    shift_right = [x + shift for x in mr8]
    chop_top = [max(0, x) for x in shift_right]
    chop_bottom = [min(0, x) for x in chop_top]
    return chop_bottom


def distance_function(entry, target_dict, fp_weights, hist_length, wing, weight):
    key = "fingerprint"
    bhat = NNSearch.distance_Bhattacharyya(entry[key], target_dict[key], fp_weights, hist_length)
    if wing == "left":
        return bhat
    elif wing == "right":
        # shift = 2
        # entry_mr8 = trim_mr8(entry["mr8"], shift)
        # target_mr8 = trim_mr8(target_dict["mr8"], shift)
        # mr8_distance = NNSearch.distance_1_k(entry_mr8, target_mr8)
        # mr8_normal = mr8_distance
        # w0 = abs(1 - int(weight))
        # return w0 * bhat + weight * mr8_normal
        entry_mr8 = entry["mr8"]
        target_mr8 = []
        for i in target_dict["mr8"]:
            target_mr8 += i
        mr8_distance = NNSearch.distance_1_k(entry_mr8, target_mr8)
        w0 = abs(1 - weight)
        return w0 * bhat + weight * mr8_distance
    return bhat


def distance_function_nate(entry, target_dict, method, rank):
    if method == "specio":
        if rank == 1:
            dist = new_finger_print.spaciograms_distance_rating(np.array(entry["sp_one"]), target_dict["sp_one"], rank)
        elif rank == 2:
            dist = new_finger_print.spaciograms_distance_rating(np.array(entry["sp_two"]), target_dict["sp_two"], rank)
        elif rank == 3:
            for i in range(len(entry["specio"])):
                entry["specio"][i] = np.array(entry["specio"][i])
            dist = new_finger_print.spaciograms_distance_rating(entry["specio"], target_dict["specio"], rank)
    elif method == "histo":
        dist = NNSearch.distance_1_k(np.asarray(entry[method]), target_dict[method])
    else:
        dist = NNSearch.distance_Bhattacharyya(entry[method], target_dict[method], fp_weights, bins)

    return dist


def find_n_nearest_neighbors_nate(method, target_dict, entries, rank=1, b_size=20):
    # list of tuples with (entry,distance). Initialize with first n distance values
    nearest_n = []
    farthest_nearest = 20000

    for i, entry in enumerate(entries):
        if i < b_size:
            d = distance_function_nate(entry, target_dict, method, rank)
            nearest_n.append((entry, d))
        else:
            if i == b_size:
                # sort by distance
                nearest_n.sort(key=lambda tup: tup[1])
                # last item in the list (index -1, go python!)
                farthest_nearest = nearest_n[-1][1]

            # Loop through remaining entries, if one of them is better, insert it in the correct location and remove last item
            d = distance_function_nate(entry, target_dict, method, rank)
            if d < farthest_nearest:
                insert_at = b_size - 2
                while d < nearest_n[insert_at][1]:
                    insert_at -= 1
                    if insert_at == -1:
                        break
                nearest_n.insert(insert_at + 1, (entry, d))
                nearest_n.pop()
                farthest_nearest = nearest_n[-1][1]
    return nearest_n


def find_n_nearest_neighbors(target_dict, entries, number_of_matches, fp_weights, hist_length, wing, weight):
    # list of tuples with (entry,distance). Initialize with first n distance values
    nearest_n = []
    farthest_nearest = 2
    for i, entry in enumerate(entries):
        if i < number_of_matches:
            d = distance_function(entry, target_dict, fp_weights, hist_length, wing, weight)
            nearest_n.append((entry, d))
        else:
            if i == number_of_matches:
                # sort by distance
                nearest_n.sort(key=lambda tup: tup[1])
                # last item in the list (index -1, go python!)
                farthest_nearest = nearest_n[-1][1]

            # Loop through remaining entries, if one of them is better, insert it in the correct location and remove last item
            d = distance_function(entry, target_dict, fp_weights, hist_length, wing, weight)
            if d < farthest_nearest:
                insert_at = number_of_matches - 2
                while d < nearest_n[insert_at][1]:
                    insert_at -= 1
                    if insert_at == -1:
                        break
                nearest_n.insert(insert_at + 1, (entry, d))
                nearest_n.pop()
                farthest_nearest = nearest_n[-1][1]
    return nearest_n

def get_svg(image_url):
    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        return

    image_hash = page_results.get_hash_of_image_from_url(image_url)
    # NEW_IMAGE !!
    clean_image = copy.copy(image)
    relevance = background_removal.image_is_relevant(image, True, image_url)
    image_dict = {'image_urls': [image_url], 'relevant': relevance.is_relevant,
                  'image_hash': image_hash, 'page_urls': ["True"], 'people': []}
    if relevance.is_relevant:
        idx = 0
        if len(relevance.faces):
            if not isinstance(relevance.faces, list):
                relevant_faces = relevance.faces.tolist()
            else:
                relevant_faces = relevance.faces
            for face in relevant_faces:
                image_copy = person_isolation(image, face)
                person = {'face': face, 'person_id': str(bson.ObjectId()), 'person_idx': 1,
                          'items': []}
                mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(image_copy, async=False).result[:3]
                final_mask = after_pd_conclusions(mask, labels, person['face'])
                # image = draw_pose_boxes(pose, image)
                item_idx = 0
                for num in np.unique(final_mask):
                    # convert numbers to labels
                    category = list(labels.keys())[list(labels.values()).index(num)]
                    if category in constants.paperdoll_shopstyle_women.keys():
                        item_mask = 255 * np.array(final_mask == num, dtype=np.uint8)
                        item_dict = {"category": category, 'item_id': str(bson.ObjectId()), 'item_idx': item_idx,
                                     'face': face}
                        svg_name = find_similar_mongo.mask2svg(
                            item_mask,
                            str(image_dict['image_hash']) + '_' + person['person_id'] + '_' + item_dict['category'],
                            constants.svg_folder)
                        item_dict["svg_url"] = constants.svg_url_prefix + svg_name
                        item_dict["type"] = category
                        # item_dict["mask"] = item_mask.tolist()
                        item_dict["fp"] = fp.fp(image, bins, fp_len, item_mask).tolist()
                        print(item_dict['face'])
                        item_dict["mr8"] = mr8_worker.mr8_4_demo(image, item_dict['face'], item_mask)
                        print(6)
                        person['items'] = [item_dict]

                        idx = db.demo_yonti.insert_one(item_dict).inserted_id
                        item = {"idx": idx, "svg_url": item_dict["svg_url"], "category": item_dict["category"]}
                        data = {"items": [item]}
                        return data

        else:
            # no faces, only general positive human detection
            person = {'face': [], 'person_id': str(bson.ObjectId()), 'person_idx': 1, 'items': []}
            mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(image, async=False).result[:3]
            final_mask = after_pd_conclusions(mask, labels)
            item_idx = 0
            for num in np.unique(final_mask):
                # convert numbers to labels
                category = list(labels.keys())[list(labels.values()).index(num)]
                if category in constants.paperdoll_shopstyle_women.keys():
                    item_mask = 255 * np.array(final_mask == num, dtype=np.uint8)
                    item_dict = {"category": category, 'item_id': str(bson.ObjectId()), 'item_idx': item_idx}
                    svg_name = find_similar_mongo.mask2svg(
                        item_mask,
                        str(image_dict['image_hash']) + '_' + person['person_id'] + '_' + item_dict['category'],
                        constants.svg_folder)
                    item_dict["svg_url"] = constants.svg_url_prefix + svg_name
                    item_dict["type"] = category
                    # item_dict["mask"] = item_mask
                    item_dict["fp"] = fp.fp(image, bins, fp_len, mask)
                    item_dict["mr8"] = mr8_worker.mr8_4_demo(image, item_dict['face'], mask)

                    # person['items'] = [item_dict]
                    # image_dict['items'] = [item for item in person["items"]]
                    idx = db.demo_yonti.insert_one(item_dict).inserted_id
                    item = {"idx": idx, "svg_url": item_dict["svg_url"], "category": item_dict["category"]}
                    data = {"items": [item]}
                    return data
                    #             person['items'].append(item_dict)
                    #             item_idx += 1
                    #     image_dict['people'].append(person)
                    # return page_results.merge_items(image_dict)
    else:  # if not relevant
        return


def get_svg_nate(image_url):
    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        return

    image_hash = page_results.get_hash_of_image_from_url(image_url)
    # NEW_IMAGE !!
    relevance = background_removal.image_is_relevant(image, True, image_url)
    image_dict = {'image_urls': [image_url], 'relevant': relevance.is_relevant,
                  'image_hash': image_hash, 'page_urls': ["True"], 'people': []}
    if relevance.is_relevant:
        idx = 0
        if len(relevance.faces):
            if not isinstance(relevance.faces, list):
                relevant_faces = relevance.faces.tolist()
            else:
                relevant_faces = relevance.faces
            for face in relevant_faces:
                image_copy = person_isolation(image, face)
                person = {'face': face, 'person_id': str(bson.ObjectId()), 'person_idx': 1,
                          'items': []}
                mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(image_copy, async=False, at_front=True,
                                                                               queue_name='pd_yonti').result[:3]
                final_mask = after_pd_conclusions(mask, labels, person['face'])
                # image = draw_pose_boxes(pose, image)
                item_idx = 0
                for num in np.unique(final_mask):
                    # convert numbers to labels
                    category = list(labels.keys())[list(labels.values()).index(num)]
                    if category == "dress":
                        item_mask = 255 * np.array(final_mask == num, dtype=np.uint8)
                        small_image, resize_ratio = background_removal.standard_resize(image, 400)
                        mask2 = background_removal.get_fg_mask(small_image)
                        item_dict = {"category": category, 'item_id': str(bson.ObjectId()), 'item_idx': item_idx,
                                     'face': face}
                        svg_name = find_similar_mongo.mask2svg(
                            item_mask,
                            str(image_dict['image_hash']) + '_' + person['person_id'] + '_' + item_dict['category'],
                            constants.svg_folder)
                        item_dict["svg_url"] = constants.svg_url_prefix + svg_name
                        item_dict["type"] = category
                        # item_dict["mask"] = item_mask.tolist()
                        item_dict["fp"] = fp.fp(image, bins, fp_len, item_mask).tolist()
                        print(item_dict['face'])
                        item_bb = paperdolls.bb_from_mask(item_mask)
                        item_gc_mask = background_removal.paperdoll_item_mask(item_mask, item_bb)
                        after_gc_mask = background_removal.get_fg_mask(image, item_bb)  # (255, 0) mask
                        specio = new_finger_print.spaciogram_finger_print(image, item_mask)
                        item_dict["specio"] = specio
                        item_dict["sp_one"] = item_dict["specio"][0]
                        item_dict["sp_two"] = item_dict["specio"][1]
                        histo = new_finger_print.histogram_stack_finger_print(image, after_gc_mask)
                        item_dict["histo"] = histo.tolist()
                        person['items'] = [item_dict]

                        # image_dict['items'] = [item for item in person["items"]]
                        idx = db.demo_yonti.insert_one(item_dict).inserted_id
                        item = {"idx": idx, "svg_url": item_dict["svg_url"], "category": item_dict["category"]}
                        data = {"items": [item]}
                        return data

        else:
            # no faces, only general positive human detection
            person = {'face': [], 'person_id': str(bson.ObjectId()), 'person_idx': 1, 'items': []}
            mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(image, async=False,
                                                                           queue_name='pd_yonti').result[:3]
            final_mask = after_pd_conclusions(mask, labels)
            item_idx = 0
            for num in np.unique(final_mask):
                # convert numbers to labels
                category = list(labels.keys())[list(labels.values()).index(num)]
                if category in constants.paperdoll_shopstyle_women.keys():
                    item_mask = 255 * np.array(final_mask == num, dtype=np.uint8)
                    item_dict = {"category": category, 'item_id': str(bson.ObjectId()), 'item_idx': item_idx}
                    svg_name = find_similar_mongo.mask2svg(
                        item_mask,
                        str(image_dict['image_hash']) + '_' + person['person_id'] + '_' + item_dict['category'],
                        constants.svg_folder)
                    item_dict["svg_url"] = constants.svg_url_prefix + svg_name
                    item_dict["type"] = category
                    # item_dict["mask"] = item_mask.tolist()
                    item_dict["fp"] = fp.fp(image, bins, fp_len, item_mask).tolist()
                    print(item_dict['face'])
                    item_bb = paperdolls.bb_from_mask(item_mask)
                    item_gc_mask = background_removal.paperdoll_item_mask(item_mask, item_bb)
                    after_gc_mask = background_removal.get_fg_mask(image, item_bb)  # (255, 0) mask
                    specio = new_finger_print.spaciogram_finger_print(image, item_mask)
                    item_dict["specio"] = specio
                    item_dict["sp_one"] = item_dict["specio"][0]
                    item_dict["sp_two"] = item_dict["specio"][1]
                    histo = new_finger_print.histogram_stack_finger_print(image, after_gc_mask)
                    item_dict["histo"] = histo.tolist()
                    person['items'] = [item_dict]

                    idx = db.demo_yonti.insert_one(item_dict).inserted_id
                    item = {"idx": idx, "svg_url": item_dict["svg_url"], "category": item_dict["category"]}
                    data = {"items": [item]}
                    return data

    else:  # if not relevant
        return

def get_results_now(idx, collection="mr8_testing", wing="left", weight='0.05'):
    oid = bson.ObjectId(idx)
    item = db.demo_yonti.find_one({"_id": oid})
    fp = item["fp"]
    mr8 = item["mr8"]
    category = "dress"
    item_dict = {"similar_results": find_top_n_results(fp,
                                                       mr8,
                                                       category,
                                                      100,
                                                      collection,
                                                      wing,
                                                       float(weight))}

    return item_dict


def get_results_now_nate(idx, method):
    oid = bson.ObjectId(idx)
    item = db.demo_yonti.find_one({"_id": oid})
    if method == "specio":
        fp = np.asarray(item["specio"])
    elif method == "histo":
        fp = np.asarray(item["histo"])
    else:
        fp = item["fp"]
    item_dict = {"similar_results": find_top_n_results_nate(fp, method)}

    return item_dict
