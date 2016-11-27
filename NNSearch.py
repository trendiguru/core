__author__ = 'jeremy'
import cv2
import logging
from time import sleep, time
import numpy as np
from rq import Queue
import pymongo
import constants
from db_stuff.annoy_dir import fanni
from features import color

q = Queue('annoy', connection=constants.redis_conn)
db = constants.db
K = constants.K  # .5 is the same as Euclidean
tmp_log = '/home/developer/logs/NN.log'


def distance_1_k(fp1, fp2, k=K):
    """This calculates distance between to arrays. When k = .5 this is the same as Euclidean."""
    if fp1 is not None and fp2 is not None:
        f12 = np.abs(np.array(fp1) - np.array(fp2))
        f12_p = np.power(f12, 1 / k)
        return np.power(np.sum(f12_p), k)
    else:
        print("null fingerprint sent to distance_1_k ")
        logging.warning("null fingerprint sent to distance_1_k ")
        return None


def distance_Bhattacharyya(fp1, fp2, weights, hist_length):
    """
    This calculates distance between to arrays.
    the first 6 elements are calculated using euclidean distance (When k = .5 this is the same as Euclidean)
    the next elements are the hue, saturation and value histograms - their distances are measured separately
    using the Bhattacharyya distance.
    later all the 4 components are combined with different weights:
    - the first 6 elements get 5%
    - the hue histogram gets 50%
    - the saturation and value histograms share the other 45% equally
    """
    Bhattacharyya = 3
    fp_division = [6, 6 + hist_length[0], 6 + hist_length[0] + hist_length[1],
                   6 + hist_length[0] + hist_length[1] + hist_length[2]]

    if fp1 is not None and fp2 is not None:
        first6 = np.abs(np.array(fp1[:fp_division[0]]) - np.array(fp2[:fp_division[0]]))
        first6 = np.power(np.sum(np.power(first6, 1 / 0.5)), 0.5)
        hue_hist1 = np.float32(fp1[fp_division[0]:fp_division[1]])
        hue_hist2 = np.float32(fp2[fp_division[0]:fp_division[1]])
        hue = float(cv2.compareHist(hue_hist1, hue_hist2, Bhattacharyya))
        sat_hist1 = np.float32(fp1[fp_division[1]:fp_division[2]])
        sat_hist2 = np.float32(fp2[fp_division[1]:fp_division[2]])
        sat = float(cv2.compareHist(sat_hist1, sat_hist2, Bhattacharyya))
        val_hist1 = np.float32(fp1[fp_division[2]:])
        val_hist2 = np.float32(fp2[fp_division[2]:])
        val = float(cv2.compareHist(val_hist1, val_hist2, Bhattacharyya))
        score = weights[0] * first6 + weights[1] * hue + weights[2] * sat + weights[3] * val
        return score

    else:
        print("null fingerprint sent to Bhattacharyya ")
        logging.warning("null fingerprint sent to Bhattacharyya ")
        return None


def distance(category, main_fp, candidate_fp, coll):
    if isinstance(main_fp, list):
        logging.warning("main_fp in distance function is a LIST!")
        return None
    if isinstance(candidate_fp, list):
        logging.warning("candidate_fp in distance function is a LIST!")
        return None
    # if not main_fp.keys() == candidate_fp.keys():
    if not len(main_fp.keys()) == len(candidate_fp.keys()) or not sorted(main_fp.keys()) == sorted(candidate_fp.keys()):
        logging.warning("2 fps has different keys: main keys: {0}, cand keys: {1}".format(main_fp.keys(), candidate_fp.keys()))
        logging.warning("category is {0}, collection {1}".format(category, coll))
        main_fp = {key: value for key, value in main_fp.iteritems() if key in candidate_fp.keys()}
        candidate_fp = {key: value for key, value in candidate_fp.iteritems() if key in main_fp.keys()}

    d = 0
    weight_keys = constants.weights_per_category.keys()
    if category not in weight_keys:
        category = 'other'
    weights = constants.weights_per_category[category]
    for feature in main_fp.keys():
        if feature == 'color':
            dist = color.distance(main_fp[feature], candidate_fp[feature])
        else:
            if not main_fp[feature] or not candidate_fp[feature]:
                dist = 0
            elif len(main_fp[feature]) and len(candidate_fp[feature]):
                # print "main {0} vector: {1}".format(feature, main_fp[feature])
                # print "candidate {0} vector: {1}".format(feature, candidate_fp[feature])
                if isinstance(main_fp[feature], dict):
                    main_fp[feature] = main_fp[feature]['data']
                if isinstance(candidate_fp[feature], dict):
                    candidate_fp[feature] = candidate_fp[feature]['data']
                dist = l2_distance(main_fp[feature], candidate_fp[feature])
            else:
                dist == 0
        d += weights[feature]*dist
    return d


def annoy_search(collection, category, color_fingerprint, num_of_results=1000):
    annoy_job = q.enqueue(fanni.lumberjack, args=(collection, category, color_fingerprint, 'angular', num_of_results))
    while not annoy_job.is_finished and not annoy_job.is_failed:
        sleep(0.01)

    if annoy_job.is_failed:
        return []
    else:
        return annoy_job.result


def find_n_nearest_neighbors(fp, collection, category, number_of_matches, annoy_top=1000):
    # t1 = time()
    entries = db[collection].find({'categories': category},
                                  {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
    # t2 = time()
    # print t2 - t1
    # t1 = time()
    if entries.count() > 2000 and 'amazon_DE' not in collection:

        annoy_top_results = annoy_search(collection, category, fp['color'], annoy_top)
        # print "annoy_search took {0} secs".format(time()-start)
        if not len(annoy_top_results):
            return []
        try:
            # entries = db[collection].find({'categories': category, "AnnoyIndex": {"$in": annoy_top_results}},
            #                               {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
            # entries = db[collection].find({"AnnoyIndex": {"$in": annoy_top_results}, 'categories': category},
            #                               {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1},
            #                               cursor_type=pymongo.cursor.CursorType.EXHAUST).hint([('AnnoyIndex', 1)])
            entries = db[collection].find({"AnnoyIndex": {"$in": annoy_top_results}, 'categories': category},
                                          {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1},
                                          cursor_type=pymongo.cursor.CursorType.EXHAUST)
            # entries = db[collection].find({"AnnoyIndex": {"$in": annoy_top_results}, 'categories': category},
            #                               {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1}).hint([('AnnoyIndex', 1)])
        except Exception as e:
            print e
            entries = db[collection].find({"AnnoyIndex": {"$in": annoy_top_results}, 'categories': category},
                                          {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
        # t2 = time()
        # print t2-t1
        # print "query by annoyIndex took {0} secs".format(time()-start)
    farthest_nearest = 1
    nearest_n = []
    # tt = 0
    i = 0
    # for i, entry in enumerate(entries):
    for entry in entries:
        # i += 1
        # t1 = time()
        # tt += t1-t2
        ent = entry['fingerprint']
        if not ent:
            continue
        if isinstance(ent, list):
            logging.warning("Old fp of type 'list' found at collection {0}, category {1}".format(collection, category))
            # t2 = time()
            # tdif = t2 - t1
            # tt += tdif
            continue
        d = distance(category, fp, ent, collection)
        if not d:
            # t2 = time()
            # tdif = t2 - t1
            # tt += tdif
            continue
        i += 1
        if i < number_of_matches:
            nearest_n.append((entry, d))
            farthest_nearest = 1
        else:
            if i == number_of_matches:
                # sort by distance
                nearest_n.sort(key=lambda tup: tup[1])
                # last item in the list (index -1, go python!)
                farthest_nearest = nearest_n[-1][1]

            # Loop through remaining entries, if one of them is better, insert it in the correct location and remove last item
            if d < farthest_nearest:
                insert_at = number_of_matches - 2
                # FAILS! WHEN len(nearest_n) < 100
                while d < nearest_n[insert_at][1]:
                    insert_at -= 1
                    if insert_at == -1:
                        break
                nearest_n.insert(insert_at + 1, (entry, d))
                nearest_n.pop()
                farthest_nearest = nearest_n[-1][1]
        # t2 = time()
        # tdif = t2-t1
        # tt+=tdif
    # print tt

    # print "sorting entries took {0} secs".format(time()-start)
    # t3 = time()
    [result[0].pop('fingerprint') for result in nearest_n]
    [result[0].pop('_id') for result in nearest_n]
    nearest_n = [result[0] for result in nearest_n]
    # t4 = time()
    # print t4-t3
    return nearest_n


def l2_distance(v1, v2):
    v1 = np.array(v1) if isinstance(v1, list) else v1
    v2 = np.array(v2) if isinstance(v2, list) else v2
    if not v1.size == v2.size:
        return None
    return np.linalg.norm(v1 - v2)
