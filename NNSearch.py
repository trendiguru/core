# TODO - write test functions for main functions here

__author__ = 'jeremy'
import logging
from .db_stuff import db_utils
import numpy as np
import cv2
from db_stuff import fanni
import constants
from rq import Queue
from time import sleep, time
q = Queue('annoy', connection=constants.redis_conn)
db = constants.db
K = constants.K  # .5 is the same as Euclidean
tmp_log= '/home/developer/logs/NN.log'

def distance_1_k(fp1, fp2,mis, take ,k=K):
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

#
# def find_n_nearest_neighbors(target_dict, entries, number_of_matches, fp_weights,
#                              hist_length, fp_key, distance_function=None):
#     distance_function = distance_function or distance_Bhattacharyya
#     # list of tuples with (entry,distance). Initialize with first n distance values
#     nearest_n = []
#     for i, entry in enumerate(entries):
#         if i < number_of_matches:
#             ent = entry[fp_key]
#             tar = target_dict["fingerprint"]
#             d = distance_function(ent, tar, fp_weights, hist_length)
#             nearest_n.append((entry, d))
#             farthest_nearest = 1
#         else:
#             if i == number_of_matches:
#                 # sort by distance
#                 nearest_n.sort(key=lambda tup: tup[1])
#                 # last item in the list (index -1, go python!)
#                 farthest_nearest = nearest_n[-1][1]
#
#             # Loop through remaining entries, if one of them is better, insert it in the correct location and remove last item
#             d = distance_function(entry[fp_key], target_dict["fingerprint"], fp_weights, hist_length)
#             if d < farthest_nearest:
#                 insert_at = number_of_matches-2
#                 while d < nearest_n[insert_at][1]:
#                     insert_at -= 1
#                     if insert_at == -1:
#                         break
#                 nearest_n.insert(insert_at + 1, (entry, d))
#                 nearest_n.pop()
#                 farthest_nearest = nearest_n[-1][1]
#     [result[0].pop('fingerprint') for result in nearest_n]
#     [result[0].pop('_id') for result in nearest_n]
#     nearest_n = [result[0] for result in nearest_n]
#     return nearest_n

def find_n_nearest_neighbors(target_dict, collection, category, number_of_matches, fp_weights,
                                 hist_length, fp_key, distance_function=None):

    msg = 'col: %s, cat: %s' %(collection, category)
    db_utils.log2file(mode='a', log_filename=tmp_log, message=msg)
    distance_function = distance_function or distance_Bhattacharyya
    # list of tuples with (entry,distance). Initialize with first n distance values
    fingerprint = target_dict["fingerprint"]
    entries = db[collection].find({'categories':category},
                                  {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
    t1= time()
    if entries.count() > 2000:
        annoy_job = q.enqueue(fanni.lumberjack, args=(collection,category, fingerprint))
        while not annoy_job.is_finished and not annoy_job.is_failed:
            sleep(0.1)
        if annoy_job.is_failed:
            return []
        t2= time()
        msg= 'annoy->%f' %(t2-t1)
        db_utils.log2file(mode='a',log_filename=tmp_log, message=msg)
        top1000 = annoy_job.result
        entries = db[collection].find({"AnnoyIndex": {"$in": top1000}, 'categories': category},
                                      {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
        t3= time()
        msg= 'query->%f' % (t3 - t2)
        db_utils.log2file(mode='a', log_filename=tmp_log, message=msg)

    t4= time()
    farthest_nearest = 1
    nearest_n = []
    for i, entry in enumerate(entries):
        if i < number_of_matches:
            ent = entry[fp_key]

            d = distance_function(ent, fingerprint, fp_weights, hist_length)
            nearest_n.append((entry, d))
            farthest_nearest = 1
        else:
            if i == number_of_matches:
                # sort by distance
                nearest_n.sort(key=lambda tup: tup[1])
                # last item in the list (index -1, go python!)
                farthest_nearest = nearest_n[-1][1]

            # Loop through remaining entries, if one of them is better, insert it in the correct location and remove last item
            d = distance_function(entry[fp_key], fingerprint, fp_weights, hist_length)
            if d < farthest_nearest:
                insert_at = number_of_matches - 2
                while d < nearest_n[insert_at][1]:
                    insert_at -= 1
                    if insert_at == -1:
                        break
                nearest_n.insert(insert_at + 1, (entry, d))
                nearest_n.pop()
                farthest_nearest = nearest_n[-1][1]
    t5= time()
    msg = 'loop->%.f' %(t5-t4)
    db_utils.log2file(mode='a', log_filename=tmp_log, message=msg)
    [result[0].pop('fingerprint') for result in nearest_n]
    [result[0].pop('_id') for result in nearest_n]
    nearest_n = [result[0] for result in nearest_n]
    return nearest_n
