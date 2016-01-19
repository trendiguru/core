__author__ = 'yuli'

import os
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import operator

#from trendi.constants import db
import pymongo
db = pymongo.MongoClient().mydb

from trendi import constants
import trendi.Utils as Utils
import trendi.NNSearch as NNSearch
from trendi import fingerprint_core as fp_core



def sort_dists_by_val(dists, n= 10):
    x = dists
    sorted_x = sorted(x.items(), key=operator.itemgetter(1))
    return sorted_x

def bild_dist_(photo, listing):
    top_dists = {}
    weights = [1, 1, 1, 1]
    fingerprint_function=fp_core.fp
    fingerprint_length = constants.fingerprint_length
    fp_weights = np.ones(fingerprint_length)

    fp1_mask =
    img_arr1 =
    fp1 = np.multiply(fingerprint_function(img_arr1, mask= fp1_mask , **fingerprint_arguments), fp_weights)
    for file in listing:
        fp2_mask =
        img_arr2 =
        fp2 = np.multiply(fingerprint_function(img_arr2, mask= fp2_mask , **fingerprint_arguments), fp_weights)
        dists[file] = NNSearch.distance_Bhattacharyya(fp1, fp2, weights, hist_len)

    top_dists[photo] = sort_dists_by_val(dists, n= 10)
    return top_dists

def match_rank(file, dists):
    photo_id, sep, matched_id = file.split('_bbox')[0].partition('photo_')
    photo_id = photo_id.strip('product_').strip('_') #strip from file product ___
    score = 0
    # position of matched photo : (wheter or not to use dict_len depends if sorted dict is in ascending or descending order)
    key_list = []
    for key in dists.keys():
        matched_key = key.strip('product_').strip('_photo')
        if matched_key == matched_id:
            key_list.append(key)
    # score.append( dict_len - dists.keys().index(str(key)) )   for key in key_list
    return score

def main_func():
    all_ = {}

    path = "/home/omer/new/trendi/fingerprint_stuff/test1/"
    listing = os.listdir(path)
    for file in listing:
        top_dists = build_dist_(file, listing) #numpy array
        all_[photo_id]["top_dists"] = top_dists
        all_[photo_id]["score"] = match_rank(file, top_dists)

    return

if __name__ == '__main__':
    main_func()