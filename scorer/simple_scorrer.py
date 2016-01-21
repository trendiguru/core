__author__ = 'yuli'

import os, os.path
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import operator
import pickle
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

def build_dists(photo, path, mpath, listing):
    dists = {}
    weights = constants.fingerprint_weights
    hist_len = constants.histograms_length
    fingerprint_function=fp_core.fp
    fingerprint_length = constants.fingerprint_length
    fp_weights = np.ones(fingerprint_length)

    if os.path.exists(mpath+photo+".npy"):
        fp1_mask = np.load(mpath+photo+".npy")
        img_arr1 = Utils.get_cv2_img_array(path+photo)# , convert_url_to_local_filename=True, download=True)
        fp1 = np.multiply(fingerprint_function(img_arr1, mask= fp1_mask ), fp_weights)
        for file in listing:
            if os.path.exists(mpath+file+".npy"):
                fp2_mask = np.load(mpath+file+".npy")
                img_arr2 = Utils.get_cv2_img_array(path+file)
                fp2 = np.multiply(fingerprint_function(img_arr2, mask= fp2_mask ), fp_weights)
                dists[file] = NNSearch.distance_Bhattacharyya(fp1, fp2, weights, hist_len)
    return dists

def match_rank(file, dists):
    photo_id, sep, matched_id = file.split('_bbox')[0].partition('photo_')
    photo_id = photo_id.strip('product_').strip('_') #strip from file product ___
    score = {}
    # position of matched photo : (wheter or not to use dict_len depends if sorted dict is in ascending or descending order)
    key_list = []
    dists_keys = [x[0] for x in dists]
    for key in dists_keys:
        matched_key = key.strip('product_').strip('_photo')
        if matched_key == matched_id:
            key_list.append(key)
    print("len key list:", len(key_list))

    for k in key_list:
        score[k] = [key[0] for key in dists].index(k)
        print("matched_key:",  k)
        print("matched_key score:", score[k])


    #[ score.append([key[0] for key in dists].index(k)) for k in key_list ]
    # score.append( dict_len - dists.keys().index(str(key)) )   for key in key_list
    return score

def main_func():
    all_ = {}

    path = "/home/netanel/meta/dataset/test1/"
    mpath = "/home/netanel/meta/dataset/test1_masks/"

    listing = os.listdir(path)
    for file in listing:
        all_[file] = {}
        distances = build_dists(file, path, mpath, listing) #numpy array
        sorted_dists = sort_dists_by_val(distances, n= 10)
        all_[file]["sorted_dists"] = sorted_dists
        all_[file]["score"] = match_rank(file, sorted_dists) # a dict

    with open('/home/netanel/meta/dataset/test1_results.pickle', 'wb') as f:
        pickle.dump(all_, f)
    return

if __name__ == '__main__':
    main_func()