__author__ = 'yonatan'

from time import sleep
import annoy
import datetime
from .. import Utils
import logging
import bson
import numpy as np
import cv2
from ..paperdoll import paperdoll_parse_enqueue
from .. import paperdolls
from .. import background_removal

from .. import constants
from testSpecio import spatiogram_fingerprints_distance, fingerprint_3D_spatiogram
db = constants.db

def distance_Bhattacharyya(fp1, fp2):
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
    weights = [0.05, 0.5, 0.225, 0.225]
    hist_length = [180, 255, 255]
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


def find_n_nearest_neighbors(target_dict, entries, number_of_matches, distance_function=None, key='fingerprint'):
    distance_function = distance_function or distance_Bhattacharyya
    # list of tuples with (entry,distance). Initialize with first n distance values
    nearest_n = []
    farthest_nearest = 1
    for i, entry in enumerate(entries):
        if i < number_of_matches:
            ent = entry[key]
            tar = target_dict[key]
            d = distance_function(ent, tar)
            nearest_n.append((entry, d))
        else:
            if i == number_of_matches:
                # sort by distance
                nearest_n.sort(key=lambda tup: tup[1])
                # last item in the list (index -1, go python!)
                farthest_nearest = nearest_n[-1][1]

            # Loop through remaining entries, if one of them is better, insert it in the correct location and remove last item
            ent = entry[key]
            tar = target_dict[key]
            d = distance_function(ent, tar)
            if d < farthest_nearest:
                insert_at = number_of_matches-2
                while d < nearest_n[insert_at][1]:
                    insert_at -= 1
                    if insert_at == -1:
                        break
                nearest_n.insert(insert_at + 1, (entry, d))
                nearest_n.pop()
                farthest_nearest = nearest_n[-1][1]
    [result[0].pop(key) for result in nearest_n]
    [result[0].pop('_id') for result in nearest_n]
    nearest_n = [result[0] for result in nearest_n]
    return nearest_n


def build_forests(tree_count=250):
    path = '/home/developer/spaciotesting/'
    db.testSpacio.update_many({}, {'$unset': {"AnnoyIndex": 1}})
    dis_func = 'angular'

    """
    forest for histogram
    """
    t = annoy.AnnoyIndex(696, dis_func)
    items = db.testSpacio.find({})
    for x, item in enumerate(items):
        v = item['fingerprint']
        t.add_item(x, v)
        db.testSpacio.update_one({'_id': item['_id']}, {'$set': {"AnnoyIndex.fp": x}})
    t.build(tree_count)
    t.save(path + 'histo250.ann')

    """
    forest for spacio
    """
    t = annoy.AnnoyIndex(2304, dis_func)
    items = db.testSpacio.find({})
    for x,item in enumerate(items):
        v= item['sp']
        vector = []
        for i in range(len(v)):
            vector+=v[i]
        t.add_item(x,vector)
        db.testSpacio.update_one({'_id':item['_id']},{'$set':{"AnnoyIndex.sp":x}})
    t.build(tree_count)
    t.save(path+'spacio250.ann')


def annoy_search(method,topN,signature):
    path = '/home/developer/spaciotesting/'
    if method is 'fp':
        name = path+'histo250.ann'
        atoms = 696
    else:
        name = path+'spacio250.ann'
        atoms = 2304
    t = annoy.AnnoyIndex(atoms, 'angular')
    t.load(name)
    result = t.get_nns_by_vector(signature,topN)
    return result

def findTop():
    """
    find top results for the fanni image base
    both for histogram and spaciogram
    """
    topN = 1000
    col = db.fanni
    col.update_one({}, {'$unset': {'topresults.sp': 1}})
    items = col.find()
    for z,item in enumerate(items):
        fp = item['fingerprint']
        annResults = annoy_search('fp', topN, fp)
        batch = db.testSpacio.find({"AnnoyIndex.fp": {"$in": annResults}}, {"fingerprint": 1,'images.XLarge':1})
        topFP = find_n_nearest_neighbors(item,batch,16,distance_Bhattacharyya,'fingerprint')

        sp = item['sp']
        vector = []
        for i in range(len(sp)):
            vector += sp[i]
        annResults = annoy_search('sp', topN, vector)
        batch = db.testSpacio.find({"AnnoyIndex.sp": {"$in": annResults}}, {"sp": 1,'images.XLarge':1})
        topSP = find_n_nearest_neighbors(item, batch, 16, spatiogram_fingerprints_distance, 'sp')

        # tmp = {'img_url': item['img_url'],
        #        'fp': topFP,
        #        'sp': topSP}
        col.update_one({'_id':item['_id']},{'$set':{'topresults.sp':topSP}})
        print (z)

def findTopFP2SP():
    """
    search annoy using fp
    fine tune using sp
    """
    topN = 2000
    col = db.fanni
    col.update_one({}, {'$unset': {'topresults.sp': 1}})
    items = col.find()
    for z,item in enumerate(items):
        fp = item['fingerprint']
        annResults = annoy_search('fp', topN, fp)
        print (len(annResults))
        batch = db.testSpacio.find({"AnnoyIndex": {"$in": annResults}}, {"sp": 1,'images.XLarge':1})
        print (batch.count())
        topSP = find_n_nearest_neighbors(item, batch, 16, spatiogram_fingerprints_distance, 'sp')
        print(len(topSP))
        col.update_one({'_id':item['_id']},{'$set':{'topresults.sp':topSP}})
        print (z)

def findTopSP41Only(id,z):
    """
    find top results using SP for one item only
    """
    topN = 1000
    col = db.fanni
    item = col.find_one_and_update({'_id':id}, {'$unset': {'topresults.sp': 1}})
    sp = item['sp']
    vector = []
    for i in range(len(sp)):
        vector += sp[i]
    annResults = annoy_search('sp', topN, vector)
    batch = db.testSpacio.find({"AnnoyIndex.sp": {"$in": annResults}}, {"sp": 1, 'images.XLarge': 1})
    topSP = find_n_nearest_neighbors(item, batch, 16, spatiogram_fingerprints_distance, 'sp')

    col.update_one({'_id': id}, {'$set': {'topresults.sp': topSP}})
    print (z)


def findToprecruit():
    """
    find top results using SP for one item only
    """
    col = db.fanni
    count = col.count()
    items = col.find()
    for z, item in enumerate(items):
        id = item['_id']
        batch = db.recruit_Female.find({'categories':'dress'}, {"fingerprint": 1, 'images.XLarge': 1})
        topSP = find_n_nearest_neighbors(item, batch, 16)

        col.update_one({'_id': id}, {'$set': {'topresults.sp': topSP}})
        print ('%d / %d' %(z, count))


def get_sp(image_url,x,update=False, id=None):
    image = Utils.get_cv2_img_array(image_url)

    paper_job = paperdoll_parse_enqueue.paperdoll_enqueue(image, str(x))
    while not paper_job.is_finished and not paper_job.is_failed:
        sleep(10)
    if paper_job.is_failed:
        return None
    mask, labels = paper_job.result[:2]
    final_mask = paperdolls.after_pd_conclusions(mask, labels)
    for num in np.unique(final_mask):
        # convert numbers to labels
        category = list(labels.keys())[list(labels.values()).index(num)]
        if category == "dress":
            item_mask = 255 * np.array(final_mask == num, dtype=np.uint8)
            # small_image, resize_ratio = background_removal.standard_resize(image, 400)
            # mask2 = background_removal.get_fg_mask(small_image)

            item_bb = paperdolls.bb_from_mask(item_mask)
            after_gc_mask = background_removal.get_fg_mask(image, item_bb)  # (255, 0) mask
            spacio = fingerprint_3D_spatiogram(image, after_gc_mask)
            if update:
                try:
                    db.fanni.update_one({'_id':id},{'$set':{'ppd_mask':after_gc_mask.tolist(), 'sp': spacio}})
                except:
                    db.fanni.delete_one({'_id': id})
                    return False, []
                findTopSP41Only(id,x)
                return False,[]
            else:
                return True, spacio

    return False, []


def workOnfanni():
    items = db.fanni.find({'ppd_mask':{'$exists':0}})
    for x, item in enumerate(items):
        url = item['img_url']
        success, sp = get_sp(url, 123456 + x,update=True, id=item['_id'] )
        if success is True:
            db.fanni.update_one({'_id': item['_id']}, {'$set': {'sp': sp}})

    print (db.fanni.find({'ppd_mask':{'$exists':0}}).count())

def sp4fanni():
    items = db.fanni.find()
    for x, item in enumerate(items):
        try:
            url = item['img_url']
            mask = np.array(item['ppd_mask']).astype('uint8')
            image = Utils.get_cv2_img_array(url)
            spacio = fingerprint_3D_spatiogram(image, mask)

            db.fanni.update_one({'_id': item['_id']}, {'$set': {'sp': spacio}})
            print(x)
        except:
            db.fanni.update_one({'_id': item['_id']}, {'$unset': {'sp': 1}})
