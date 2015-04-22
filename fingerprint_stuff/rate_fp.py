from __future__ import print_function

__author__ = 'jeremy'
# todo weight averages by number of pics
# DONE add mask instead oF BB
# TODO add fp to image and present in single frame, also dow both images being compared
# compute stdev and add to report
# done: fix ConnectionError: HTTPConnectionPool(host='img.sheinside.com', port=80): Max retries exceeded with url: /images/lookbook/wearing/201428/04181405101082542276157.jpg (Caused by <class 'socket.error'>: [Errno 104] Connection reset by peer)
# TODO make sure fp is correct when image is missing/not available (make sure its not counted)
# TODO maybe add already-used image sets as input to avoid re-searching every time
# TODO load all images for a given set  and keep in memory

# from joblib import Parallel, delayed
# NOTE - cross-compare not yet implementing weights, fp_function,distance_function,distance_power
import multiprocessing
import datetime
import json
import cv2
import constants
import random
import math
from memory_profiler import profile

# realpath() will make your script run, even if you symlink it :)
# cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
#if cmd_folder not in sys.path:
#    sys.path.insert(0, cmd_folder)

# $ use this if you want to include modules from a subfolder
##cmd_subfolder = os.path.realpath(
# os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "subfolder")))
#if cmd_subfolder not in sys.path:
#   sys.path.insert(0, cmd_subfolder)

import fingerprint_core as fp_core

# import default
# import find_similar_mongo
import sys
import pymongo
import Utils
import NNSearch
import numpy as np
import cProfile
import StringIO
import pstats
import logging
import argparse

Reserve_cpus = constants.Reserve_cpus
fingerprint_length = constants.fingerprint_length
min_images_per_doc = constants.min_images_per_doc
max_items = constants.max_items

BLUE = [255, 0, 0]  # rectangle color
RED = [0, 0, 255]  # PR BG
GREEN = [0, 255, 0]  # PR FG
BLACK = [0, 0, 0]  # sure BG
WHITE = [255, 255, 255]  # sure FG
#    def tear_down(self):
#        shutil.rmtree(self.temp_dir)

#this is for the training collection, where there's a set of images from different angles in each record
def find_stats(confusion_vector, stdev_vector, report):
    weighted_average = 0
    tot_images = 0
    cumulative_error = 0
    for j in range(0, len(confusion_vector)):
        weighted_average = weighted_average + report['n_images'][j] * confusion_vector[j]
        tot_images = tot_images + report['n_images'][j]
        cumulative_error = cumulative_error + (report['n_images'][j] * stdev_vector[j]) * (
            report['n_images'][j] * stdev_vector[j])  # error adds in quadrature
        # print('error element:' + str((report['n_images'][j] * stdev_vector[j]) * (report['n_images'][j] * stdev_vector[j])))
    weighted_average = weighted_average / tot_images
    cumulative_error = np.sqrt(cumulative_error) / tot_images
    # print('confusion vector:' + str(confusion_vector))
    # print('stdev vector:' + str(stdev_vector))
    print('weighted_average:' + str(weighted_average))
    print('cumulative error:' + str(cumulative_error))
    n_elements = len(confusion_vector)
    same_item_avg = np.sum(confusion_vector) / n_elements
    print('unweighted same item average:' + str(same_item_avg))

    report['average_weighted'] = weighted_average
    report['error_cumulative'] = cumulative_error
    report['average_unweighted'] = same_item_avg
    # print('report:' + str(report))
    return (report)


def mytrace(matrix):
    sum = 0
    for i in range(0, matrix.shape[0]):
        sum = sum + matrix[i, i]
    return (sum)


def save_full_report(report):
    name = 'fp_report.' + datetime.datetime.now().strftime("%Y-%m-%d-%H.%M")
    print(name)
    try:
        f = open('fp_ratings' + str(name) + '.txt', 'a')  # ha!! mode 'w+' .... overwrites the file!!!
    except IOError:
        print('cannot open fp_ratings.txt')
    else:
        print('reporting...' + str(report))
        json.dump(report, f, indent=4, sort_keys=True, separators=(',', ':'))
        f.close()


def save_short_report(report):
    name = 'shortfp_report.' + datetime.datetime.now().strftime("%Y-%m-%d-%H.%M")
    print(name)
    short_report = {}
    if 'goodness' in report:
        short_report['goodness'] = report['goodness']
        short_report['goodness_error'] = report['goodness_error']

    rep = report['self_report']
    short_report1 = {}
    short_report1['distance_function'] = rep['distance_function']
    short_report1['timestamp'] = rep['timestamp']
    short_report1['average_weighted'] = rep['weighted_average']
    short_report1['error_cumulative'] = rep['cumulative_error']
    short_report1['average_unweighted'] = rep['same_item_avg']
    self_report['self_report'] = short_report1

    rep = report['cross_report']
    short_report1['distance_function'] = rep['distance_function']
    short_report1['timestamp'] = rep['timestamp']
    short_report1['average_weighted'] = rep['weighted_average']
    short_report1['error_cumulative'] = rep['cumulative_error']
    short_report1['average_unweighted'] = rep['same_item_avg']
    self_report['cross_report'] = short_report1

    try:
        f = open('fp_ratings' + str(name) + '.txt', 'a')  # ha!! mode 'w+' .... overwrites the file!!!
    except IOError:
        print('cannot open fp_ratings.txt')
    else:
        print('reporting...' + str(report))
        json.dump(short_report, f, indent=4, sort_keys=True, separators=(',', ':'))
        f.close()


def get_docs(n_items=max_items):
    db = pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()
    assert (training_collection_cursor)  # make sure training collection exists
    doc = next(training_collection_cursor, None)
    i = 0
    tot_answers = []
    report = {'n_groups': 0, 'n_images': []}
    while doc is not None and i < n_items:
        # print('doc:'+str(doc))
        images = doc['images']
        if images is not None:
            n_images = len(images)
            n_good = Utils.count_human_bbs_in_doc(images, skip_if_marked_to_skip=True)
            if n_good >= min_images_per_doc:
                i = i + 1
                print('got ' + str(n_good) + ' bounded images, ' + str(min_images_per_doc) + ' required, ' + str(
                    n_images) + ' images tot        ')
                tot_answers.append(get_images_from_doc(images))
                report['n_images'].append(n_good)
            else:
                print('not enough bounded boxes (only ' + str(n_good) + ' found, of ' + str(
                    min_images_per_doc) + ' required, ' + str(n_images) + ' images tot)          ', end='\r', sep='')
        doc = next(training_collection_cursor, None)
    report['n_groups'] = i
    return (report, tot_answers)


def get_images_from_doc(images):
    '''
    return the good (bounded) images from an images doc
    '''
    pruned_images = []
    for img in images:
        if Utils.good_bb(img, skip_if_marked_to_skip=True) and good_img(img):
            pruned_images.append(img)
            # print('pruned images:')
            # nice_print(pruned_images)
    return (pruned_images)


def good_img(dict):
    '''
    make sure dict has good image - url is in dict['url']
    :param dict:
    :return:True for good image, False for no image
    '''
    if not 'url' in dict:
        return False
    elif dict['url'] is None:
        return False
    else:
        img_arr = Utils.get_cv2_img_array(dict['url'], try_url_locally=True, download=True)
        if img_arr is None:
            return False
        # print('human bb ok:'+str(dict['human_bb']))
        else:
            return True


def nice_print(images):
    i = 1
    for img in images:
        print('img ' + str(i) + ':' + str(img))
        i = i + 1

def lookfor_image_group(queryobject, string):
    n = 1
    urlN = None  #if nothing eventually is found None is returned for url
    answer_url_list = []
    bb = None
    while (1):
        theBB = None
        strN = string + str(n)  #this is to build strings like 'Main Image URL angle 5' or 'Style Gallery Image 7'
        #        bbN = strN+' bb' #this builds strings like 'Main Image URL angle 5 bb' or 'Style Gallery Image 7 bb'
        print('looking for string:' + str(strN))
        #	logging.debug('looking for string:'+str(strN)+' and bb '+str(bbN))
        if strN in queryobject:
            urlN = queryobject[strN]
            if not 'human_bb' in queryobject:  # got a pic without a bb
                got_unbounded_image = True
                print('image from string:' + strN + ' :is not bounded!!')
            elif queryobject['human_bb'] is None:
                got_unbounded_image = True
                print('image from string:' + strN + ' :is not bounded!!')
            else:
                got_unbounded_image = False
                print('image from string:' + strN + ' :is bounded :(')
                theBB = queryobject['human_bb']
            current_answer = {'url': urlN, 'bb': theBB}
            answer_url_list.append(current_answer)
            print('current answer:' + str(current_answer))
        else:
            print('didn\'t find expected string in training db')
            break
    return (answer_url_list)
# maybe return(urlN,n) at some point

def lookfor_next_imageset():  # IS THIS EVER USED
    print('path=' + str(sys.path))
    resultDict = {}  #return empty dict if no results found
    prefixes = ['Main Image URL angle ', 'Style Gallery Image ']
    db = pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()  #The db with multiple figs of same item
    assert (training_collection_cursor)  #make sure training collection exists

    tot_answers = []

    doc = next(training_collection_cursor, None)
    while doc is not None:
        print('doc:' + str(doc))
        tot_answers = []
        for prefix in prefixes:
            answers = lookfor_image_group(doc, prefix)
            if answers is not None:
                tot_answers.append(answers)
        print('result:' + str(tot_answers))
    return tot_answers

def check_img_array(image_array):
    if image_array is not None and isinstance(image_array, np.ndarray) and isinstance(image_array[0][0], np.ndarray):
        return True
    else:
        return False


def normalize_matrix(matrix):
    # the matrix should be square and is only populated in top triangle , including the diagonal
    # so the number of elements is 1+2+...+N  for an  NxN array, which comes to N*(N+1)/2
    n_elements = float(matrix.shape[0] * matrix.shape[0] + matrix.shape[0]) / 2.0
    sum = np.sum(matrix)
    avg = sum / n_elements
    normalized_matrix = np.divide(matrix, avg)
    return (normalized_matrix)


def cross_compare(image_sets):
    '''
    compares image set i to image set j (including j=i)
    '''
    confusion_matrix = np.zeros((len(image_sets), len(image_sets)))
    print('confusion matrix size:' + str(len(image_sets)) + ' square')
    for i in range(0, len(image_sets)):
        for j in range(i, len(image_sets)):
            print('comparing group ' + str(i) + ' to group ' + str(j))
            print('group 1:' + str(image_sets[i]))
            print('group 2:' + str(image_sets[j]))
            if (i == j):
                avg_dist = compare_fingerprints_except_diagonal(image_sets[i], image_sets[j])
            else:
                avg_dist = compare_fingerprints(image_sets[i], image_sets[j])
            confusion_matrix[i, j] = avg_dist
            print('confusion matrix is currently:' + str(confusion_matrix))
            # normalized_matrix = normalize_matrix(confusion_matrix)
            # return(normalized_matrix)
    return (confusion_matrix)


def calculate_cross_confusion_matrix():
    global cross_report
    cross_report = {'n_groups': 0, 'n_items': [], 'confusion_matrix': []}
    min_images_per_doc = 5
    db = pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()  # The db with multiple figs of same item
    assert (training_collection_cursor)  # make sure training collection exists
    doc = next(training_collection_cursor, None)
    i = 0
    tot_answers = []
    while doc is not None and i < max_items:  # just take 1st N for testing
        # print('doc:'+str(doc))
        images = doc['images']
        n_images = len(images)
        n_good = Utils.count_human_bbs_in_doc(images)
        if n_good > min_images_per_doc:
            i = i + 1
            print('got ' + str(n_good) + ' bounded images, ' + str(min_images_per_doc) + ' required, ' + str(
                n_images) + ' images tot                         ')
            tot_answers.append(get_images_from_doc(images))
            cross_report['n_items'].append(n_good)
        else:
            print('not enough bounded boxes (only ' + str(n_good) + ' found, of ' + str(
                min_images_per_doc) + ' required, ' + str(n_images) + ' images tot                        ', end='\r',
                  sep='')
        doc = next(training_collection_cursor, None)
    print('tot number of groups:' + str(i) + '=' + str(len(tot_answers)))
    confusion_matrix = cross_compare(tot_answers)
    print('confusion matrix:' + str(confusion_matrix))
    cross_report['confusion_matrix'] = confusion_matrix.tolist()  # this is required for json dumping
    # cross_report['fingerprint_function']='fp'
    cross_report['distance_function'] = 'NNSearch.distance_1_k(fp1, fp2,power=1.5)'
    cross_report['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d-%H.%M")
    return (confusion_matrix)


########
def compare_fingerprints(image_array1, image_array2, fingerprint_function=fp_core.fp,
                         weights=np.ones(fingerprint_length), distance_function=NNSearch.distance_1_k,
                         distance_power=0.5):
    tot_dist = 0
    n = 0
    i = 0
    j = 0
    distance_array = []
    print(
        'comparing image group of size ' + str(len(image_array1)) + ' to other group of size ' + str(len(image_array2)))
    for entry1 in image_array1:
        # print('image 1:'+str(entry1))
        bb1 = entry1['human_bb']
        url1 = entry1['url']
        img_arr1 = Utils.get_cv2_img_array(url1, try_url_locally=True, download=True)
        if img_arr1 is not None:
            i = i + 1
            # print('comparing image ' + str(i) + ' to other group')
            # background_removal.standard_resize(image, 400)
            mask = Utils.bb_to_mask(bb1, img_arr1)
            fp1 = fingerprint_function(img_arr1, mask=mask, weights=weights)
            #		print('fp1:'+str(fp1))
            j = 0
            if visual_output1:
                cv2.rectangle(img_arr1, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), color=GREEN, thickness=2)
                cv2.imshow('im1', img_arr1)
                k = cv2.waitKey(50) & 0xFF
                fig = fp_core.show_fp(fp1)
                # to parallelize
                #[sqrt(i ** 2) for i in range(10)]
                #Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
            for entry2 in image_array2:
                # print('image 2:'+str(entry2))
                bb2 = entry2['human_bb']
                url2 = entry2['url']
                img_arr2 = Utils.get_cv2_img_array(url2, try_url_locally=True, download=True)
                if img_arr2 is not None:
                    j = j + 1
                    if visual_output2:
                        cv2.rectangle(img_arr2, (bb2[0], bb2[1]), (bb2[0] + bb2[2], bb2[1] + bb2[3]), color=BLUE,
                                      thickness=2)
                        cv2.imshow('im2', img_arr2)
                        k = cv2.waitKey(50) & 0xFF
                        # pdb.set_trace()
                    mask = Utils.bb_to_mask(bb2, img_arr2)
                    fp2 = fingerprint_function(img_arr1, mask=mask, weights=weights)
                    #print('fp2:'+str(fp2))
                    dist = distance_function(fp1, fp2, k=distance_power)
                    # print('comparing image ' + str(i) + ' to ' + str(j) + ' gave distance:' + str(
                    # dist) + ' totdist:' + str(tot_dist) + '             ', end='\r', sep='')
                    distance_array.append(dist)
                    tot_dist = tot_dist + dist
                    n = n + 1
                else:
                    print('bad img array 2')
                    logging.debug('bad image array 2 in rate_fingerprint.py:compare_fignreprints_ecept_diagonal')
        else:
            print('bad img array 1')
            logging.debug('bad image array 1 in rate_fingerprint.py:compare_fignreprints_ecept_diagonal')
    n_diagonal_elements = i
    avg_dist = float(tot_dist) / float(n)
    distances_np_array = np.array(distance_array)
    distances_stdev = np.std(distances_np_array)
    distances_mean = np.mean(distances_np_array)
    print(
        'average distance:' + str(distances_mean) + '=' + str(avg_dist) + ',stdev' + str(distances_stdev) + ',n=' + str(
            n) + ',tot=' + str(tot_dist) + ' diag elements:' + str(i))
    # print('average distance numpy:'+str(distances_mean)+',stdev'+str(distances_stdev))
    return (avg_dist, distances_stdev)

def compare_fingerprints_except_diagonal(image_array1, image_array2, fingerprint_function=fp_core.fp,
                                         weights=np.ones(fingerprint_length), distance_function=NNSearch.distance_1_k,
                                         distance_power=0.5):
    tot_dist = 0
    n = 0
    i = 0
    j = 0
    distance_array = []
    print(
        'comparing image group of size ' + str(len(image_array1)) + ' to same group of size ' + str(len(image_array2)))
    for entry1 in image_array1:
        #	print('image 1:'+str(entry1))
        bb1 = entry1['human_bb']
        url1 = entry1['url']
        img_arr1 = Utils.get_cv2_img_array(url1, try_url_locally=True, download=True)
        if img_arr1 is not None:
            i = i + 1
#            print('comparing image ' + str(i) + ' to rest of same group')
            #background_removal.standard_resize(image, 400)
            mask = Utils.bb_to_mask(bb1, img_arr1)
            fp1 = fingerprint_function(img_arr1, mask=mask, weights=weights)
            #		print('fp1:'+str(fp1))
            j = 0
            if visual_output1:
                cv2.rectangle(img_arr1, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), color=GREEN, thickness=2)
                cv2.imshow('im1', img_arr1)
                k = cv2.waitKey(50) & 0xFF
                fig = fp_core.show_fp(fp1)
            for entry2 in image_array2:
                #			print('image 2:'+str(entry2))
                bb2 = entry2['human_bb']
                url2 = entry2['url']
                img_arr2 = Utils.get_cv2_img_array(url2, try_url_locally=True, download=True)
                if img_arr2 is not None:
                    j = j + 1
                    if visual_output2:
                        cv2.rectangle(img_arr2, (bb2[0], bb2[1]), (bb2[0] + bb2[2], bb2[1] + bb2[3]), color=BLUE,
                                      thickness=2)
                        cv2.imshow('im2', img_arr2)
                        k = cv2.waitKey(50) & 0xFF
                        #pdb.set_trace()
                    mask = Utils.bb_to_mask(bb2, img_arr2)
                    fp2 = fingerprint_function(img_arr2, mask=mask, weights=weights)  # bounding_box=bb2
                    #print('fp2:'+str(fp2))
                    dist = distance_function(fp1, fp2, k=distance_power)
                    # print('comparing image ' + str(i) + ' to ' + str(j) + ' gave distance:' + str(
                    # dist) + ' totdist:' + str(tot_dist) + '             ', end='\r', sep='')
                    if i != j:  #dont record comparison of image to itself
                        distance_array.append(dist)
                        tot_dist = tot_dist + dist
                        n = n + 1
                else:
                    print('bad img array 2')
                    logging.debug('bad image array 2 in rate_fingerprint.py:compare_fingreprints_ecept_diagonal')
        else:
            print('bad img array 1')
            logging.debug('bad image array 1 in rate_fingerprint.py:compare_fingreprints_ecept_diagonal')
    n_diagonal_elements = i
    avg_dist = float(tot_dist) / float(n)
    distances_np_array = np.array(distance_array)
    distances_stdev = np.std(distances_np_array)
    distances_mean = np.mean(distances_np_array)
    print(
        'average distance:' + str(distances_mean) + '=' + str(avg_dist) + ',stdev' + str(distances_stdev) + ',n=' + str(
            n) + ',tot=' + str(tot_dist) + ' diag elements:' + str(i))
    #    print('average distance numpy:'+str(distances_mean)+',stdev'+str(distances_stdev))
    return (avg_dist, distances_stdev)


#########33

##########
def make_cross_comparison_sets(image_sets):
    '''
    This is for cross comparison with n image sets instead of the full possible n(n-1)/2 sets,
    for n image sets.  For each image set, a single comparison set is chosen.
    Just make sure the one chosen for comparison is not the same set.
    '''
    answers = []
    for i in range(0, len(image_sets)):
        j = random.randint(0, len(image_sets) - 1)
        while j == i:
            # print('had to rechoose (i='+str(i)+'='+str(j)+'=j)')
            j = random.randint(0, len(image_sets) - 1)
        print('set1:' + str(i) + ', set2:' + str(j))
        answers.append([image_sets[i], image_sets[j]])
    return answers

def partial_cross_compare_wrapper(image_sets, fingerprint_function=fp_core.fp, weights=np.ones(fingerprint_length),
                                  distance_function=NNSearch.distance_1_k, distance_power=0.5):
    # print ('module name:'+str( __name__))
    # if hasattr(os, 'getppid'):  # only available on Unix
    # print ('parent process:'+str( os.getppid()))
    # print ('process id:'+str( os.getpid()))
    image_set1 = image_sets[0]
    image_set2 = image_sets[1]
    # print('im1' + str(image_set1))
    #  print('im2' + str(image_set2))
    avg_dist, stdev = compare_fingerprints(image_set1, image_set2, fingerprint_function=fingerprint_function,
                                           weights=weights, distance_function=distance_function,
                                           distance_power=distance_power)
    confusion_matrix = avg_dist
    stdev_matrix = stdev
    return ([confusion_matrix, stdev_matrix])


def calculate_partial_cross_confusion_vector(image_sets, fingerprint_function=fp_core.fp,
                                             weights=np.ones(fingerprint_length),
                                             distance_function=NNSearch.distance_1_k, distance_power=0.5, report=None,
                                             comparisons_to_make=None):
    # print('s.fp_func:' + str(fingerprint_function))
    # print('s.weights:' + str(weights))
    # print('s.distance_function:' + str(distance_function))
    # print('s.distance_power:' + str(distance_power))

    confusion_vector = np.zeros((len(image_sets)))
    stdev_vector = np.zeros((len(image_sets)))

    if comparisons_to_make is None:
        comparisons_to_make = make_cross_comparison_sets(image_sets)
    # comparisons_dict = {'comparisons_to_make':comparisons_to_make,'image_sets':image_sets}
    # attempt to parallelize
    parallelize = True
    if parallelize:
        results = []
        # n_cpus = cpu_count.available_cpu_count() - Reserve_cpus
        n_cpus = multiprocessing.cpu_count() - Reserve_cpus or 1
        # n_cpus = 2
        print('attempting to use ' + str(n_cpus) + ' cpus')
        p = multiprocessing.Pool(processes=n_cpus)
        answers = p.map(partial_cross_compare_wrapper, comparisons_to_make)
        confusion_vector = [a[0] for a in answers]
        stdev_vector = [a[1] for a in answers]
    else:
        i = 0
        for imset1, imset2 in image_sets:
            # print('comparing group ' + str(imset1) + ' to group ' + str(imset2))
            avg_dist, stdev = compare_fingerprints(imset1, imset2,
                                                   fingerprint_function=fingerprint_function,
                                                   weights=weights, distance_function=distance_function,
                                                   distance_power=distance_power)
            confusion_vector[i] = avg_dist
            stdev_vector[i] = stdev
            i = i + 1

            # print('confusion vector is currently:'+str(confusion_matrix))
            #    normalized_matrix = normalize_matrix(confusion_matrix)
            #    return(normalized_matrix)
    print('conf vector:' + str(confusion_vector))
    print('stdev vector:' + str(stdev_vector))
    report['confusion_vector'] = confusion_vector
    report['stdev_vector'] = stdev_vector
    report['distance_power'] = distance_power
    report['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    report['weights'] = weights
    report = find_stats(confusion_vector, stdev_vector, report)
#    print('report:' + str(report))
#    report['comparisons'] = comparisons_to_make
    return (report)


def self_compare_wrapper(image_set, fingerprint_function=fp_core.fp, weights=np.ones(fingerprint_length),
                         distance_function=NNSearch.distance_1_k, distance_power=0.5):
    # print ('module name:'+str( __name__))
    # if hasattr(os, 'getppid'):  # only available on Unix
    # print ('parent process:'+str( os.getppid()))
    # print ('process id:'+str( os.getpid()))
    avg_dist, stdev = compare_fingerprints_except_diagonal(image_set, image_set, fingerprint_function, weights,
                                                           distance_function, distance_power)
    confusion_matrix = avg_dist
    stdev_matrix = stdev
    return ([confusion_matrix, stdev_matrix])


def calculate_self_confusion_vector(image_sets, fingerprint_function=fp_core.fp, weights=np.ones(fingerprint_length),
                                    distance_function=NNSearch.distance_1_k, distance_power=0.5, report=None):
    '''
    compares image set i to image set i
    '''
    global self_report

 #   print('s.fp_func:' + str(fingerprint_function))
    # print('s.weights:' + str(weights))
    # print('s.distance_function:' + str(distance_function))
    # print('s.distance_power:' + str(distance_power))
    confusion_vector = np.zeros((len(image_sets)))
    stdev_vector = np.zeros((len(image_sets)))

    # attempt to parallelize
    parallelize = True
    if parallelize:
        results = []
        # n_cpus = cpu_count.available_cpu_count() - Reserve_cpus
        n_cpus = multiprocessing.cpu_count() - Reserve_cpus or 1
        # n_cpus = 2
        print('attempting to use ' + str(n_cpus) + ' cpus')
        p = multiprocessing.Pool(processes=n_cpus)
        # answer_matrices = p.map_async(compare_wrapper, [image_sets[i] for i in range(0, len(image_sets))])
        answers = p.map(self_compare_wrapper, image_sets)
        # for i in range(0,len(image_sets)):
        # p.apply_async(compare_wrapper,args=(image_sets[i],))
        # p.close()
        # p.join()
        # answer_matrices.wait()
        #        print(str(answers))
        confusion_vector = [a[0] for a in answers]
        stdev_vector = [a[1] for a in answers]
        print('conf vector:' + str(confusion_vector))
        print('stdev vector:' + str(stdev_vector))
        print('orig vector:' + str(answers))
    else:
        for i in range(0, len(image_sets)):
            print('comparing group ' + str(i) + ' to itself (doc index=' + str(self_report['doc_indices'][i]) + ')')
            avg_dist, stdev = compare_fingerprints_except_diagonal(image_sets[i], image_sets[i],
                                                                   fingerprint_function=fingerprint_function,
                                                                   weights=weights, distance_function=distance_function,
                                                                   distance_power=distance_power)
            confusion_vector[i] = avg_dist
            stdev_vector[i] = stdev


            #	print('confusion vector is currently:'+str(confusion_matrix))
            #    normalized_matrix = normalize_matrix(confusion_matrix)
            #    return(normalized_matrix)
    report['confusion_vector'] = confusion_vector
    report['stdev_vector'] = stdev_vector
    report['distance_power'] = distance_power
    report['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    report['weights'] = weights
    report = find_stats(confusion_vector, stdev_vector, report)
  #  print('report:' + str(report))
    return (report)


#########33


def cross_rate_fingerprint():
    global cross_report
    cross_report = {}
    confusion_matrix = calculate_cross_confusion_matrix()
    print('confusion matrix final:' + str(confusion_matrix))
    normalized_confusion_matrix = normalize_matrix(confusion_matrix)
    #number of diagonal and offdiagonal elements for NxN array  is N and (N*N-1)/2
    n_diagonal_elements = normalized_confusion_matrix.shape[0]
    n_offdiagonal_elements = float(normalized_confusion_matrix.shape[0] * normalized_confusion_matrix.shape[0]
                                   - normalized_confusion_matrix.shape[0]) / 2.0
    same_item_avg = mytrace(normalized_confusion_matrix) / n_diagonal_elements
    different_item_avg = (float(np.sum(normalized_confusion_matrix)) - float(
        mytrace(normalized_confusion_matrix))) / n_offdiagonal_elements
    goodness = different_item_avg - same_item_avg
    print('same item average:' + str(same_item_avg) + ' different item average:' + str(
        different_item_avg) + ' difference:' + str(goodness))
    cross_report['same_item_average'] = same_item_avg
    cross_report['different_item_average'] = different_item_avg
    cross_report['goodness'] = goodness
    save_full_report(cross_report)
    return (different_item_avg)


@profile
def analyze_fingerprint(fingerprint_function=fp_core.fp, weights=np.ones(fingerprint_length),
                        distance_function=NNSearch.distance_1_k,
                        distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                        use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None):
    global visual_output1
    global visual_output2

    visual_output1 = use_visual_output1
    visual_output2 = use_visual_output2

    if image_sets is None:
        # get the initial info , also we can use same image set for cross comparisons below
        self_report, image_sets = get_docs(n_docs)
    else:
        # i am doing this retarded thing since if i call the parameter 'self_report' in the argument list
        # i get a warning about 'shadowing'.
        self_report = self_reporting

    cross_report = dict(self_report)

    calculate_self_confusion_vector(image_sets, fingerprint_function=fingerprint_function,
                                    weights=weights, distance_function=distance_function,
                                    distance_power=distance_power, report=self_report)
    print('self report:' + str(self_report))
    if not isinstance(self_report['confusion_vector'], list):
        self_report['confusion_vector'] = self_report['confusion_vector'].tolist()  # this is required for json dumping
    if not isinstance(self_report['stdev_vector'], list):
        self_report['error_vector'] = self_report['stdev_vector'].tolist()  # this is required for json dumping
    if not isinstance(self_report['weights'], list):
        self_report['weights'] = self_report['weights'].tolist()

    calculate_partial_cross_confusion_vector(image_sets, fingerprint_function=fingerprint_function, weights=weights,
                                             distance_function=distance_function,
                                             distance_power=distance_power, report=cross_report,
                                             comparisons_to_make=comparisons_to_make)

    print('cross report:' + str(cross_report))
    if not isinstance(cross_report['confusion_vector'], list):
        cross_report['confusion_vector'] = cross_report[
            'confusion_vector'].tolist()  # this is required for json dumping
    if not isinstance(cross_report['stdev_vector'], list):
        cross_report['error_vector'] = cross_report['stdev_vector'].tolist()  # this is required for json dumping
    if not isinstance(cross_report['weights'], list):
        cross_report['weights'] = cross_report['weights'].tolist()

    same_item_average = self_report['average_weighted']
    cross_item_average = cross_report['average_weighted']
#     print(self_report)
    same_item_error = self_report['error_cumulative']
    cross_item_error = cross_report['error_cumulative']
    numerator = cross_item_average - same_item_average
    denominator = cross_item_average
    goodness = numerator / denominator
#    print('n,d,n_e,d_e' + str(numerator), str(numerator), str(denominator), str(cross_item_error))
    numerator_error = math.sqrt(cross_item_error ** 2 + same_item_error ** 2)
    goodness_error = Utils.error_of_fraction(numerator, numerator_error, denominator, cross_item_error)
    tot_report = {'self_report': self_report, 'cross_report': cross_report, 'goodness': goodness,
                  'goodness_error': goodness_error}
    save_full_report(tot_report)
    save_short_report(tot_report)
    # print('tot report:' + str(tot_report))
    print('goodness:' + str(goodness) + ' same item average:' + str(same_item_average) + ' cross item averag:' + str(
        cross_item_average))
    return (goodness, tot_report)


visual_output1 = True
visual_output2 = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rate ye olde fingerprinte')
    #   parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    parser.add_argument('--use_visual_output', default=False,
                        help='show output once for each item')
    parser.add_argument('--use_visual_output2', default=False,
                        help='show output for each image')
    args = parser.parse_args()
    visual_output1 = args.use_visual_output
    visual_output2 = args.use_visual_output2
    print('use_visual_output:' + str(visual_output1) + ' visual_output2:' + str(visual_output2))

    pr = cProfile.Profile()
    pr.enable()
    weights = np.ones(fingerprint_length)
    report = analyze_fingerprint(fingerprint_function=fp_core.fp, weights=weights,
                                 distance_function=NNSearch.distance_1_k,
                                 distance_power=0.5)
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


