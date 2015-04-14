from __future__ import print_function

__author__ = 'jeremy'
# todo weight averages by number of pics
# compute stdev and add to report
# done: fix ConnectionError: HTTPConnectionPool(host='img.sheinside.com', port=80): Max retries exceeded with url: /images/lookbook/wearing/201428/04181405101082542276157.jpg (Caused by <class 'socket.error'>: [Errno 104] Connection reset by peer)
# TODO make sure fp is correct when image is missing/not available (make sure its not counted)

# from joblib import Parallel, delayed
# NOTE - cross-compare not yet implementing weights, fp_function,distance_function,distance_power
from multiprocessing import Pool
import datetime
import json
import fingerprint_core as fp_core
import cv2

import os, sys, inspect
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "subfolder")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import utils.cpu_count as cpu_count



# import default
#import find_similar_mongo
import sys
import pymongo
import Utils
import NNSearch
import numpy as np
import cProfile
import StringIO
import pstats
import logging
import constants
import argparse


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

def lookfor_next_imageset():
    print('path=' + str(sys.path))
    resultDict = {}  #return empty dict if no results found
    prefixes = ['Main Image URL angle ', 'Style Gallery Image ']
    db = pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()  #The db with multiple figs of same item
    assert (training_collection_cursor)  #make sure training collection exists

    doc = next(training_collection_cursor, None)
    while doc is not None:
        print('doc:' + str(doc))
        tot_answers = []
        for prefix in prefixes:
            answers = lookfor_image_group(doc, prefix)
            if answers is not None:
                tot_answers.append[answers]
        print('result:' + str(tot_answers))


def compare_fingerprints(image_array1, image_array2, fingerprint_function=fp_core.fp,
                         weights=np.ones(fingerprint_length), distance_function=NNSearch.distance_1_k,
                         distance_power=1.5):
    # assert(len(image_array1) == len(image_array2)) #maybe not require that these be the same set...
    #    print('fp_func:'+str(fingerprint_function))
    #    print('weights:'+str(weights))
    #    print('distance_function:'+str(distance_function))
    #    print('distance_power:'+str(distance_power))
    good_results = []
    tot_dist = 0
    n = 0
    i = 0
    j = 0
    distance_array = []
    for entry1 in image_array1:
        i = i + 1
        # print('image 1:'+str(entry1))
        bb1 = entry1['human_bb']
        url1 = entry1['url']
        img_arr1 = Utils.get_cv2_img_array(url1, try_url_locally=True, download=True)
        if img_arr1 is not None:
            # background_removal.standard_resize(image, 400)
            fp1 = fingerprint_function(img_arr1, bounding_box=bb1, weights=weights)
            #		print('fp1:'+str(fp1))
            j = 0
            if use_visual_output:
                cv2.rectangle(img_arr1, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), color=GREEN, thickness=2)
                cv2.imshow('im1', img_arr1)
                k = cv2.waitKey(50) & 0xFF
                fig = fp_core.show_fp(fp1)
                # to parallelize
                #[sqrt(i ** 2) for i in range(10)]
                #Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
            for entry2 in image_array2:
                j = j + 1
                # print('image 2:'+str(entry2))
                bb2 = entry2['human_bb']
                url2 = entry2['url']
                img_arr2 = Utils.get_cv2_img_array(url2, try_url_locally=True, download=True)
                if img_arr2 is not None:
                    if use_visual_output2:
                        cv2.rectangle(img_arr2, (bb2[0], bb2[1]), (bb2[0] + bb2[2], bb2[1] + bb2[3]), color=BLUE,
                                      thickness=2)
                        cv2.imshow('im2', img_arr2)
                        k = cv2.waitKey(50) & 0xFF
                        # pdb.set_trace()
                    fp2 = fingerprint_function(img_arr2, bounding_box=bb2, weights=weights)
                    #print('fp2:'+str(fp2))
                    dist = distance_function(fp1, fp2, k=distance_power)
                    print('comparing image ' + str(i) + ' to ' + str(j) + ' gave distance:' + str(
                        dist) + ' totdist:' + str(tot_dist) + '             ', end='\r', sep='')
                    # if i != j:  #dont record comparison of image to itself
                    distance_array.append(dist)
                    tot_dist = tot_dist + dist
                    n = n + 1
                else:
                    print('bad img array 2')
                    logging.debug('bad image array 1 in rate_fingerprint.py:compare_fignreprints_ecept_diagonal')
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
                                         distance_power=1.5):
    #    assert(len(image_array1) == len(image_array2)) #maybe not require that these be the same set...
    #    print('fp_func:'+str(fingerprint_function))
    #    print('weights:'+str(weights))
    #    print('distance_function:'+str(distance_function))
    #    print('distance_power:'+str(distance_power))
    good_results = []
    tot_dist = 0
    n = 0
    i = 0
    j = 0
    distance_array = []
    for entry1 in image_array1:
        i = i + 1
        #	print('image 1:'+str(entry1))
        bb1 = entry1['human_bb']
        url1 = entry1['url']
        img_arr1 = Utils.get_cv2_img_array(url1, try_url_locally=True, download=True)
        if img_arr1 is not None:
            #background_removal.standard_resize(image, 400)
            fp1 = fingerprint_function(img_arr1, bounding_box=bb1, weights=weights)
            #		print('fp1:'+str(fp1))
            j = 0
            if use_visual_output:
                cv2.rectangle(img_arr1, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), color=GREEN, thickness=2)
                cv2.imshow('im1', img_arr1)
                k = cv2.waitKey(50) & 0xFF
                fig = fp_core.show_fp(fp1)
                #to parallelize
                #[sqrt(i ** 2) for i in range(10)]
                #Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
            for entry2 in image_array2:
                j = j + 1
                #			print('image 2:'+str(entry2))
                bb2 = entry2['human_bb']
                url2 = entry2['url']
                img_arr2 = Utils.get_cv2_img_array(url2, try_url_locally=True, download=True)
                if img_arr2 is not None:
                    if use_visual_output2:
                        cv2.rectangle(img_arr2, (bb2[0], bb2[1]), (bb2[0] + bb2[2], bb2[1] + bb2[3]), color=BLUE,
                                      thickness=2)
                        cv2.imshow('im2', img_arr2)
                        k = cv2.waitKey(50) & 0xFF
                        #pdb.set_trace()
                    fp2 = fingerprint_function(img_arr2, bounding_box=bb2, weights=weights)
                    #print('fp2:'+str(fp2))
                    dist = distance_function(fp1, fp2, k=distance_power)
                    print('comparing image ' + str(i) + ' to ' + str(j) + ' gave distance:' + str(
                        dist) + ' totdist:' + str(tot_dist) + '             ', end='\r', sep='')
                    if i != j:  #dont record comparison of image to itself
                        distance_array.append(dist)
                        tot_dist = tot_dist + dist
                        n = n + 1
                else:
                    print('bad img array 2')
                    logging.debug('bad image array 1 in rate_fingerprint.py:compare_fignreprints_ecept_diagonal')
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
    #    print('average distance numpy:'+str(distances_mean)+',stdev'+str(distances_stdev))
    return (avg_dist, distances_stdev)


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
            #    normalized_matrix = normalize_matrix(confusion_matrix)
            #    return(normalized_matrix)
    return (confusion_matrix)


def compare_wrapper(image_set, fingerprint_function=fp_core.fp, weights=np.ones(fingerprint_length),
                    distance_function=NNSearch.distance_1_k, distance_power=0.5):
    avg_dist, stdev = compare_fingerprints_except_diagonal(image_set, image_set, fingerprint_function, weights,
                                                           distance_function, distance_power)
    confusion_matrix = avg_dist
    stdev_matrix = stdev
    return ([confusion_matrix, stdev_matrix])


def self_compare(image_sets, fingerprint_function=fp_core.fp, weights=np.ones(fingerprint_length),
                 distance_function=NNSearch.distance_1_k, distance_power=0.5):
    '''
    compares image set i to image set i
    '''
    global report
    confusion_matrix = np.zeros((len(image_sets)))
    stdev_matrix = np.zeros((len(image_sets)))

    # attempt to parallelize
    parallelize = True
    if parallelize:
        n_cpus = cpu_count.available_cpu_count()
        print('attempting to use ' + str(n_cpus) + ' cpus')
        p = Pool(n_cpus)
        answer_matrices = p.map(compare_wrapper, [image_sets[i] for i in range(0, len(image_sets))])
        confusion_matrix = answer_matrices[0, :]
        stdev_matrix = answer_matrices[1, :]
        print('conf matrix:' + str(confusion_matrix))
        print('stdev matrix:' + str(stdev_matrix))
        print('orig  matrix:' + str(answer_matrices))
    else:
        for i in range(0, len(image_sets)):
            print('comparing group ' + str(i) + ' to itself (doc index=' + str(report['doc_indices'][i]) + ')')
            avg_dist, stdev = compare_fingerprints_except_diagonal(image_sets[i], image_sets[i],
                                                                   fingerprint_function=fingerprint_function,
                                                                   weights=weights, distance_function=distance_function,
                                                                   distance_power=distance_power)
            confusion_matrix[i] = avg_dist
            stdev_matrix[i] = stdev


            #	print('confusion vector is currently:'+str(confusion_matrix))
            #    normalized_matrix = normalize_matrix(confusion_matrix)
            #    return(normalized_matrix)
    return (confusion_matrix, stdev_matrix)


def mytrace(matrix):
    sum = 0
    for i in range(0, matrix.shape[0]):
        sum = sum + matrix[i, i]
    return (sum)

    #    print('confusion vector size:'+str(len(image_sets))+' long')
    for i in range(0, len(image_sets)):
        print('comparing group ' + str(i) + ' to itself')
        #	print('group '+str(i)+':'+str(image_sets[i]))
        avg_dist, stdev = compare_fingerprints_except_diagonal(image_sets[i], image_sets[i],
                                                               fingerprint_function=fingerprint_function,
                                                               weights=weights, distance_function=distance_function,
                                                               distance_power=distance_power)
        confusion_matrix[i] = avg_dist
        stdev_matrix[i] = stdev
    #	print('confusion vector is currently:'+str(confusion_matrix))
    #    normalized_matrix = normalize_matrix(confusion_matrix)
    #    return(normalized_matrix)
    return (confusion_matrix, stdev_matrix)


def mytrace(matrix):
    sum = 0
    for i in range(0, matrix.shape[0]):
        sum = sum + matrix[i, i]
    return (sum)


def calculate_confusion_matrix():
    global report
    report = {'n_groups': 0, 'n_items': [], 'confusion_matrix': []}
    min_images_per_doc = 5
    db = pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()  #The db with multiple figs of same item
    assert (training_collection_cursor)  #make sure training collection exists
    doc = next(training_collection_cursor, None)
    i = 0
    tot_answers = []
    while doc is not None and i < max_items:  #just take 1st N for testing
        #        print('doc:'+str(doc))
        images = doc['images']
        n_images = len(images)
        n_good = Utils.count_human_bbs_in_doc(images)
        if n_good > min_images_per_doc:
            i = i + 1
            print('got ' + str(n_good) + ' bounded images, ' + str(min_images_per_doc) + ' required, ' + str(
                n_images) + ' images tot             ');
            tot_answers.append(get_images_from_doc(images))
            report['n_items'].append(n_good)
        else:
            print('not enough bounded boxes (only ' + str(n_good) + ' found, of ' + str(
                min_images_per_doc) + ' required, ' + str(n_images) + ' images tot                  ', end='\r', sep='')
        doc = next(training_collection_cursor, None)
    print('tot number of groups:' + str(i) + '=' + str(len(tot_answers)))
    confusion_matrix = cross_compare(tot_answers)
    print('confusion matrix:' + str(confusion_matrix))
    report['confusion_matrix'] = confusion_matrix.tolist()  #this is required for json dumping
    #    report['fingerprint_function']='fp'
    report['distance_function'] = 'NNSearch.distance_1_k(fp1, fp2,power=1.5)'
    report['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (confusion_matrix)


def get_images_from_doc(images):
    '''
    return the good (bounded) images from an images doc
    '''
    pruned_images = []
    for img in images:
        if Utils.good_bb(img, skip_if_marked_to_skip=True) and good_img(img):
            pruned_images.append(img)
            #    print('pruned images:')
            #    nice_print(pruned_images)
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
        #	print('human bb ok:'+str(dict['human_bb']))
        else:
            return True


def nice_print(images):
    i = 1
    for img in images:
        print('img ' + str(i) + ':' + str(img))
        i = i + 1


def calculate_self_confusion_vector(fingerprint_function=fp_core.fp, weights=np.ones(fingerprint_length),
                                    distance_function=NNSearch.distance_1_k, distance_power=1.5):
    #don't look at sets with less than this number of images
    global report
    db = pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()
    assert (training_collection_cursor)  #make sure training collection exists
    doc = next(training_collection_cursor, None)
    i = 0
    tot_answers = []
    report = {'n_groups': 0, 'n_images': []}
    while doc is not None and i < max_items:
        #        print('doc:'+str(doc))
        images = doc['images']
        if images is not None:
            n_images = len(images)
            n_good = Utils.count_human_bbs_in_doc(images, skip_if_marked_to_skip=True)
            if n_good > min_images_per_doc:
                i = i + 1
                print('got ' + str(n_good) + ' bounded images, ' + str(min_images_per_doc) + ' required, ' + str(
                    n_images) + ' images tot')
                tot_answers.append(get_images_from_doc(images))
                report['n_images'].append(n_good)
            else:
                print('not enough bounded boxes (only ' + str(n_good) + ' found, of ' + str(
                    min_images_per_doc) + ' required, ' + str(n_images) + ' images tot)', end='\n', sep='')
        doc = next(training_collection_cursor, None)
    confusion_vector, stdev_vector = self_compare(tot_answers, fingerprint_function=fingerprint_function,
                                                  weights=weights, distance_function=distance_function,
                                                  distance_power=distance_power)
    print('tot number of groups:' + str(i) + '=' + str(len(tot_answers)))
    print('confusion vector:' + str(confusion_vector))
    report['n_groups'] = i
    report['confusion_vector'] = confusion_vector.tolist()  #this is required for json dumping
    report['error_vector'] = stdev_vector.tolist()  #this is required for json dumping
    #   report['fingerprint_function']=fingerprint_function
    report['weights'] = weights.tolist()
    #report['distance_function'] = distance_function
    report['distance_power'] = distance_power
    report['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    #save_report(report)
    weighted_average = 0
    tot_images = 0
    cumulative_error = 0
    for j in range(0, i):
        weighted_average = weighted_average + report['n_images'][j] * confusion_vector[j]
        tot_images = tot_images + report['n_images'][j]
        cumulative_error = cumulative_error + (report['n_images'][j] * stdev_vector[j]) * (
            report['n_images'][j] * stdev_vector[j])  #error adds in quadrature
        print('error element:' + str(
            (report['n_images'][j] * stdev_vector[j]) * (report['n_images'][j] * stdev_vector[j])))
    weighted_average = weighted_average / tot_images
    cumulative_error = np.sqrt(cumulative_error) / tot_images
    print('weighted_average:' + str(weighted_average))
    report['average_weighted'] = weighted_average
    print('cumulative error:' + str(cumulative_error))
    report['error_cumulative'] = cumulative_error
    return (confusion_vector)


def save_report(report):
    try:
        f = open('fp_ratings.txt', 'a')  #ha!! mode 'w+' .... overwrites the file!!!
    except IOError:
        print('cannot open fp_ratings.txt')
    else:
        print('reporting...' + str(report))
        json.dump(report, f, indent=4, sort_keys=True, separators=(',', ':'))
        f.close()


def cross_rate_fingerprint():
    global report
    report = {}
    confusion_matrix = calculate_confusion_matrix()
    print('confusion matrix final:' + str(confusion_matrix))
    normalized_confusion_matrix = normalize_matrix(confusion_matrix)
    #number of diagonal and offdiagonal elements for NxN array  is N and (N*N-1)/2
    n_diagonal_elements = normalized_confusion_matrix.shape[0]
    n_offdiagonal_elements = float(
        normalized_confusion_matrix.shape[0] * normalized_confusion_matrix.shape[0] - normalized_confusion_matrix.shape[
            0]) / 2.0
    same_item_avg = mytrace(normalized_confusion_matrix) / n_diagonal_elements
    different_item_avg = (float(np.sum(normalized_confusion_matrix)) - float(
        mytrace(normalized_confusion_matrix))) / n_offdiagonal_elements
    goodness = different_item_avg - same_item_avg
    print('same item average:' + str(same_item_avg) + ' different item average:' + str(
        different_item_avg) + ' difference:' + str(goodness))
    report['same_item_average'] = same_item_avg
    report['different_item_average'] = different_item_avg
    report['goodness'] = goodness
    save_report(report)
    return (goodness)


def self_rate_fingerprint(fingerprint_function=fp_core.fp,
                          weights=np.ones(fingerprint_length), distance_function=NNSearch.distance_1_k,
                          distance_power=1.5,
                          **fingerprint_args):
    print('s.fp_func:' + str(fingerprint_function))
    print('s.weights:' + str(weights))
    print('s.distance_function:' + str(distance_function))
    print('s.distance_power:' + str(distance_power))
    global report
    report = {}
    confusion_vector = calculate_self_confusion_vector(fingerprint_function=fingerprint_function, weights=weights,
                                                       distance_function=distance_function,
                                                       distance_power=distance_power)

    print('confusion vector final:' + str(confusion_vector))
    n_elements = len(confusion_vector)
    same_item_avg = np.sum(confusion_vector) / n_elements
    print('unweighted same item average:' + str(same_item_avg))
    report['average_unweighted'] = same_item_avg
    print('report:' + str(report))
    save_report(report)
    return (same_item_avg)


use_visual_output = False
use_visual_output2 = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rate ye olde fingerprinte')
    #   parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    parser.add_argument('--use_visual_output', default=False,
                        help='show output once for each item')
    parser.add_argument('--use_visual_output2', default=False,
                        help='show output for each image')
    args = parser.parse_args()
    use_visual_output = args.use_visual_output
    use_visual_output2 = args.use_visual_output2
    print('use_visual_output:' + str(use_visual_output) + ' use_visual_output2:' + str(use_visual_output2))

    pr = cProfile.Profile()
    pr.enable()
    weights = np.ones(fingerprint_length)
    self_rate_fingerprint(fingerprint_function=fp_core.fp, weights=weights, distance_function=NNSearch.distance_1_k,
                          distance_power=0.5)
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


