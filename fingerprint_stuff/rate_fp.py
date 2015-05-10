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
# TODO fix trendibb_editor, only first image is shown correctly
# TODO combine check fingerprint and check_fp_except_diag
# from joblib import Parallel, delayed
# NOTE - cross-compare not yet implementing weights, fp_function,distance_function,distance_power
import multiprocessing
import datetime
import json
import cv2
import constants
import random
import math
import resource
import os
import inspect
import sys
import matplotlib

matplotlib.use('Agg')  # prevents problems generating plots on server where no display is defined
import matplotlib.pyplot as plt

# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# $ use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "subfolder")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

# import default
# import find_similar_mongo
import sys
import pymongo
import numpy as np
import cProfile
import StringIO
import pstats
import logging
import argparse

import Utils
import NNSearch
import fingerprint_core as fp_core


Reserve_cpus = constants.Reserve_cpus
fingerprint_length = constants.fingerprint_length
min_images_per_doc = constants.min_images_per_doc
max_images_per_doc = constants.max_images_per_doc
max_items = constants.max_items

BLUE = [255, 0, 0]  # rectangle color
RED = [0, 0, 255]  # PR BG
GREEN = [0, 255, 0]  # PR FG
BLACK = [0, 0, 0]  # sure BG
WHITE = [255, 255, 255]  # sure FG
# def tear_down(self):
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
    unweighted_avg = np.sum(confusion_vector) / n_elements
    print('unweighted distance average:' + str(unweighted_avg))

    report['average_weighted'] = round(weighted_average, 5)
    report['error_cumulative'] = round(cumulative_error, 5)
    report['average_unweighted'] = round(unweighted_avg, 5)
    # print('report:' + str(report))
    return (report)


def mytrace(matrix):
    sum = 0
    for i in range(0, matrix.shape[0]):
        sum = sum + matrix[i, i]
    return (sum)


def save_full_report(report, name=None):
    # print('reporting...' + str(report))
    if name == None:
        name = 'longfp_report.' + datetime.datetime.now().strftime("%Y-%m-%d.%H%M.txt")
    name = os.path.join('./fp_ratings', name + '_long.txt')
    dir = os.path.dirname(name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    print('writing to ' + name)
    print('full report - ' + str(report))

    try:
        f = open(name, 'a')  # ha!! mode 'w+' .... overwrites the file!!!
    except IOError:
        print('cannot open fp_ratings.txt')
    else:
        json.dump(report, f, indent=4, sort_keys=True, separators=(',', ':'))
        f.close()


def save_short_report(report, name=None):
    if name == None or name == '':
        name = 'shortfp_report.' + datetime.datetime.now().strftime("%Y-%m-%d.%H%M.txt")
    name = os.path.join('./fp_ratings', name + '_short.txt')
    dir = os.path.dirname(name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    print('writing to ' + name)
    short_report = {}
    if 'goodness' in report:
        short_report['goodness'] = report['goodness']
        short_report['goodness_error'] = report['goodness_error']
    short_report['chi'] = report['chi']
    short_report['self_var'] = report['self_var']
    short_report['cross_var'] = report['cross_var']

    rep = report['self_report']
    short_report1 = {}
    short_report1['fingerprint_function'] = rep['fingerprint_function']
    short_report1['distance_function'] = rep['distance_function']
    short_report1['distance_power'] = rep['distance_power']
    short_report1['timestamp'] = rep['timestamp']
    short_report1['average_weighted'] = rep['average_weighted']
    short_report1['average_unweighted'] = rep['average_unweighted']
    short_report1['error_cumulative'] = rep['error_cumulative']
    short_report1['n_groups'] = rep['n_groups']
    short_report1['tot_images'] = rep['tot_images']
    short_report['self_report'] = short_report1

    rep = report['cross_report']
    short_report1 = {}
    short_report1['fingerprint_function'] = rep['fingerprint_function']
    short_report1['distance_function'] = rep['distance_function']
    short_report1['distance_power'] = rep['distance_power']
    short_report1['timestamp'] = rep['timestamp']
    short_report1['average_weighted'] = rep['average_unweighted']
    short_report1['average_unweighted'] = rep['average_unweighted']
    short_report1['error_cumulative'] = rep['error_cumulative']
    short_report1['n_groups'] = rep['n_groups']
    short_report1['tot_images'] = rep['tot_images']
    short_report['cross_report'] = short_report1

    print('short reporting - ' + str(short_report))
    try:
        f = open(name, 'a')  # ha!! mode 'w+' .... overwrites the file!!!
    except IOError:
        print('cannot open fp_ratings.txt')
    else:
        json.dump(short_report, f, indent=4, sort_keys=True, separators=(',', ':'))
        f.close()


def display_two_histograms(same_distances, different_distances, name=None):
    max1 = max(same_distances)
    max2 = max(different_distances)
    print('maxb1' + str(max1) + str(max2))
    maxboth = max(max1, max2)
    print('maxboth' + str(maxboth))
    bins = np.linspace(0, maxboth, 50)
    width = 0.3

    hist, bins = np.histogram(same_distances, bins=20)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.clf()
    plt.bar(center, hist, align='center', width=width, color='b')
    hist2, bins = np.histogram(different_distances, bins=20)
    neg_hist2 = []
    for val in hist2:
        neg_hist2.append(-val)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, neg_hist2, align='center', width=width, color='g')
    plt.show()

    # fig, ax = plt.subplots()
    #    rects1 = ax.bar(bins, same_distances, width, color='g')  #, yerr=menStd)
    #    rects2 = ax.bar(bins, neg_diff, width, color='b')  #, yerr=menStd)

    #    ax.set_ylabel('Scores')
    # ax.set_title('Scores by group and gender' )
    #    ax.set_xticks(ind+width)
    #    ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )

    #    ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )


    #
    #    plt.hist(same_distances, bins, alpha=0.5, label='sameItem', color='r')
    #    plt.hist(neg_diff, bins, alpha=0.5, label='differentItem', color='b', )
    plt.legend(loc='upper right')
    if name == None or name == '':
        name = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    name = os.path.join('./fp_ratings', 'histograms_' + name + '.jpg')
    dir = os.path.dirname(name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    print('writing histogram to ' + name)
    plt.savefig(name)

    use_visual_output = True
    if use_visual_output:
        # plt.show(block=False)
        plt.show()


def calc_full_variance(distance_arrays):
    tot_dists = []
    for distances in distance_arrays:
        for val in distances:
            tot_dists.append(val)
    var = np.std(tot_dists)
    return var

def display_tons_of_histograms(same_distances_arrays, different_distances_arrays, name=None):
    max1 = 0
    totsame = []  # np.ndarray.flatten(same_distances_arrays)
    totdiff = []  #np.ndarray.flatten(-different_distances_arrays)
    for same_distances in same_distances_arrays:
        max1l = max(same_distances)
        max1 = max(max1, max1l)
        for val in same_distances:
            totsame.append(val)
    max2 = 0
    for different_distances in different_distances_arrays:
        max2l = max(different_distances)
        max2 = max(max2, max2l)
        for val in different_distances:
            totdiff.append(val)
    plt.clf()
    maxboth = max(max1, max2)
    bins = np.linspace(0, maxboth, 50)

    # for same_distances in same_distances_arrays:
    #       plt.hist(same_distances, bins, alpha=0.5, label='sameItem')


    # for same_distances in same_distances_arrays:
    #        hist, bins2 = np.histogram(same_distances, bins=20)
    #        plt.bar(center, hist, align='center', width=width, color='b')
    #    for different_distances in different_distances_arrays:
    #        hist2, bins2 = np.histogram(different_distances, bins=20)
    #        neg_hist2 = []
    #        for val in hist2:
    #            neg_hist2.append(-val)
    #        plt.bar(center, neg_hist2, align='center', width=width, color='g')


    hist, bins = np.histogram(totsame, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color='g')
    plt.show()

    hist2, bins = np.histogram(totdiff, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    neg_hist2 = []
    for val in hist2:
        neg_hist2.append(-val)
    plt.bar(center, neg_hist2, align='center', width=width, color='r')
    plt.show()

    plt.legend(loc='upper right')
    if name == None or name == '':
        name = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    name = os.path.join('./fp_ratings', 'allhistograms_' + name + '.jpg')
    dir = os.path.dirname(name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    print('writing histogram to ' + name)
    plt.savefig(name)

    use_visual_output = False
    if use_visual_output:
        plt.show(block=False)


def get_docs(n_items=max_items):
    db = pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()
    assert (training_collection_cursor)  # make sure training collection exists
    doc = next(training_collection_cursor, None)
    i = 0
    tot_images = 0
    tot_answers = []
    report = {'n_groups': 0, 'n_images': []}
    print('looking for docs')
    while doc is not None and i < n_items:
        images = doc['images']
        id = doc['_id']
        print('have:' + str(i) + ' good docs so far and ' + str(tot_images) + ' images, checking id:' + str(id))
        if images is not None:
            n_images = len(images)
            n_good = Utils.count_human_bbs_in_doc(images, skip_if_marked_to_skip=True)
            if n_good >= min_images_per_doc and n_good <= max_images_per_doc:
                tot_images = tot_images + n_good
                i = i + 1
                print('got ' + str(n_good) + ' bounded images, ' + str(min_images_per_doc) + '<=n_images<=' + str(
                    max_images_per_doc) + ' , ' + str(
                    n_images) + ' images tot in doc #' + str(i) + ' id:' + str(id))
                tot_answers.append(get_images_from_doc(images))
                report['n_images'].append(n_good)
            else:
                print('not enough bounded boxes (only ' + str(n_good) + ' found, of ' + str(
                    min_images_per_doc) + ' required, ' + str(n_images) + ' images tot)          ', end='\r', sep='')
        doc = next(training_collection_cursor, None)
    report['n_groups'] = i
    report['tot_images'] = tot_images
    report['images_per_group'] = round(tot_images / i, 3)
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
        img_arr = Utils.get_cv2_img_array(dict['url'], convert_url_to_local_filename=True, download=True)
        if img_arr is None:
            return False
        # print('human bb ok:'+str(dict['human_bb']))
        else:
            return True


def show_fps_and_images(fp1, img1, fp2, img2, fig=None):
    extras_length = constants.extras_length
    histograms_length = constants.histograms_length
    if fig:
        plt.close(fig)
    plt.close('all')

    fig, ax = plt.subplots()

    fig.add_subplot(2, 2, 1)
    plt.imshow(img1)

    fig.add_subplot(2, 2, 2)
    plt.imshow(img1)

    # fingerprint 1 bargraph
    fig.add_subplot(2, 2, 3)
    ind = np.arange(fingerprint_length)  # the x locations for the groups
    width = 0.35
    fig.add_subplot(2, 2, 3)
    energy_maxindex = extras_length
    hue_maxindex = energy_maxindex + histograms_length
    sat_maxindex = hue_maxindex + histograms_length
    rects1 = ax.bar(ind[0:energy_maxindex], fp1[0:energy_maxindex], width, color='r')  # , yerr=menStd)
    rects2 = ax.bar(ind[energy_maxindex + 1: hue_maxindex], fp1[energy_maxindex + 1: hue_maxindex], width,
                    color='g')  # , yerr=menStd)
    rects3 = ax.bar(ind[hue_maxindex + 1: sat_maxindex], fp1[hue_maxindex + 1: sat_maxindex], width,
                    color='b')  # , yerr=menStd)
    # add some text for labels, title and axes tisatcks
    ax.set_ylabel('y')
    ax.set_title('fingerprint')
    ax.set_xticks(ind + width)
    # ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )
    # ax.legend( (rects1[0]), ('Men', 'Women') )

    # fingerprint 2 bargraph
    fig.add_subplot(2, 2, 4)
    rects1 = ax.bar(ind[0:energy_maxindex], fp1[0:energy_maxindex], width, color='r')  # , yerr=menStd)
    rects2 = ax.bar(ind[energy_maxindex + 1: hue_maxindex], fp1[energy_maxindex + 1: hue_maxindex], width,
                    color='g')  # , yerr=menStd)
    rects3 = ax.bar(ind[hue_maxindex + 1: sat_maxindex], fp1[hue_maxindex + 1: sat_maxindex], width,
                    color='b')  # , yerr=menStd)
    ax.set_ylabel('y')
    ax.set_title('fingerprint')
    ax.set_xticks(ind + width)
    #

    plt.show(block=False)

    return (fig)


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


def normalize_matrix(matrix):
    # the matrix should be square and is only populated in top triangle , including the diagonal
    # so the number of elements is 1+2+...+N  for an  NxN array, which comes to N*(N+1)/2
    n_elements = float(matrix.shape[0] * matrix.shape[0] + matrix.shape[0]) / 2.0
    sum = np.sum(matrix)
    avg = sum / n_elements
    normalized_matrix = np.divide(matrix, avg)
    return (normalized_matrix)


# maybe delete
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


# maybe delete
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


######## in use
def compare_fingerprints(image_array1, image_array2, fingerprint_function=fp_core.fp,
                         weights=np.ones(fingerprint_length), distance_function=NNSearch.distance_1_k,
                         distance_power=0.5, **fingerprint_arguments):
    global visual_output1
    global visual_output2
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
        img_arr1 = Utils.get_cv2_img_array(url1, convert_url_to_local_filename=True, download=True)
        if Utils.is_valid_image(img_arr1):
            i = i + 1
            # print('comparing image ' + str(i) + ' to other group')
            # background_removal.standard_resize(image, 400)
            mask = Utils.bb_to_mask(bb1, img_arr1)
            # fp1 = fp_core.gc_and_fp(img_arr1, bb1, weights,**fingerprint_arguments)
            try:
                if bb1[2] == 0 or bb1[3] == 0:
                    print('aaaagggghh!! this is a zero-area bb in bb1!!! how did that happen??!?!? bb:' + str(bb1))
                if img_arr1.shape[0] == 0 or img_arr1.shape[1] == 0:
                    print('aaaagggghh!! this is a zero-area image1 !!! how did that happen??!?!? shape:' + str(
                        img_arr1.shape))
                if img_arr1.shape[0] == bb1[3] or img_arr1.shape[1] == bb1[2]:
                    print('bb and img have same shape, bb:' + str(bb1) + ' im:' + str(img_arr1.shape))
                # print('bb:' + str(bb1) + ' im:' + str(img_arr1.shape))
                fp1 = fingerprint_function(img_arr1, bounding_box=bb1, weights=weights, **fingerprint_arguments)
            except:
                print('something bad happened, bb1=' + str(bb1) + ' and imsize1=' + str(img_arr1.shape))
                fp1 = np.ones(fingerprint_length)  # this is arbitrary but lets keep going instead of crashing
            #		print('fp1:'+str(fp1))
            j = 0
            if visual_output1:
                cv2.rectangle(img_arr1, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), color=GREEN, thickness=2)
                cv2.imshow('im1', img_arr1)
                k = cv2.waitKey(50) & 0xFF
                fig = fp_core.show_fp(fp1, **fingerprint_arguments)
                # to parallelize
                #[sqrt(i ** 2) for i in range(10)]
                #Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
            for entry2 in image_array2:
                # print('image 2:'+str(entry2))
                bb2 = entry2['human_bb']
                url2 = entry2['url']
                img_arr2 = Utils.get_cv2_img_array(url2, convert_url_to_local_filename=True, download=True)
                if Utils.is_valid_image(img_arr2):
                    j = j + 1
                    if visual_output2:
                        cv2.rectangle(img_arr2, (bb2[0], bb2[1]), (bb2[0] + bb2[2], bb2[1] + bb2[3]), color=BLUE,
                                      thickness=2)
                        cv2.imshow('im2', img_arr2)
                        k = cv2.waitKey(50) & 0xFF
                        # pdb.set_trace()
                    mask = Utils.bb_to_mask(bb2, img_arr2)
                    try:
                        if bb2[2] == 0 or bb2[3] == 0:
                            print('aaaagggghh!! this is a zero-area bb in bb2!!! how did that happen??!?!? bb:' + str(
                                bb2))
                        if img_arr2.shape[0] == 0 or img_arr2.shape[1] == 0:
                            print('aaaagggghh!! this is a zero-area image2 !!! how did that happen??!?!? area:' + str(
                                img_arr2.shape))
                        if img_arr2.shape[0] == bb2[3] or img_arr2.shape[1] == bb2[2]:
                            print('bb and img have same shape, bb:' + str(bb2) + ' im:' + str(img_arr2.shape))
                        #print('bb:' + str(bb2) + ' im:' + str(img_arr2.shape))
                        fp2 = fingerprint_function(img_arr2, bounding_box=bb2, weights=weights, **fingerprint_arguments)
                    except:
                        print('something bad happened, bb2=' + str(bb2) + ' and imsize2=' + str(img_arr2.shape))
                        fp2 = np.ones(fingerprint_length)  # this is arbitrary but lets keep going instead of crashing
                    # fp2 = fp_core.gc_and_fp(img_arr2, bb2, weights)
                    #print('fp2:'+str(fp2))
                    dist = distance_function(fp1, fp2, k=distance_power)
                    # print('comparing image ' + str(i) + ' to ' + str(j) + ' gave distance:' + str(
                    # dist) + ' totdist:' + str(tot_dist) + '             ', end='\r', sep='')
                    distance_array.append(dist)
                    tot_dist = tot_dist + dist
                    n = n + 1
                # sys.stdout.write(str(n) + '.')
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
    return (avg_dist, distances_stdev, distances_np_array)


##in use
def compare_fingerprints_except_diagonal(image_array1, image_array2, fingerprint_function=fp_core.fp,
                                         weights=np.ones(fingerprint_length), distance_function=NNSearch.distance_1_k,
                                         distance_power=0.5, **fingerprint_arguments):
    global visual_output1
    global visual_output2
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
        img_arr1 = Utils.get_cv2_img_array(url1, convert_url_to_local_filename=True, download=True)
        if Utils.is_valid_image(img_arr1):
            i = i + 1
            try:
                if bb1[2] == 0 or bb1[3] == 0:
                    print('aaaagggghh!! this is a zero-area bb in bb1!!! how did that happen??!?!? bb:' + str(bb1))
                if img_arr1.shape[0] == 0 or img_arr1.shape[1] == 0:
                    print('aaaagggghh!! this is a zero-area image1 !!! how did that happen??!?!? shape:' + str(
                        img_arr1.shape))
                if img_arr1.shape[0] == bb1[3] or img_arr1.shape[1] == bb1[2]:
                    print('bb and img have same shape, bb:' + str(bb1) + ' im:' + str(img_arr1.shape))
                #print('bb:' + str(bb1) + ' im:' + str(img_arr1.shape))
                fp1 = fingerprint_function(img_arr1, bounding_box=bb1, weights=weights, **fingerprint_arguments)
            except:
                print('something bad happened, bb1=' + str(bb1) + ' and imsize1=' + str(img_arr1.shape))
                fp1 = np.ones(fingerprint_length)  # this is arbitrary but lets keep going instead of crashing
            #		print('fp1:'+str(fp1))
            j = 0
            if visual_output1:
                cv2.rectangle(img_arr1, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), color=GREEN, thickness=2)
                cv2.imshow('im1', img_arr1)
                k = cv2.waitKey(50) & 0xFF
                fig = fp_core.show_fp(fp1, **fingerprint_arguments)
            for entry2 in image_array2:
                #			print('image 2:'+str(entry2))
                bb2 = entry2['human_bb']
                url2 = entry2['url']
                img_arr2 = Utils.get_cv2_img_array(url2, convert_url_to_local_filename=True, download=True)
                if Utils.is_valid_image(img_arr2):
                    j = j + 1
                    if visual_output2:
                        cv2.rectangle(img_arr2, (bb2[0], bb2[1]), (bb2[0] + bb2[2], bb2[1] + bb2[3]), color=BLUE,
                                      thickness=2)
                        cv2.imshow('im2', img_arr2)
                        k = cv2.waitKey(50) & 0xFF
                        #pdb.set_trace()
                    mask = Utils.bb_to_mask(bb2, img_arr2)
                    try:
                        if bb2[2] == 0 or bb2[3] == 0:
                            print('aaaagggghh!! this is a zero-area bb in bb2!!! how did that happen??!?!? bb:' + str(
                                bb2))
                        if img_arr2.shape[0] == 0 or img_arr2.shape[1] == 0:
                            print('aaaagggghh!! this is a zero-area image2 !!! how did that happen??!?!? area:' + str(
                                img_arr2.shape))
                        if img_arr2.shape[0] == bb2[3] or img_arr2.shape[1] == bb2[2]:
                            print('bb and img have same shape, bb:' + str(bb2) + ' im:' + str(img_arr2.shape))
                        #print('bb:' + str(bb2) + ' im:' + str(img_arr2.shape))
                        fp2 = fingerprint_function(img_arr2, bounding_box=bb2, weights=weights, **fingerprint_arguments)
                    except:
                        print('something bad happened, bb2=' + str(bb2) + ' and imsize2=' + str(img_arr2.shape))
                        fp2 = np.ones(fingerprint_length)  # this is arbitrary but lets keep going instead of crashing
                    dist = distance_function(fp1, fp2, k=distance_power)
                    # print('comparing image ' + str(i) + ' to ' + str(j) + ' gave distance:' + str(
                    # dist) + ' totdist:' + str(tot_dist) + '             ', end='\r', sep='')
                    if i != j:  #dont record comparison of image to itself
                        distance_array.append(dist)
                        tot_dist = tot_dist + dist
                        n = n + 1
                        sys.stdout.write(str(n) + '.')

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
    return (avg_dist, distances_stdev, distances_np_array)


#########33
# start code in use
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
        # print('set1:' + str(image_sets[i]) + ', set2:' + str(image_sets[j]))
        answers.append([image_sets[i], image_sets[j]])
    return answers


def partial_cross_compare_wrapper((image_sets, fingerprint_function, weights,
                                  distance_function, distance_power, fingerprint_arguments)):
    # print ('module name:'+str( __name__))
    # if hasattr(os, 'getppid'):  # only available on Unix
    # print ('parent process:'+str( os.getppid()))
    # print ('process id:'+str( os.getpid()))

    # print('imset:' + str(image_set))
    # print('fp_func:' + str(fingerprint_function))
    #   print('weights:' + str(weights))
    #   print('d_func:' + str(distance_function))
    # print('d_pow:' + str(distance_power))
    #  print('fp_args:' + str(fingerprint_arguments))

    image_set1 = image_sets[0]
    image_set2 = image_sets[1]
    print('imset1 has ' + str(len(image_set1)) + ' images, imset2 has ' + str(len(image_set2)) + ' images')
    proc_name = multiprocessing.current_process().name
    # print('proc_name:' + str(proc_name))
    # print('im1' + str(image_set1))
    #    print('im2' + str(image_set2))
    avg_dist, stdev, all_distances = compare_fingerprints(image_set1, image_set2, fingerprint_function,
                                                          weights, distance_function,
                                                          distance_power, **fingerprint_arguments)
    confusion_matrix = avg_dist
    stdev_matrix = stdev
    return ([confusion_matrix, stdev_matrix, all_distances])


def calculate_partial_cross_confusion_vector(image_sets, fingerprint_function=fp_core.fp,
                                             weights=np.ones(fingerprint_length),
                                             distance_function=NNSearch.distance_1_k, distance_power=0.5, report=None,
                                             comparisons_to_make=None, parallelize=True, **fingerprint_arguments):
    # print('s.fp_func:' + str(fingerprint_function))
    # print('s.weights:' + str(weights))
    # print('s.distance_function:' + str(distance_function))
    # print('s.distance_power:' + str(distance_power))
    print('cpccv Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if report is None:
        report = {}
    confusion_vector = np.zeros((len(image_sets)))
    stdev_vector = np.zeros((len(image_sets)))
    all_distances = []

    if comparisons_to_make is None:
        comparisons_to_make = make_cross_comparison_sets(image_sets)
    # comparisons_dict = {'comparisons_to_make':comparisons_to_make,'image_sets':image_sets}
    # attempt to parallelize
    if parallelize:
        results = []
        # n_cpus = cpu_count.available_cpu_count() - Reserve_cpus
        n_cpus = multiprocessing.cpu_count() - Reserve_cpus or 1
        # n_cpus = 2
        print('attempting to use ' + str(n_cpus) + ' cpus')
        p = multiprocessing.Pool(processes=n_cpus)
        print('done calculating self comparisons')
        tupled_arguments = []
        for image_set in comparisons_to_make:
            tupled_arguments.append((image_set, fingerprint_function, weights,
                                     distance_function, distance_power, fingerprint_arguments))
        answers = p.map(partial_cross_compare_wrapper, tupled_arguments)
        # answers = p.map(partial_cross_compare_wrapper, comparisons_to_make)
        # p.join()
        #        p.close()
        confusion_vector = [a[0] for a in answers]
        stdev_vector = [a[1] for a in answers]
        all_distances = [a[2] for a in answers]
    else:
        i = 0
        for i in range(0, len(image_sets)):
            imset1 = comparisons_to_make[i][0]
            imset2 = comparisons_to_make[i][1]
            # print('comparing group ' + str(imset1) + ' to group ' + str(imset2))
            avg_dist, stdev, all_dists = compare_fingerprints(imset1, imset2,
                                                              fingerprint_function=fingerprint_function,
                                                              weights=weights, distance_function=distance_function,
                                                              distance_power=distance_power, **fingerprint_arguments)
            confusion_vector[i] = round(avg_dist, 3)
            stdev_vector[i] = round(stdev, 3)
            all_distances.append(all_dists)
            i = i + 1

            # print('confusion vector is currently:'+str(confusion_matrix))
            #    normalized_matrix = normalize_matrix(confusion_matrix)
            #    return(normalized_matrix)
    print('conf vector:' + str(confusion_vector))
    print('stdev vector:' + str(stdev_vector))
    # print('alldistances vector:' + str(all_distances))
    report['confusion_vector'] = confusion_vector
    report['stdev_vector'] = stdev_vector
    report['distance_power'] = distance_power
    report['distance_function'] = str(distance_function)
    report['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    report['weights'] = weights
    report = find_stats(confusion_vector, stdev_vector, report)
    report['fingerprint_function'] = str(fingerprint_function)
    report['all_distances'] = all_distances

    # print('s.fp_func:' + str(fingerprint_fu

    #    print('report:' + str(report))
    #    report['comparisons'] = comparisons_to_make
    return (report)


def self_compare_wrapper2(image_set, fingerprint_function=fp_core.fp, weights=np.ones(fingerprint_length),
                          distance_function=NNSearch.distance_1_k, distance_power=0.5, **fingerprint_arguments):
    # print ('module name:'+str( __name__))
    # if hasattr(os, 'getppid'):  # only available on Unix
    # print ('parent process:'+str( os.getppid()))
    # print ('process id:'+str( os.getpid()))
    proc_name = multiprocessing.current_process().name
    print('proc_name:' + str(proc_name))
    avg_dist, stdev, all_distances = compare_fingerprints_except_diagonal(image_set, image_set, fingerprint_function,
                                                                          weights,
                                                                          distance_function, distance_power,
                                                                          **fingerprint_arguments)
    confusion_matrix = avg_dist
    stdev_matrix = stdev
    return ([confusion_matrix, stdev_matrix, all_distances])


def self_compare_wrapper(( image_set, fingerprint_function, weights,
                         distance_function, distance_power, fingerprint_arguments)):
    # print ('module name:'+str( __name__))
    # if hasattr(os, 'getppid'):  # only available on Unix
    # print ('parent process:'+str( os.getppid()))
    # print ('process id:'+str( os.getpid()))
    proc_name = multiprocessing.current_process().name
    print('imset has ' + str(len(image_set)) + ' images')
    print('proc_name:' + str(proc_name))
    #   print('imset:' + str(image_set))
    #  print('fp_func:' + str(fingerprint_function))
    #   print('weights:' + str(weights))
    # print('d_func:' + str(distance_function))
    # print('d_pow:' + str(distance_power))
    # print('fp_args:' + str(fingerprint_arguments))
    avg_dist, stdev, all_distances = compare_fingerprints_except_diagonal(image_set, image_set, fingerprint_function,
                                                                          weights,
                                                                          distance_function, distance_power,
                                                                          **fingerprint_arguments)
    confusion_matrix = avg_dist
    stdev_matrix = stdev
    return ([confusion_matrix, stdev_matrix, all_distances])


def calculate_self_confusion_vector(image_sets, fingerprint_function=fp_core.fp, weights=np.ones(fingerprint_length),
                                    distance_function=NNSearch.distance_1_k, distance_power=0.5, report=None,
                                    parallelize=True, **fingerprint_arguments):
    '''
    compares image set i to image set i
    '''
    global self_report
    print('cscv Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    #   print('s.fp_func:' + str(fingerprint_function))
    # print('s.weights:' + str(weights))
    # print('s.distance_function:' + str(distance_function))
    # print('s.distance_power:' + str(distance_power))
    if report is None:
        report = {}
    confusion_vector = np.zeros((len(image_sets)))
    stdev_vector = np.zeros((len(image_sets)))
    all_distances = []

    # attempt to parallelize
    if parallelize:
        results = []
        # n_cpus = cpu_count.available_cpu_count() - Reserve_cpus
        n_cpus = multiprocessing.cpu_count() - Reserve_cpus or 1
        # n_cpus = 2
        print('attempting to use ' + str(n_cpus) + ' cpus')
        p = multiprocessing.Pool(processes=n_cpus)
        # answer_matrices = p.map_async(compare_wrapper, [image_sets[i] for i in range(0, len(image_sets))])
        print('done calculating self comparisons')
        tupled_arguments = []
        for image_set in image_sets:
            tupled_arguments.append((image_set, fingerprint_function, weights,
                                     distance_function, distance_power, fingerprint_arguments))

        answers = p.map(self_compare_wrapper, tupled_arguments)
        # TODO pass all the arugments thru tothe wrapper!! including **kwargs!!!
        # for i in range(0,len(image_sets)):
        # p.apply_async(compare_wrapper,args=(image_sets[i],))
        # p.close()
        #    p.join()
        # answer_matrices.wait()
        #        print(str(answers))
        confusion_vector = [a[0] for a in answers]
        stdev_vector = [a[1] for a in answers]
        all_distances = [a[2] for a in answers]
    # print('conf vector:' + str(confusion_vector))
    # print('stdev vector:' + str(stdev_vector))
    #        print('orig vector:' + str(answers))
    else:
        for i in range(0, len(image_sets)):
            print('comparing group ' + str(i) + ' to itself')
            print('imageset:' + str(image_sets[i]))
            avg_dist, stdev, all_dists = compare_fingerprints_except_diagonal(image_sets[i], image_sets[i],
                                                                              fingerprint_function=fingerprint_function,
                                                                              weights=weights,
                                                                              distance_function=distance_function,
                                                                              distance_power=distance_power,
                                                                              **fingerprint_arguments)
            confusion_vector[i] = round(avg_dist, 3)
            stdev_vector[i] = round(stdev, 3)
            all_distances[i] = round(all_dists, 3)


            #	print('confusion vector is currently:'+str(confusion_matrix))
            #    normalized_matrix = normalize_matrix(confusion_matrix)
            #    return(normalized_matrix)
    report['confusion_vector'] = confusion_vector
    report['stdev_vector'] = stdev_vector
    report['distance_power'] = distance_power
    report['distance_function'] = str(distance_function)
    report['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    report['weights'] = weights
    report = find_stats(confusion_vector, stdev_vector, report)
    report['fingerprint_function'] = str(fingerprint_function)
    report['all_distances'] = all_distances
    # print('report:' + str(report))
    return (report)


###################
# end code in use
###################3


# maybe delete
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


# in use
# @profile
def analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                        distance_function=NNSearch.distance_1_k,
                        distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                        use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                        filename=None, **fingerprint_arguments):
    global visual_output1
    global visual_output2
    print('hi')
    print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

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

    self_report = calculate_self_confusion_vector(image_sets, fingerprint_function=fingerprint_function,
                                                  weights=weights, distance_function=distance_function,
                                                  distance_power=distance_power, report=self_report,
                                                  **fingerprint_arguments)

    all_self = self_report['all_distances']
    self_var = calc_full_variance(all_self)
    del self_report['all_distances']

    print('self report:' + str(self_report))
    if not isinstance(self_report['confusion_vector'], list):
        self_report['confusion_vector'] = self_report['confusion_vector'].tolist()  # this is required for json dumping
    if not isinstance(self_report['stdev_vector'], list):
        self_report['error_vector'] = self_report['stdev_vector'].tolist()  # this is required for json dumping
        del (self_report['stdev_vector'])
    if not isinstance(self_report['weights'], list):
        self_report['weights'] = self_report['weights'].tolist()

    cross_report = calculate_partial_cross_confusion_vector(image_sets, fingerprint_function=fingerprint_function,
                                                            weights=weights,
                                                            distance_function=distance_function,
                                                            distance_power=distance_power, report=cross_report,
                                                            comparisons_to_make=comparisons_to_make,
                                                            **fingerprint_arguments)

    print('cross report:' + str(cross_report))
    if not isinstance(cross_report['confusion_vector'], list):
        cross_report['confusion_vector'] = cross_report[
            'confusion_vector'].tolist()  # this is required for json dumping
    if not isinstance(cross_report['stdev_vector'], list):
        cross_report['error_vector'] = cross_report['stdev_vector'].tolist()  # this is required for json dumping
        del (cross_report['stdev_vector'])
    if not isinstance(cross_report['weights'], list):
        cross_report['weights'] = cross_report['weights'].tolist()

    all_cross = cross_report['all_distances']
    cross_var = calc_full_variance(all_cross)
    del cross_report['all_distances']

    same_item_average = self_report['average_weighted']
    cross_item_average = cross_report['average_weighted']
    #     print(self_report)
    same_item_error = self_report['error_cumulative']
    cross_item_error = cross_report['error_cumulative']
    numerator = cross_item_average - same_item_average
    denominator = cross_item_average
    try:
        goodness = numerator / denominator
    except ZeroDivisionError:
        print('DENOMINATOR for goodness=0')
        goodness = 0
    #    print('n,d,n_e,d_e' + str(numerator), str(numerator), str(denominator), str(cross_item_error))
    numerator_error = math.sqrt(cross_item_error ** 2 + same_item_error ** 2)
    if numerator == 0 or denominator == 0:
        goodness_error = -1
    else:
        goodness_error = Utils.error_of_fraction(numerator, numerator_error, denominator, cross_item_error)

    chi = numerator / (np.sqrt(self_var ** 2 + cross_var ** 2))
    tot_report = {'self_report': self_report, 'cross_report': cross_report, 'goodness': goodness,
                  'goodness_error': goodness_error, 'chi': chi, 'self_var': self_var, 'cross_var': cross_var}

    save_full_report(tot_report, filename)
    save_short_report(tot_report, filename)

    display_two_histograms(self_report['confusion_vector'], cross_report['confusion_vector'], filename)
    display_tons_of_histograms(all_self, all_cross, filename)
    # print('tot report:' + str(tot_report))
    print('goodness:' + str(goodness) + ' same item average:' + str(same_item_average) + ' cross item averag:' + str(
        cross_item_average))
    return (chi, tot_report)


global visual_output1
global visual_output2

if __name__ == '__main__':
    report = analyze_fingerprint(fingerprint_function=fp_core.fp_with_kwargs, use_visual_output1=True,
                                 histogram_length=30)

    print('hi0')
    print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    parser = argparse.ArgumentParser(description='rate ye olde fingerprinte')
    #   parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    parser.add_argument('--use_visual_output', default=True,
                        help='show output once for each item')
    parser.add_argument('--use_visual_output2', default=False,
                        help='show output for each image')

    parser.add_argument('--fp_function', default=fp_core.regular_fp,
                        help='what fingerprint function to use')
    args = parser.parse_args()
    visual_output1 = args.use_visual_output
    visual_output2 = args.use_visual_output2
    fp_function = args.fp_function
    print('use_visual_output:' + str(visual_output1) + ' visual_output2:' + str(visual_output2))
    print('fp function to use:' + str(fp_function))
    pr = cProfile.Profile()
    pr.enable()
    weights = np.ones(fingerprint_length)
    report = analyze_fingerprint(fingerprint_function=fp_function, weights=weights,
                                 distance_function=NNSearch.distance_1_k,
                                 distance_power=0.5, use_visual_output1=visual_output1,
                                 use_visual_output2=visual_output2)
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())