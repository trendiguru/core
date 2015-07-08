from __future__ import print_function
#!/usr/bin/env python
__author__ = 'jeremy'
# theirs
import cv2
import json
from pylab import *
import os
import numpy as np
import argparse
from time import gmtime, strftime
# import matplotlib as plt

#ours
import background_removal
import Utils
import pylab as pl

# TODO
#TODO allow specific directories, and alow recursive directory traverse for test_classifiers
# TODO print directory name being checked so you know whassup during test
# add raw data to conf matrix graph - number of targets, # detected, # extra
# make output as json
# put precision annotation on graph instead of recall

cascades=[]
classifierNames=[]
imageDirectories=[]
searchStrings=[]
resultsDir=''
trainDir=''

iouThreshold=0.5
use_visual_output=False
#@maxRows=1000  #cut off analysis after this number of rows

BLUE = [255,0,0]        # rectangle color

def show_positives_file(positives_file):
    pass


def detect_no_bb(classifier, img_array, use_visual_output=False):
    bbs_computer = classifier.detectMultiScale(img_array)
    if bbs_computer is () or bbs_computer is None:
        n_bbs_computer=0
        # print('classifier found no match')
        return (0)
        #check why this is ()
    if use_visual_output is True:
        for x,y, width,height in bbs_computer:
            cv2.rectangle(img_array, (x,y), (x+width, y+height), color=BLUE,thickness=1)
        cv2.imshow('input', img_array)
    #    cv2.waitKey(100)
        k = 0xFF & cv2.waitKey(10)
    return (len(bbs_computer))


def calc_precision_recall(tot_targets, tot_found, extras_found, classifier_names, image_dirs):
    print('tot targets:' + str(tot_targets))
    print('tot found:' + str(tot_found))
    answers = []
    for i in range(0, len(classifier_names)):
        classifier_path = classifier_names[i]
        head, classifier_name = os.path.split(classifier_path)

        tot_targets_in_class = 0
        tot_found_for_class = 0
        for j in range(0, len(image_dirs)):
            tot_found_for_class = tot_found_for_class + tot_found[i, j] + extras_found[i, j]
        print('tot found for class ' + str(i) + ' ' + classifier_name + ' :' + str(
            tot_found_for_class) + ' tot targets:' + str(tot_targets_in_class))
        for j in range(0, len(image_dirs)):
            image_path = image_dirs[j]
            head, image_dir = os.path.split(image_path)
            if tot_targets[i, j]:
                recall = tot_found[i, j] / tot_targets[i, j]
            else:
                recall = 0
            if tot_found_for_class:
                precision = tot_found[i, j] / tot_found_for_class
            else:
                precision = 0
            print('found:' + str(tot_found[i, j]) + ' tot_targets:' + str(
                tot_targets[i, j]) + ' tot_found_for_class:' + str(tot_found_for_class))
            pstr = 'precision of classifier ' + classifier_name + ' against ' + image_dir + ' :' + str(precision)
            rstr = 'recall of classifier ' + classifier_name + ' against ' + image_dir + ' :' + str(recall)
            print(pstr)
            print(rstr)
            answers.append(pstr)
            answers.append(rstr)
    return answers


def write_classifier_html(html_name, date_string, results, targets_matrix, matches_matrix, extras_matrix):
    f = open(html_name, 'a')
    # write html file
    f.write('<HTML><HEAD><TITLE>classifier results</TITLE>\n')
    fig_name = 'confusion' + date_string + '.png'
    f.write('<BODY text=#999999 vLink=#555588 aLink=#88ff88 link=#8888ff bgColor=#000000>\n ')
    f.write('</HEAD>\n')
    f.write('<IMG style=\"WIDTH: 600px;\"  src=' + fig_name + '>\n')
    for result in results:
        f.write('<br>' + str(result) + '\n')

    f.write('<br>TARGETS\n')
    f.write(str(targets_matrix))
    f.write('<br>MATCHES\n')
    f.write(str(matches_matrix))
    f.write('<br>EXTRAS\n')
    f.write(str(extras_matrix))

    f.write('</html>\n')
    f.close


def plot_confusion_matrix2(cm, image_dirs, classifier_names, targets_matrix, matches_matrix, extras_matrix,
                           title='Confusion matrix', cmap=plt.cm.GnBu, use_visual_output=False):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    rows, cols = np.shape(cm)
    for i in range(0, rows):
        for j in range(0, cols):
            # if type(cm[0, 0]) is float:
            txt = '{:.2}'.format(cm[i, j])
            plt.text(j - .2, i - .1, txt, fontsize=8)
            txt = '({:.0f},{:.0f},{:.0f})'.format(matches_matrix[i, j], targets_matrix[i, j], extras_matrix[i, j])
            plt.text(j - .2, i + .1, txt, fontsize=8)
            # else:
            # txt = '{:.2}'.format(cm[i, j])
            #            plt.text(j , i + .2, txt, fontsize=8)

    plt.title(title + '\nNmatches/Ntargets\n(Nmatches,Ntargets,Nextras)', fontsize=10)
    plt.colorbar()
    ylabels = classifier_names
    xlabels = image_dirs
    xtick_marks = np.arange(len(xlabels))
    ytick_marks = np.arange(len(ylabels))
    plt.xticks(xtick_marks, xlabels, rotation=90, fontsize=8)
    plt.yticks(ytick_marks, ylabels, fontsize=8)
    # plt.tight_layout()
    plt.ylabel('classifier')
    plt.xlabel('image_dir')
    date_string = strftime("%d%m%Y_%H%M%S", gmtime())
    figName = 'classifier_results/confusion' + date_string + '.png'
    plt.subplots_adjust(left=.1, right=1, bottom=0.2, top=0.9)
    savefig(figName, format="png")
    if use_visual_output:
        plt.show(block=True)
    return date_string


def plot_confusion_matrix(m, image_dirs, classifier_names, use_visual_output=False):
    #conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], [3,31,0,0,0,0,0,0,0,0,0], [0,4,41,0,0,0,0,0,0,0,1], [0,1,0,30,0,6,0,0,0,0,1], [0,0,0,0,38,10,0,0,0,0,0], [0,0,0,3,1,39,0,0,0,0,4], [0,2,2,0,4,1,31,0,0,0,2], [0,1,0,0,0,0,0,36,0,2,0], [0,0,0,0,0,0,1,5,37,5,1], [3,0,0,0,0,0,0,0,0,39,0], [0,0,0,0,0,0,0,0,0,0,38] ]

    ylabels = classifier_names
    xlabels = image_dirs
    # cm = confusion_matrix(y_test, pred, labels)
    # print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(m)
    pl.title('Confusion matrix of classifiers')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + xlabels)
    ax.set_yticklabels([''] + ylabels)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show()

    fig = plt.figure()
    # ax = fig.add_subplot(111)
    max = np.amax(m)
    scaled_matrix = m * 255 / max
    res = plt.imshow(array(m) * 256 / max, cmap=cm.jet, interpolation='nearest', vmin=0, vmax=max)
    plt.xticks(rotation=90)
    plt.xticks(np.arange(0, len(image_dirs)), image_dirs, fontsize=8)  # check if x is classifier or image category
    plt.yticks(np.arange(0, len(classifier_names)), classifier_names,
               fontsize=8)  # check if y is classifiers or categories
    cb = fig.colorbar(res)

    rows, cols = np.shape(m)

    for i in range(0, rows):
        for j in range(0, cols):
            if type(m[0, 0]) is float:
                txt = '{:.2}'.format(m[i, j])
                plt.text(j, i + .2, txt, fontsize=8)
            else:
                txt = '{:.2}'.format(m[i, j])
                plt.text(j, i + .2, txt, fontsize=8)
            # plt.text(j + 1.2, i + .2, str(m[i, j]), fontsize=6)

            if i == j:  # detector x on pics of x
                #                classifier_sum=classifier_sum+float(j)/float(den)
                pass
            else:  # detector x on pics of something else
                #                classifier_sum=classifier_sum-float(j)/float(den)
                pass

    date_string = strftime("%d%m%Y_%H%M%S", gmtime())
    figName = 'classifier_results/confusion' + date_string + '.png'
    plt.subplots_adjust(left=-.2, right=1.2, bottom=-.2, top=1.0). \
        savefig(figName, format="png")
    if use_visual_output:
        plt.show(block=True)
    return (date_string)


def test_classifier(classifier, imagesDir, max_files_to_try=10000, use_visual_output=False):
    '''
    run classifier on all images in dict - assume only one or no target items per image
    :param classifier: the classifier xml
    :param imagesDir: directory containing images
    :return:
    '''
    n_files=0
    i=0
    totTargets=0
    totMatches = 0
    totExtra = 0
    files = Utils.files_in_directory(imagesDir)
    print('testing ' + str(len(files)) + ' files in directory ' + str(imagesDir))
    for file in files:
        img_array = cv2.imread(file)
        if img_array is None:
            print('file:' + file + ' read error')  #
            continue
        img_array, ratio = background_removal.standard_resize(img_array, 400)
        h, w, d = img_array.shape
        n_files = n_files + 1
        if use_visual_output is True:
            cv2.imshow('input', img_array)
            k = 0xFF & cv2.waitKey(10)

        nMatches = detect_no_bb(classifier, img_array, use_visual_output=use_visual_output)
        n_extra = 0
        if nMatches:
            n_extra = nMatches - 1  # any more than 1 match is assumed wrong here
        totTargets = totTargets + 1
        totMatches = totMatches + (nMatches > 0)
        totExtra = totExtra + n_extra
        print('totTargets:' + str(totTargets) + ' nMatches:' + str(nMatches) + ' totMatches:' + str(
            totMatches) + ' nExtra:' + str(n_extra) + ' totextra:' + str(totExtra), end='\r')
        #.filter(db.items.fingerprint.is_(None))
        if n_files==max_files_to_try:
            print('reached max of ' + str(max_files_to_try) + ' files to check')
            break
    return (totTargets, totMatches, totExtra)

###########
# run using classifier we have on images from db
###########
#classifierNames.append("dressClassifier013.xml")
#TODO allow specific directories, and alow recursive directory traverse for test_classifiers


def test_classifiers(classifierDir='../classifiers/', imageDir='images', use_visual_output=False):
    resultsDir='classifier_results'
    results_filename=os.path.join(resultsDir,'classifier_results_'+trainDir+'.txt')
    max_files_to_try = 1000

    #go thru image directories
    imagedirs = Utils.immediate_subdirs(imageDir)
    if len(imagedirs) == 0:
        print('empty image directory:' + str(imagedirs))
        return None
    for subdir in imagedirs:
        searchStrings.append('class:'+subdir)
        print('image directory:'+str(subdir))
    n_categories = len(imagedirs)

    #go thru classifier directories
    classifiers = Utils.files_in_directory(classifierDir)
#	print(' subdirlist'+str(subdirlist))
    if len(classifiers) == 0:
        print('empty classifier directory:' + str(classifierDir))
        exit()
    n_classifiers = len(classifiers)
    classifier_names = ''
    results_list=[]
    results_matrix=np.zeros([n_classifiers,n_categories])
    matches_over_targets_matrix = np.zeros([n_classifiers, n_categories])
    targets_matrix = np.zeros([n_classifiers, n_categories])
    matches_matrix = np.zeros([n_classifiers, n_categories])
    extras_matrix = np.zeros([n_classifiers, n_categories])
    samplesize=[]
    gotSizesFlag=False
    totTargets = 0
    for i in range(0, len(classifiers)):
        classifier = classifiers[i]
        head, tail = os.path.split("/tmp/d/a.dat")
        classifier_names = classifier_names + tail + '_'
        results_row=[]
        results_matrixrow = []
        cascade_classifier = cv2.CascadeClassifier(classifier)
        check_empty = cascade_classifier.empty()
        if check_empty:
            print('classifier ' + str(classifier) + ' is empty')
            continue
        for j in range(0, len(imagedirs)):
            imagedir = imagedirs[j]
            print('classifier:' + classifier + ' image directory:' + imagedir)
            totTargets, totMatches, totExtraMatches = test_classifier(cascade_classifier, imagedir,
                                                                      use_visual_output=use_visual_output)
            print('totTargets:' + str(totTargets) + ' totMatches:' + str(totMatches) + ' tot ExtraMatches:' + str(
                totExtraMatches) + '           ')
            if totTargets:
                matches_over_targets_matrix[i, j] = float(totMatches) / totTargets
            else:
                matches_over_targets_matrix[i, j] = 0
            targets_matrix[i, j] = totTargets
            matches_matrix[i, j] = totMatches
            extras_matrix[i, j] = totExtraMatches
    results_dict = {'classifiers': classifiers, 'imagedirectories': imagedirs, 'totTargets': str(targets_matrix),
                    'totMatches': str(matches_matrix), 'ExtraMatches': str(extras_matrix)}

    with open(results_filename, 'a') as outfile:
        json.dumps(results_dict, outfile, indent=2)
    #print json.dumps(d, indent = 2, separators=(',', ': '))

    answers = calc_precision_recall(targets_matrix, matches_matrix, extras_matrix, classifiers, imagedirs)
    date_string = plot_confusion_matrix2(matches_over_targets_matrix, imagedirs, classifiers, targets_matrix,
                                      matches_matrix, extras_matrix)

    # maybe add cliassifier names
    html_name = 'classifier_results/results_' + '.' + date_string + '.html'

    write_classifier_html(html_name, date_string, answers, targets_matrix, matches_matrix, extras_matrix)

    return (answers, date_string, targets_matrix, matches_matrix, extras_matrix)


if __name__ == "__main__":
    print('start')

    parser = argparse.ArgumentParser(description='rate classifier')
    # parser.add_argument('-i', dest='imageDir', default='/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/databaseImages/googImages/shirt/longSleeve/BUTTONSHIRT/',help='directory with the images')
    dir = '/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/databaseImages/testdirs'
    # dir = '/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/databaseImages/googImages/shirt/'
    parser.add_argument('-i', dest='imageDir', default=dir, help='directory with the images')
    parser.add_argument('-v', dest='use_visual_output', default=False, help='whether to use vis output')
    classifier_dir = '/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/trendi_guru_modules/classifier_stuff/classifiers_to_test/face'
    # classifier_dir = 'classifiers_to_test'
    parser.add_argument('-c', dest='classifierDir', default=classifier_dir,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()
    classifierDir = args.classifierDir
    imageDir = args.imageDir
    use_visual_output = args.use_visual_output
    print('classifierDir ' + classifierDir)
    print('imageDir ' + imageDir)
    print('visual output ' + str(use_visual_output))
    # print 'Output file is "', outputfile

    answers, date_string, targets_matrix, matches_matrix, extras_matrix = test_classifiers(classifierDir=classifierDir,
                                                                                           imageDir=imageDir,
                                                                                           use_visual_output=use_visual_output)
    # head, classifier_name = os.path.split(classifier_path)


