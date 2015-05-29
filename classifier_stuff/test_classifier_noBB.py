from __future__ import print_function
#!/usr/bin/env python
__author__ = 'jeremy'
import cv2
import json
from pylab import *
import os
import numpy as np
import argparse

import Utils
import background_removal

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

def detect_no_bb(classifier, img_array, use_visual_output=True):
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


def plot_confusion_matrix(m, image_dirs, classifier_names):
    #conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], [3,31,0,0,0,0,0,0,0,0,0], [0,4,41,0,0,0,0,0,0,0,1], [0,1,0,30,0,6,0,0,0,0,1], [0,0,0,0,38,10,0,0,0,0,0], [0,0,0,3,1,39,0,0,0,0,4], [0,2,2,0,4,1,31,0,0,0,2], [0,1,0,0,0,0,0,36,0,2,0], [0,0,0,0,0,0,1,5,37,5,1], [3,0,0,0,0,0,0,0,0,39,0], [0,0,0,0,0,0,0,0,0,0,38] ]

    fig = plt.figure()
#    ax = fig.add_subplot(111)
    max = np.amax(m)
    scaled_matrix = m * 255 / max
    res = plt.imshow(array(m) * 256 / max, cmap=cm.jet, interpolation='nearest')
    plt.xticks(rotation=90)
    plt.xticks(np.arange(0, len(image_dirs)), image_dirs)  # check if x is classifier or image category
    plt.yticks(np.arange(0, len(classifier_names)), classifier_names)  #check if y is classifiers or categories
    cb = fig.colorbar(res)

    rows, cols = np.shape(m)

    for i in range(0, rows):
        for j in range(0, cols):
            plt.text(j + 1.2, i + .2, str(m[i, j]), fontsize=6)
            if i == j:  # detector x on pics of x
                #                classifier_sum=classifier_sum+float(j)/float(den)
                pass
            else:  # detector x on pics of something else
                #                classifier_sum=classifier_sum-float(j)/float(den)
                pass
    figName = 'classifier_results/confusion.png'
    savefig(figName, format="png")
    plt.show()
#=======
#    #conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], [3,31,0,0,0,0,0,0,0,0,0], [0,4,41,0,0,0,0,0,0,0,1], [0,1,0,30,0,6,0,0,0,0,1], [0,0,0,0,38,10,0,0,0,0,0], [0,0,0,3,1,39,0,0,0,0,4], [0,2,2,0,4,1,31,0,0,0,2], [0,1,0,0,0,0,0,36,0,2,0], [0,0,0,0,0,0,1,5,37,5,1], [3,0,0,0,0,0,0,0,0,39,0], [0,0,0,0,0,0,0,0,0,0,38] ]
#    conf_arr=m
#    norm_conf = []
#    classifier_goodness=[]
#    k=0
#    for i in conf_arr:
#        classifier_sum=0.0
#        a = 0
#        tmp_arr = []
#        n=0
#        a = sum(i,0)
#        for j in i:
#            den=float(n_samples[n])
#            if a==0:  #this is prob unneeded as long as n_samples[i] != 0
#                tmp_arr.append(0)
#                #no change to classifier sum
#            else:
#                print('den:'+str(den))
#                tmp_arr.append(float(j)/float(a))
#                if den>0:
#                    if n==k:  #detector x on pics of x
#                        classifier_sum=classifier_sum+float(j)/float(den)
#                    else: #detector x on pics of something else
#                        classifier_sum=classifier_sum-float(j)/float(den)
#                    l=float(j)/float(den)
#                    print('l:'+str(l))
#            print('detected:'+str(j)+' tot targets'+str(n_samples[n])+ ' value:'+str(tmp_arr[-1])+'sum:'+str(classifier_sum))
#            n=n+1
#        classifier_sum=(classifier_sum+n-2)/(n-1)
#        classifier_goodness.append(classifier_sum)
#        norm_conf.append(tmp_arr)
#        k=k+1
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    res = ax.imshow(array(norm_conf), cmap=cm.jet, interpolation='nearest')
#    m=0
#    n=0
#    for i, cas in enumerate(conf_arr):
#        classifier_goodsum=0
#        classifier_badsum=0
#        n=0
#        for j, c in enumerate(cas):
#           # if c>0:
#            plt.text(j-.2, i+.2, c, fontsize=14)
#            n=n+1
#        plt.text(j+1.2, i+.2, classifier_goodness[m], fontsize=14)
#        m=m+1
#    cb = fig.colorbar(res)
#
#    figName=os.path.join(resultsDir,'confusion.'+trainDir+'.png')
#    savefig(figName, format="png")
#    plt.show()
#>>>>>>> 78605526ed1ae5ce2b7fe35564ddda8cb1f019a3

def test_classifier(classifier, imagesDir, max_files_to_try=10000):
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
        nMatches = detect_no_bb(classifier, img_array)
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
def test_classifiers(classifierDir='../classifiers/', imageDir='images', use_visual_output=True):
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
            totTargets, totMatches, totExtraMatches = test_classifier(cascade_classifier, imagedir)
            print('totTargets:' + str(totTargets) + ' totMatches:' + str(totMatches) + ' tot ExtraMatches:' + str(
                totExtraMatches) + '           ')
            matches_over_targets_matrix[i, j] = float(totMatches) / totTargets
            targets_matrix[i, j] = totTargets
            matches_matrix[i, j] = totMatches
            extras_matrix[i, j] = totExtraMatches
            results_dict = {'classifier': classifier, 'directory': imagedir, 'totTargets': totTargets,
                            'totMatches': totMatches, 'FalseMatches': totExtraMatches}

            with open(results_filename, 'a') as outfile:
                json.dump(results_list, outfile, indent=2)
    #print json.dumps(d, indent = 2, separators=(',', ': '))
    plot_confusion_matrix(matches_over_targets_matrix, imagedirs, classifiers)


if __name__ == "__main__":
    print('start')

    parser = argparse.ArgumentParser(description='rate classifier')
    # parser.add_argument('-i', dest='imageDir', default='/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/databaseImages/googImages/shirt/longSleeve/BUTTONSHIRT/',help='directory with the images')
    dir = '/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/databaseImages/testdirs'
    # dir = '/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/databaseImages/googImages/shirt/'
    parser.add_argument('-i', dest='imageDir', default=dir, help='directory with the images')
    parser.add_argument('-v', dest='use_visual_output', default=True, help='whether to use vis output')
    parser.add_argument('-c', dest='classifierDir', default='../classifiers/',
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()
    classifierDir = args.classifierDir
    imageDir = args.imageDir
    use_visual_output = args.use_visual_output
    print('classifierDir ' + classifierDir)
    print('imageDir ' + imageDir)
    print('visual output ' + str(use_visual_output))
    # print 'Output file is "', outputfile

    test_classifiers(classifierDir=classifierDir, imageDir=imageDir, use_visual_output=use_visual_output)

