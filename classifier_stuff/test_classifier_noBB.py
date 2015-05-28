from __future__ import print_function
#!/usr/bin/env python
__author__ = 'jeremy'
import cv2
import time
import json
from pylab import *
import os
import numpy as np
import argparse

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
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG


def show_positives_file(positives_file):
    pass


def detect_no_bb(classifier, img_array, use_visual_output=True):
    bbs_computer = classifier.detectMultiScale(img_array)
    if bbs_computer is () or bbs_computer is None:
        n_bbs_computer=0
        print('no computer BB')
        return (0)
        #check why this is ()
    if use_visual_output is True:
        for x,y, width,height in bbs_computer:
            cv2.rectangle(img_array, (x,y), (x+width, y+height), color=BLUE,thickness=1)
        cv2.imshow('input', img_array)
    #    cv2.waitKey(100)
        k = 0xFF & cv2.waitKey(10)
        return (1)

def plot_confusion_matrix(n_samples,m):
    #conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], [3,31,0,0,0,0,0,0,0,0,0], [0,4,41,0,0,0,0,0,0,0,1], [0,1,0,30,0,6,0,0,0,0,1], [0,0,0,0,38,10,0,0,0,0,0], [0,0,0,3,1,39,0,0,0,0,4], [0,2,2,0,4,1,31,0,0,0,2], [0,1,0,0,0,0,0,36,0,2,0], [0,0,0,0,0,0,1,5,37,5,1], [3,0,0,0,0,0,0,0,0,39,0], [0,0,0,0,0,0,0,0,0,0,38] ]
    conf_arr=m
    norm_conf = []
    classifier_goodness=[]
    k=0
    for i in conf_arr:
        classifier_sum=0.0
        a = 0
        tmp_arr = []
        n=0
        a = sum(i,0)
        for j in i:
            den=float(n_samples[n])
            if a==0:  #this is prob unneeded as long as n_samples[i] != 0
                tmp_arr.append(0)
                #no change to classifier sum
            else:
                print('den:'+str(den))
                tmp_arr.append(float(j)/float(a))
                if den>0:
                    if n==k:  #detector x on pics of x
                        classifier_sum=classifier_sum+float(j)/float(den)
                    else: #detector x on pics of something else
                        classifier_sum=classifier_sum-float(j)/float(den)
                    l=float(j)/float(den)
                    print('l:'+str(l))
            print('detected:'+str(j)+' tot targets'+str(n_samples[n])+ ' value:'+str(tmp_arr[-1])+'sum:'+str(classifier_sum))
            n=n+1
        classifier_sum=(classifier_sum+n-2)/(n-1)
        classifier_goodness.append(classifier_sum)
        norm_conf.append(tmp_arr)
        k=k+1
    fig = plt.figure()
#    ax = fig.add_subplot(111)
    res = plt.imshow(array(norm_conf), cmap=cm.jet, interpolation='nearest')
    plt.xticks(rotation=90)
    plt.xticks(np.arange(0,len(n_samples)),imageDirectories)   #check if x is classifier or image category
    plt.yticks(np.arange(0,len(n_samples)),classifierNames)   #check if y is classifiers or categories
    m=0
    n=0
    for i, cas in enumerate(conf_arr):
        classifier_goodsum=0
        classifier_badsum=0
        n=0
        for j, c in enumerate(cas):
           # if c>0:
            plt.text(j-.2, i+.2, int(c), fontsize=10)
            n=n+1
        plt.text(j+1.2, i+.2, classifier_goodness[m], fontsize=6)
        m=m+1
    cb = fig.colorbar(res)

    figName=os.path.join(resultsDir,'confusion.'+trainDir+'.png')
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

def test_classifier(classifier, positivesDirectory, negativesDirectory, *args, **kwargs):
    n_files=0
    i=0
    totTargets=0
    totMatches=0
    totFalseMatches=0
    for dirName,subdirList,fileList in os.walk(imageDirectory,topdown=True):
        for file in fileList:
            full_name=os.path.join(dirName,file)
    #		print('fullName:'+full_name)
            img_array = cv2.imread(full_name)
            if img_array is not None:
                h, w, d = img_array.shape
#				f.write(full_name + ' 1 1 1 '+ str(w-2) + ' ' + str(h-2)  + '\n')
#				relative_filename=os.path.join(dirName3,file)`
#				absolute_filename=os.path.join(abs_root_dir,file)
                n_files=n_files+1
                if use_visual_output is True:
                    cv2.imshow('input', img_array)
                    #    cv2.waitKey(100)
                    k = 0xFF & cv2.waitKey(10)
                bb = [1,1,w-2,h-2]
                bbs = [bb]   #in case we ever need to deal with multiple items per image, this is a list
                nTargets,nMatches,nFalseMatches=detect(classifier,img_array,bbs)
                totTargets=totTargets+nTargets
                totMatches=totMatches+nMatches
                totFalseMatches=totFalseMatches+nFalseMatches
                print('nTargets:'+str(nTargets)+' tot:'+str(totTargets)+' nMatches:'+str(nMatches)+' tot:'+str(totMatches)+' nFalse:'+str(nFalseMatches)+' tot:'+str(totFalseMatches)+' name:'+str(full_name))
                #.filter(db.items.fingerprint.is_(None))
            else:
                print('file:' + full_name +' read error')  #
            if n_files==max_files_to_try:
                print('reached max of '+str(max_files_to_try)+' files to check')
                break
        if n_files==max_files_to_try:
        #	print('reached max of '+str(max_files_to_try)+' files to check')
            break
    return(totTargets,totMatches,totFalseMatches)

###########
# run using classifier we have on images from db
###########
#classifierNames.append("dressClassifier013.xml")
def test_classifiers(classifierDir='../classifiers/', imageDir='images', use_visual_output=True):
    resultsDir='classifier_results'
    results_filename=os.path.join(resultsDir,'classifier_results_'+trainDir+'.txt')
    max_files_to_try = 1000

    #go thru image directories
    subdirlist = Utils.immediate_subdirs(dir)
    if len(subdirlist)==0:
        print('empty image directory:'+str(rootDir)+str(subdirlist))
        exit()
    n_testcategories = len(top_subdirlist)
    for subdir in subdirlist:
        searchStrings.append('class:'+subdir)
        print('image directory:'+str(subdir))

    #go thru classifier directories
    classifierlist = [x[1] for x in os.walk(trainDir)]
#	print(' subdirlist'+str(subdirlist))
    if len(subdirlist)==0:
        print('empty classifier directory:'+str(trainDir)+str(subdirlist))
        exit()
    top_subdirlist=subdirlist[0]
    print('top subdirlist'+str(top_subdirlist))

    subdirlist = [x[0] for x in os.walk(classifier)]

    n_classifiers=len(top_subdirlist)
    for subdir in top_subdirlist:
        classifier_name=os.path.join(trainDir,subdir)
        classifier_name=os.path.join(classifier_name,'cascade.xml')
    #	classifier_name=trainDir+'/'+subdir+'/cascade.xml'
        print('classifierName:'+classifier_name)
        classifierNames.append(classifier_name)

    results_list=[]
    results_matrix=np.zeros([n_classifiers,n_categories])
    samplesize=[]
    results_matrixrow=[]
    gotSizesFlag=False
    for i in range(0,n_classifiers):
        results_row=[]
        results_matrixrow=[]
        cascade=cv2.CascadeClassifier(classifierNames[i])
        check_empty=cascade.empty()

        for j in range(0,n_categories):
            if check_empty:
                print('in test_classifier.py:classify_image: classifier '+classifierNames[i]+' is empty!! ',str(check_empty))
                totTargets,totMatches,totFalseMatches = [0,0,0]
            else:
                print('cascade:'+classifierNames[i]+' directory:'+imageDirectories[j]+' search string:'+searchStrings[j])
                totTargets,totMatches,totFalseMatches = test_classifier(cascade,imageDirectories[j],searchStrings[j])

            print('totTargets:'+str(totTargets)+' totMatches:'+str(totMatches)+' tot FalseMatches:'+str(totFalseMatches)+'               ')
            results_dict={'classifier':classifierNames[i], 'directory':imageDirectories[j],'search string':searchStrings[j],'totTargets':totTargets,'totMatches':totMatches,'FalseMatches':totFalseMatches,'time':time.strftime('%c')}
            results_row.append(results_dict)
            results_list.append(results_row)
            if i==j:
                results_matrixrow.append(totMatches-totFalseMatches) #think about whether this is right for category i,j and for category i,i
            else:
                results_matrixrow.append(totMatches+totFalseMatches) #think about whether this is right for category i,j and for category i,i
            print('col '+str(j)+' entry:'+str(results_matrixrow[-1]))

            with open(results_filename, 'a') as outfile:
                json.dump(results_list, outfile, indent=2)
            if check_empty is False and gotSizesFlag is False:
                samplesize.append(totTargets)
        if check_empty is False and gotSizesFlag is False:
            gotSizesFlag=True
        results_matrix[i,:]=results_matrixrow
        print('row '+str(i)+' = '+str(results_matrixrow))
        print(results_matrix)
        print(samplesize)
    #print json.dumps(d, indent = 2, separators=(',', ': '))
    plot_confusion_matrix(samplesize,results_matrix)  #tottargets should prob be a vector of # of pics checked



if __name__ == "__main__":
    print('start')

    parser = argparse.ArgumentParser(description='rate classifier')
    # parser.add_argument('-i', dest='imageDir', default='/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/databaseImages/googImages/shirt/longSleeve/BUTTONSHIRT/',help='directory with the images')
    parser.add_argument('-i', dest='imageDir',
                        default='/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/databaseImages/googImages/shirt/',
                        help='directory with the images')
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

