from __future__ import print_function
#!/usr/bin/env python
__author__ = 'jeremy'
import sys, getopt
import cv2,cv
import time
import re
import string
import sys
import sqlsoup
import pdb
import datetime
import json
import matplotlib.pyplot as plt
from pylab import *
import os
import numpy as np
import cProfile, pstats, StringIO
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

def intersectionOverUnion(r1,r2):
#    print(r1,r2)

#    a if test else b

    intersectionx = max(r1[0],r2[0])
    intersectiony = max(r1[1],r2[1])
    intersectionw = min(r1[0] + r1[2], r2[0] + r2[2])-intersectionx
    if intersectionw<0:
        intersectionw=0
    intersectionh = min(r1[1] + r1[3], r2[1] + r2[3])-intersectiony
    if intersectionh<0:
        intersectionh=0
 #   intersectionh -= intersectiony;
#    print('x,y,w,h:'+str(intersectionx)+','+str(intersectiony)+','+str(intersectionw)+','+str(intersectionh))
    totarea=r1[2]*r1[3]+r2[2]*r2[3]   #this includes overlap twice
    intersectionarea=intersectionw*intersectionh
    totarea=totarea-intersectionarea  #now totarea includes overlap only once
    iou=float(intersectionarea)/float(totarea)
  #  print('totarea,intarea,iou:'+str(totarea)+','+str(intersectionarea)+','+str(iou))
    return(iou)

def detect(classifier,img_array,bbs_human):
    global iouThreshold
    nTargets=0
    nMatches=0
    nFalseMatches=0
    bbs_computer = classifier.detectMultiScale(img_array)
#	n=[]#
#	m=()
#	print('len null='+str(len(n))+' len null tuple='+str(len(m)))
    n_bbs_computer=len(bbs_computer)
    n_bbs_human=len(bbs_human)
    if bbs_computer is ():
        n_bbs_computer=0
        print('no computer BB')
        #check why this is ()
    else:
    #	print('classifier:'+str(classifier)+' #bbsHuman:'+str(n_bbs_human)+' #bbsComputer'+str(n_bbs_human))
        #print('bb:'+str(bbs_human)+' bbComputer:'+str(bbs_computer))
#9		print('bbComputer:'+str(bbs_computer[0]))
        # display until escape key is hit
        # get a list of rectangles
        for x,y, width,height in bbs_computer:
            cv2.rectangle(img_array, (x,y), (x+width, y+height), color=BLUE,thickness=1)
    for x,y, width,height in bbs_human:
        cv2.rectangle(img_array, (x,y), (x+width, y+height), color=GREEN,thickness=1)
    if bbs_human is None:
        return(0,0,n_bbs_computer)
    if bbs_computer is None:
        return(n_bbs_human,0,0)
    for r1 in bbs_human:
        iou=0
        correctMatch=False
        for r2 in bbs_computer:
            iou=intersectionOverUnion(r1,r2)
            if iou>iouThreshold:
                correctMatch=True
            else:
                nFalseMatches=nFalseMatches+1     #check this for case of multiple bbs, i think its not quite right
            #print('roi:'+str(iou))
        nTargets=nTargets+1
        if correctMatch:
            nMatches=nMatches+1
 #       print('percentage:'+str(iou))
     # display!
#    print('# detected rects:'+str(len(bbs_computer))+' # actual rects:'+str(len(bbs_human))+' false matches:'+str(nFalseMatches))
    k=0
    if use_visual_output is True:
        cv2.imshow('input', img_array)
    #    cv2.waitKey(100)
        k = 0xFF & cv2.waitKey(10)
        return(nTargets,nMatches,nFalseMatches)
        # escape key (ASCII 27) closes window
    #    k=27
        if k == 27:
            return(nTargets,nMatches,nFalseMatches)
        #	k=30
        elif k == ord(' '):
                if (videoMaker.isOpened()):
                        #cv.Resize(img, img2, interpolation=CV_INTER_LINEAR)
                    img2=cv2.resize(img,(420,420))
                    hhh,www,ddd=img2.shape
                    print('video:h:'+str(hhh)+'w:'+str(www))
                    videoMaker.write(img2)
                    return(nTargets,nMatches,nFalseMatches)
    else:
        return(nTargets,nMatches,nFalseMatches)

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

def test_classifier(classifier,imageDirectory,itemDescriptionString,*args,**kwargs):
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
def test_classifiers(trainDir='057',imageDir='images/cjdb'):
    resultsDir='classifier_results'
    results_filename=os.path.join(resultsDir,'classifier_results_'+trainDir+'.txt')
    rootDir = imageDir
    global max_files_to_try
    max_files_to_try=100

    #results_matrix=[[1,2,3],[4,5,6],[7,8,9]]
    #plot_confusion_matrix(10,results_matrix)
    #variable = raw_input('input something!: ')

    #go thru image directories
    subdirlist=[x[1] for x in os.walk(rootDir)]
#	print(' subdirlist'+str(subdirlist))
    if len(subdirlist)==0:
        print('empty image directory:'+str(rootDir)+str(subdirlist))
        exit()
    top_subdirlist=subdirlist[0]
    print('top subdirlist'+str(top_subdirlist))
    n_categories=len(top_subdirlist)
    for subdir in top_subdirlist:
        imageDirectories.append(os.path.join(rootDir,subdir))
        searchStrings.append('class:'+subdir)
        print('image directory:'+str(subdir))

    #go thru classifier directories
    subdirlist=[x[1] for x in os.walk(trainDir)]
#	print(' subdirlist'+str(subdirlist))
    if len(subdirlist)==0:
        print('empty classifier directory:'+str(trainDir)+str(subdirlist))
        exit()
    top_subdirlist=subdirlist[0]
    print('top subdirlist'+str(top_subdirlist))
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


##############
#outermost prog here
##############

def main(argv):
   print('start')
   imageDir = 'images/cjdb'
   outputDir = '057'

#   parser = OptionParser()
#   parser.add_option('-o','-c','--classifiers','--input','--inputfiles',dest='outputDir',help='classifier directory',metavar='FILE',default='060')
#   parser.add_option('-i','-f','--imageDir',dest='imageDir',metavar='FILE2',help='image direcotry',default='images/imageNet/images')
#   (options, args) = parser.parse_args()

   #parser = argparse.ArgumentParser(description='Process some integers.')
   #parser.add_argument('integers', metavar='N', type=int, nargs='+',help='an integer for the accumulator')
   #parser.add_argument('--sum', dest='accumulate', action='store_const',const=sum, default=max,help='sum the integers (default: find the max)')
   parser = argparse.ArgumentParser(description='Process some integers.')
#   parser.add_argument('integers', metavar='N', type=int, nargs='+',help='an integer for the accumulator')
   parser.add_argument('-o', dest='outputDir', default='057',help='sum the integers (default: find the max)')
   parser.add_argument('-i', dest='imageDir', default='images/cjdb',help='sum the integers (default: find the max)')
   parser.add_argument('-v', dest='use_visual_output', default='images/cjdb',help='sum the integers (default: find the max)')

   args = parser.parse_args()


   outputDir=args.outputDir
   imageDir=args.imageDir
   use_visual_output=args.use_visual_output
   print('imageDir '+imageDir)
   print('outputDir '+outputDir)
   print('visual output '+str(use_visual_output))
#  print 'Output file is "', outputfile

   pr = cProfile.Profile()
   pr.enable()

   test_classifiers(outputDir,imageDir)

   pr.disable()
   s = StringIO.StringIO()
   sortby = 'cumulative'
   ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
   ps.print_stats()
   print(s.getvalue())

if __name__ == "__main__":
    main(sys.argv[1:])


''''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile="])
#      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile=:"])
   except getopt.GetoptError:
      print('test.py -i <inputfile>')
#      print 'test.py -i <inputfile> -o <outputfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
#         print('test.py -i <inputfile>')
          print 'test.py -i <imageDir> -o <trainDir>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         imageDir = arg
         print('infile '+imageDir)
#      elif opt in ("-o", "--ofile"):
#         outputfile = arg
'''''
