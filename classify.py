#!/usr/bin/env python
__author__ = 'jeremy'
import cv2
import re
import string
import sys
#import pdb

#basedir = "/home/jeremy/Dropbox/projects/clothing_recognition/Images/shirts/BUTTONSHIRT/"
#datafilename = "buttonShirtPositives.txt"

def intersectionOverUnion(r1,r2):
    print(r1,r2)

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
    print('x,y,w,h:'+str(intersectionx)+','+str(intersectiony)+','+str(intersectionw)+','+str(intersectionh))
    totarea=r1[2]*r1[3]+r2[2]*r2[3]   #this includes overlap twice
    intersectionarea=intersectionw*intersectionh
    totarea=totarea-intersectionarea  #now totarea includes overlap only once
    iou=float(intersectionarea)/float(totarea)
    print('totarea,intarea,iou:'+str(totarea)+','+str(intersectionarea)+','+str(iou))
    return(iou)

def detect(img,actualRects):
    global cascade,iouThreshold
    nTargets=0
    nMatches=0
    detectedRects = cascade.detectMultiScale(img)
#    print('detected rects:'+str(detectedRects))
#    print('actual rects:'+str(actualRects))
    # display until escape key is hit
    while True:
        # get a list of rectangles
        for x,y, width,height in detectedRects:
            cv2.rectangle(img, (x,y), (x+width, y+height), color=BLUE,thickness=1)
        for x,y, width,height in actualRects:
            cv2.rectangle(img, (x,y), (x+width, y+height), color=GREEN,thickness=1)
        for r1 in actualRects:
            iou=0
            correctMatch=False
            for r2 in detectedRects:
                iou=intersectionOverUnion(r1,r2)
                if iou>iouThreshold:
                    correctMatch=True
                    break
            nTargets=nTargets+1
            if correctMatch:
                nMatches=nMatches+1
            print('percentage:'+str(iou))
         # display!
        cv2.imshow('input', img)

        k = 0xFF & cv2.waitKey(0)
        # escape key (ASCII 27) closes window
        if k == 27:
            break
        elif k == ord(' '):
            if (videoMaker.isOpened()):
                #cv.Resize(img, img2, interpolation=CV_INTER_LINEAR)
                img2=cv2.resize(img,(420,420))
                hhh,www,ddd=img2.shape
                print('video:h:'+str(hhh)+'w:'+str(www))
                videoMaker.write(img2)
            return(nTargets,nMatches)

    # if esc key is hit, quit!
    exit()


BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG


def classify_image(pathToImageFile_Or_cv2ImageArray):

    #if we were given a path string, load image
    if isinstance(pathToImageFile_Or_cv2ImageArray, str):
        img = cv2.imread(pathToImageFile_Or_cv2ImageArray)
        print('in classify.py:classify_image: got string')
    else:
        print('in classify.py:classify_image: got array')
        img = pathToImageFile_Or_cv2ImageArray

    Nrects=0
    Nimages=0
    Ntargets=0
    Nmatches=0
    #cascade=[]

    cascades=[]
    xmlNames=[]

    xmlNames.append("/home/www-data/web2py/applications/fingerPrint/modules/dressClassifier001.xml")
#    xmlNames.append("/home/www-data/web2py/applications/fingerPrint/modules/shirtClassifier.xml")
#    xmlNames.append("/home/www-data/web2py/applications/fingerPrint/modules/pantsClassifier.xml")

# maybe bug is here
    cascades.append(cv2.CascadeClassifier(xmlNames[0]))
    check_empty=cascades[0].empty()
    print('in classify.py:classify_image:fist classifier empty?',str(check_empty))
#    cascades.append(cv2.CascadeClassifier(xmlNames[1]))
#    check_empty=cascades[0].empty()
 #   print('in classify.py:classify_image:second classifier empty?',str(check_empty))
 #   cascades.append(cv2.CascadeClassifier.load(xmlNames[0]))
 #   cascades.append(cv2.CascadeClassifier.load(xmlNames[1]))

    iouThreshold=0.1
#    cv2.CascadeClassifier.load('shirtClassifier.xml')
    #cv2.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) -> objects
    #cv2.CascadeClassifier.detectMultiScale(image, rejectLevels, levelWeights[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize[, outputRejectLevels]]]]]]) -> objects
    #videoMaker = cv2.VideoWriter('testShirt.mov',cv2.cv.CV_FOURCC('m', 'p', '4', 'v'),50,(420,420),isColor=True)
    #fourcc = cv2.VideoWriter(*'XVID')
    #videoMaker = cv2.VideoWriter('output.avi',fourcc, 50.0, (420,420))
    #cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    #videoMaker = cv2.VideoWriter('test.mov',,50,(420,420),isColor=True)


    #def greetings(word1=first_arg, word2=second_arg):
    #print("{} {}".format(word1, word2))

    #not sure what the if _name_=_main_ is good for
    #if __name__ == "__main__":
    answers={}
    for i in range(0,len(cascades)):
        detectedRects = cascades[i].detectMultiScale(img)
        Nmatches=len(detectedRects)
    #    print('targets:'+str(Nmatches))
        answers[xmlNames[i]] = detectedRects
    return answers


def classify_image_with_classifier(pathToImageFile_Or_cv2ImageArray, classifier_xml):

    #if we were given a path string, load image
    if isinstance(pathToImageFile_Or_cv2ImageArray, str):
        img = cv2.imread(pathToImageFile_Or_cv2ImageArray)
        print('in classify.py:classify_image: got string')
    else:
        print('in classify.py:classify_image: got array')
        img = pathToImageFile_Or_cv2ImageArray

    Nrects=0
    Nimages=0
    Ntargets=0
    Nmatches=0
    #cascade=[]

    cascades=[]
    xmlNames=[]

    xmlNames.append(classifier_xml)
#    xmlNames.append("/home/www-data/web2py/applications/fingerPrint/modules/shirtClassifier.xml")
#    xmlNames.append("/home/www-data/web2py/applications/fingerPrint/modules/pantsClassifier.xml")

# maybe bug is here
    cascades.append(cv2.CascadeClassifier(xmlNames[0]))
    check_empty=cascades[0].empty()
    print('in classify.py:classify_image:fist classifier empty?',str(check_empty))
#    cascades.append(cv2.CascadeClassifier(xmlNames[1]))
#    check_empty=cascades[0].empty()
 #   print('in classify.py:classify_image:second classifier empty?',str(check_empty))
 #   cascades.append(cv2.CascadeClassifier.load(xmlNames[0]))
 #   cascades.append(cv2.CascadeClassifier.load(xmlNames[1]))

    iouThreshold=0.1
#    cv2.CascadeClassifier.load('shirtClassifier.xml')
    #cv2.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) -> objects
    #cv2.CascadeClassifier.detectMultiScale(image, rejectLevels, levelWeights[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize[, outputRejectLevels]]]]]]) -> objects
    #videoMaker = cv2.VideoWriter('testShirt.mov',cv2.cv.CV_FOURCC('m', 'p', '4', 'v'),50,(420,420),isColor=True)
    #fourcc = cv2.VideoWriter(*'XVID')
    #videoMaker = cv2.VideoWriter('output.avi',fourcc, 50.0, (420,420))
    #cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    #videoMaker = cv2.VideoWriter('test.mov',,50,(420,420),isColor=True)


    #def greetings(word1=first_arg, word2=second_arg):
    #print("{} {}".format(word1, word2))

    #not sure what the if _name_=_main_ is good for
    #if __name__ == "__main__":
    answers={}
    for i in range(0,len(cascades)):
        detectedRects = cascades[i].detectMultiScale(img)
        Nmatches=len(detectedRects)
    #    print('targets:'+str(Nmatches))
        answers[xmlNames[i]] = detectedRects
    return(answers)
