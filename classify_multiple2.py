#!/usr/bin/env python
__author__ = 'jeremy'
import cv2
import re
import string
import sys
import urllib
import Utils
import json
#import pdb


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


def classify_image(pathToImageFile_Or_cv2ImageArray):
    REMOTE_FILE = False
    #shirt_found = False
    #fingerprint = ""
    classification_dict = {}

    #if we were given a path string, load image
    if isinstance(pathToImageFile_Or_cv2ImageArray, str):
        if "://" in pathToImageFile_Or_cv2ImageArray:
            FILENAME = pathToImageFile_Or_cv2ImageArray.split('/')[-1].split('#')[0].split('?')[0]
            res = urllib.urlretrieve (pathToImageFile_Or_cv2ImageArray, FILENAME)
            pathToImageFile_Or_cv2ImageArray = FILENAME
            REMOTE_FILE = True
        
	img = cv2.imread(pathToImageFile_Or_cv2ImageArray)
        print('in classify.py:classify_image: got string')
    else:
        print('in classify.py:classify_image: got array')
        img = pathToImageFile_Or_cv2ImageArray

    Nrects=0
    Nimages=0
    Ntargets=0
    Nmatches=0
    
    CLASSIFIER_FOLDER = "/home/www-data/web2py/applications/fingerPrint/modules/"
    classifier_xml_dict = {"shirtClassifier.xml":"shirt", "pantsClassifier.xml":"pants", "dressClassifier.xml":"dress"}
    cascades={}
    
    #create all the cascades
    for classifier_xml, classifier_id in classifier_xml_dict.iteritems():
        cascades[classifier_id] = cv2.CascadeClassifier(CLASSIFIER_FOLDER + classifier_xml)

    answers = {}

    for classifier_id, cascade in cascades.iteritems():
        detectedRects = cascade.detectMultiScale(img)
        Nmatches=len(detectedRects)
	if Nmatches > 0:
            answers[classifier_id] = detectedRects.tolist()
    return(answers)
