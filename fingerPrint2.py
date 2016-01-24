__author__ = 'jeremy'
#try adding local binary patterns , and global stuff - color, mean, variance and other moments for fingerprinting
#lalala

import sys
import numpy as np
import cv2
import re
import string
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import urllib
import os
import classify
import json

# for console debugging:
import pdb

#PATH_TO_FILE = sys.argv[1]
REMOTE_FILE = False
FILENAME = ""

BLUE = [255, 0, 0]        # rectangle color
RED = [0, 0, 255]         # PR BG
GREEN = [0, 255, 0]       # PR FG
BLACK = [0, 0, 0]         # sure BG
WHITE = [255, 255, 255]   # sure FG


def crop_image_to_BB(img, BB_coordinates_string_or_array):
#    pdb.set_trace()
    if isinstance(BB_coordinates_string_or_array, basestring):
        BB_array = json.loads(BB_coordinates_string_or_array)
        for i in range (0, len(BB_array)):
            BB_array[i] = int(BB_array[i])
    else:
        BB_array = BB_coordinates_string_or_array


    x=BB_array[0]
    y=BB_array[1]
    w=BB_array[2]
    h=BB_array[3]
    hh, ww, d = img.shape
    if (x+w <= ww) and (y+h <= hh):
        rectok=True
        r=[x,y,w,h]
        #allRects.append(r)
        mask = np.zeros(img.shape[:2], np.uint8)
        roi= np.zeros((r[3],r[2],3),np.uint8)
        mask[r[0]:r[2], r[1]:r[3]] = 255

        for xx in range(r[2]):
            for yy in range(r[3]):
                roi[yy,xx,:]=img[yy+r[1],xx+r[0],:]

    else:
        rectok=False
        print('badrect for file:'+ FILENAME +' imsize:'+str(ww)+','+str(hh)+' vs.:'+str(x+w)+','+str(y+h))

    return roi #this is what fp(img) is expecting

def fp(pathToImageFile_Or_cv2ImageArray, bb=None):
    REMOTE_FILE = False
    #if given a string, check if URL and download if necessary
    if isinstance(pathToImageFile_Or_cv2ImageArray, str):
        if "://" in pathToImageFile_Or_cv2ImageArray:
            FILENAME = "temp.jpg"#pathToImageFile_Or_cv2ImageArray.split('/')[-1].split('#')[0].split('?')[0]
            res = urllib.urlretrieve (pathToImageFile_Or_cv2ImageArray, FILENAME)
            pathToImageFile_Or_cv2ImageArray = FILENAME
            REMOTE_FILE = True
        img = cv2.imread(pathToImageFile_Or_cv2ImageArray)
        os.remove(pathToImageFile_Or_cv2ImageArray)
    else: 
        img = pathToImageFile_Or_cv2ImageArray

    img = crop_image_to_BB(img, bb)
    
    s=5   #crop out the outer 1/s of the image for color/texture-based features
    h=img.shape[1]
    w=img.shape[0]
# old dumb crop
#    r=[h/s,w/s,h-2*h/s,w-2*w/s]
    #print('r='+str(r))
#    roi= np.zeros((r[3],r[2],3),np.uint8)
#    for xx in range(r[2]):
 #       for yy in range(r[3]):
  #          roi[yy  ,xx,:]=img[yy+r[1],xx+r[0],:]

#new awesome efficient crop
    r=[w/s,h/s,w-2*w/s,h-2*h/s] #x,y,w,h
    roi = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    hueValue=hsv[:,:,0]
    satValue=hsv[:,:,1]
    intValue=hsv[:,:,2]
    # print('hueV'+str(hueValue.shape))
    # plt.subplot(2,3,4)
    # plt.title('hue')
    # plt.imshow(hueValue,cmap=plt.cm.jet) #cm.Greys_r)
    # plt.subplot(2,3,5)
    # plt.title('sat')
    # plt.imshow(satValue,cmap = cm.Greys_r)
    # plt.subplot(2,3,6)
    # plt.title('int')
    # plt.imshow(intValue,cmap = cm.Greys_r)

#OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
#histograms
    bins=25
    hueXs = np.linspace(0,180,bins)
    nPixels=roi.shape[0]*roi.shape[1]

    histHue=cv2.calcHist([hsv],[0],None,[bins],[0,180])
    histHue=[item for sublist in histHue for item in sublist] #flatten nested
    histHue=np.divide(histHue,nPixels)

    histSat=cv2.calcHist([hsv],[1],None,[bins],[0,255])
    histSat=[item for sublist in histSat for item in sublist]
    histSat=np.divide(histSat,nPixels)

    histInt=cv2.calcHist([hsv],[2],None,[bins],[0,255])
    histInt=[item for sublist in histInt for item in sublist] #flatten nested list
    histInt=np.divide(histInt,nPixels)

#Uniformity  t(5)=sum(p.^ 2);
    hueUniformity=np.dot(histHue,histHue)
    satUniformity=np.dot(histSat,histSat)
    intUniformity=np.dot(histInt,histInt)

#Entropy   t(6)=-sum(p. *(log2(p+ eps)));
    eps=1e-15
    lHue=np.log2(histHue+eps)
    hueEntropy=np.dot(histHue,lHue)
    lSat=np.log2(histSat+eps)
    satEntropy=np.dot(histSat,lSat)
    lInt=np.log2(histInt+eps)
    intEntropy=np.dot(histInt,lInt)



    resultVector=[hueUniformity,satUniformity,intUniformity,hueEntropy,satEntropy,intEntropy]
    resultVector=np.concatenate((resultVector,histHue, histSat), axis=0)
    #print('result')
    return(resultVector)

def rmse(predictions, targets):
    return np.sqrt((((predictions - targets)) ** 2).mean())

def myRange(start,stop,inc):
    r=start
    while r<stop:
        yield r
        r+=inc

def gaussian1(x,x0,c,sigma):
    return c*np.exp(-(x-x0)**2/(2*sigma**2))

#def classify_and_fingerprint(path_to_image_file=None):
def classify_and_fingerprint(path_to_image_file):
    #pdb.set_trace()
    print("in fingerPrint2:classify_and_fingerprint:path_to_image_file: " + str(path_to_image_file))
    REMOTE_FILE = False
    shirt_found = False
    fingerprint = ""
    classification_dict = {}
    #read arg from command line if none given to function
    if path_to_image_file is not None:
        PATH_TO_FILE = path_to_image_file
    #else:
        #wtf is supposed to happen here - who is calling this from command line??

    #we want to be able to read URL as well as local file path
    if "://" in path_to_image_file:
        FILENAME = path_to_image_file.split('/')[-1].split('#')[0].split('?')[0]
        res = urllib.urlretrieve (PATH_TO_FILE, FILENAME)
        PATH_TO_FILE = FILENAME
        REMOTE_FILE = True

    #pdb.set_trace()
    #main prog starts here
    img = cv2.imread(PATH_TO_FILE)
    roi = []
    classification_dict = classify.classify_image(img)
    BB_coordinates = classification_dict["/home/www-data/web2py/applications/fingerPrint/modules/shirtClassifier.xml"]
    if len(BB_coordinates) > 0:
        if BB_coordinates[0] is not None:
            shirt_found = True
            roi = crop_image_to_BB(img, BB_coordinates[0])
        else:
            roi=img  
            BB_coordinates = [[0,0,0,0]]
            print("in fingerPrint2.py: len(BB_coordinates)>0 but BB[0] is None -- REALLY WEIRD:")
            print(BB_coordinates)

    else:
#        roi = None  #if no BB was formed , don't return an answer!!!!
        print("in fingerPrint2.py:bad roi (could not crop?)-using entire img - len(BB)!>0")
	roi=img
        BB_coordinates = [[0,0,0,0]]
    if roi is not None:
        fingerprint = fp(roi)
    else:
        print("in fingerPrint2.py:bad roi (could not crop?) - using entire img (again)")
        fingerprint_length=56
        fingerprint=[0 for x in range(fingerprint_length)]
        fingerprint[0]=-1
        print("in fingerPrint2.py:fp="+str(fingerprint))

    if REMOTE_FILE:
        os.remove(PATH_TO_FILE)

    # right now, we're only doing shirts, so it's binary
    # 0 - nothing found, 1 - at least one shirt found.
    # even within this simplified case, still need to figure out
    # how to deal with multiple shirts in an image
    classification_list = []
    if shirt_found:
        classification_list.append(1)
    else:
        classification_list.append(0)

    print('in fingerPrint2.py:classify_and_fingerprint:fingerprint='+str(fingerprint))
    print('in fingerPrint2.py:classify_and_fingerprint:BB='+str(BB_coordinates))
    return classification_list, fingerprint, BB_coordinates

#takes classifier file path
def classify_and_fingerprint_with_classifier(path_to_image_file, classifier_xml):
    #pdb.set_trace()
    print("in fingerPrint2:classify_and_fingerprint:path_to_image_file: " + str(path_to_image_file))
    REMOTE_FILE = False
    item_found = False
    fingerprint = ""
    classification_dict = {}
    #read arg from command line if none given to function
    if path_to_image_file is not None:
        PATH_TO_FILE = path_to_image_file
    #else:
        #wtf is supposed to happen here - who is calling this from command line??

    #we want to be able to read URL as well as local file path
    if "://" in path_to_image_file:
        FILENAME = path_to_image_file.split('/')[-1].split('#')[0].split('?')[0]
        res = urllib.urlretrieve (PATH_TO_FILE, FILENAME)
        PATH_TO_FILE = FILENAME
        REMOTE_FILE = True

    #pdb.set_trace()
    #main prog starts here
    img = cv2.imread(PATH_TO_FILE)
    roi = []
    classification_dict = classify.classify_image_with_classifier(img, classifier_xml)
    BB_coordinates = classification_dict[classifier_xml]
    if len(BB_coordinates) > 0:
        if BB_coordinates[0] is not None:
            item_found = True
            roi = crop_image_to_BB(img, BB_coordinates[0])
        else:
            roi=img  
            BB_coordinates = [[0,0,0,0]]
            print("in fingerPrint2.py: len(BB_coordinates)>0 but BB[0] is None -- REALLY WEIRD:")
            print(BB_coordinates)

    else:
#        roi = None  #if no BB was formed , don't return an answer!!!!
        print("in fingerPrint2.py:bad roi (could not crop?)-using entire img - len(BB)!>0")
	roi=img
        BB_coordinates = [[0,0,0,0]]
    if roi is not None:
        fingerprint = fp(roi)
    else:
        print("in fingerPrint2.py:bad roi (could not crop?) - using entire img (again)")
        fingerprint_length=56
        fingerprint=[0 for x in range(fingerprint_length)]
        fingerprint[0]=-1
        print("in fingerPrint2.py:fp="+str(fingerprint))

    if REMOTE_FILE:
        os.remove(PATH_TO_FILE)

    # right now, we're only doing shirts, so it's binary
    # 0 - nothing found, 1 - at least one shirt found.
    # even within this simplified case, still need to figure out
    # how to deal with multiple shirts in an image
    classification_list = []
    if item_found:
        classification_list.append(1)
    else:
        classification_list.append(0)

    print('in fingerPrint2.py:classify_and_fingerprint:fingerprint='+str(fingerprint))
    print('in fingerPrint2.py:classify_and_fingerprint:BB='+str(BB_coordinates))
    return classification_list, fingerprint, BB_coordinates



def classify_and_fingerprint_dresses(path_to_image_file):
    #pdb.set_trace()
    print("in fingerPrint2:classify_and_fingerprint_dresses:path_to_image_file: " + str(path_to_image_file))
    REMOTE_FILE = False
    item_found = False
    fingerprint = ""
    classification_dict = {}
    #read arg from command line if none given to function
    if path_to_image_file is not None:
        PATH_TO_FILE = path_to_image_file
    #else:
        #wtf is supposed to happen here - who is calling this from command line??

    #we want to be able to read URL as well as local file path
    if "://" in path_to_image_file:
        FILENAME = path_to_image_file.split('/')[-1].split('#')[0].split('?')[0]
        res = urllib.urlretrieve (PATH_TO_FILE, FILENAME)
        PATH_TO_FILE = FILENAME
        REMOTE_FILE = True

    #pdb.set_trace()
    #main prog starts here
    img = cv2.imread(PATH_TO_FILE)
    roi = []
    classification_dict = classify.classify_image(img)
    BB_coordinates = classification_dict["/home/www-data/web2py/applications/fingerPrint/modules/dressClassifier001.xml"]
    if len(BB_coordinates) > 0:
        if BB_coordinates[0] is not None:
            item_found = True
            roi = crop_image_to_BB(img, BB_coordinates[0])
        else:
            roi=img  
            BB_coordinates = [[0,0,0,0]]
            print("in fingerPrint2.py: len(BB_coordinates)>0 but BB[0] is None -- REALLY WEIRD:")
            print(BB_coordinates)

    else:
#        roi = None  #if no BB was formed , don't return an answer!!!!
        print("in fingerPrint2.py:bad roi (could not crop?)-using entire img - len(BB)!>0")
	roi=img
        BB_coordinates = [[0,0,0,0]]
    if roi is not None:
        fingerprint = fp(roi)
    else:
        print("in fingerPrint2.py:bad roi (could not crop?) - using entire img (again)")
        fingerprint_length=56
        fingerprint=[0 for x in range(fingerprint_length)]
        fingerprint[0]=-1
        print("in fingerPrint2.py:fp="+str(fingerprint))

    if REMOTE_FILE:
        os.remove(PATH_TO_FILE)

    # right now, we're only doing shirts, so it's binary
    # 0 - nothing found, 1 - at least one shirt found.
    # even within this simplified case, still need to figure out
    # how to deal with multiple shirts in an image
    classification_list = []
    if item_found:
        classification_list.append(2)
    else:
        classification_list.append(0)

    print('in fingerPrint2.py:classify_and_fingerprint_dresses:fingerprint='+str(fingerprint))
    print('in fingerPrint2.py:classify_and_fingerprint_dresses:BB='+str(BB_coordinates))
    return classification_list, fingerprint, BB_coordinates




