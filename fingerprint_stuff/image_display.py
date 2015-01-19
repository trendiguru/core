#!/usr/bin/env python
from __future__ import print_function
'''
===============================================================================
Interactive Image display
===============================================================================
'''
#for bonus points - display this in a dedicated window like the pyplot window shows x,y but show hsv too
#or like the skimage io.imshow function with histograms and all

import os
import cv2


############################################################################################################


BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness
#img2=cv2.imread('~/Dropbox/projects/clothing_recognition/Images/test/images_003.jpeg')
img=[0]
img2=[0]

def onmouse(event,x,y,flags,param):
    global img,hsv,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

    mystr=str(x)+','+str(y)+' r:'+str(img[y,x,0])+' g:'+str(img[y,x,1])+' b:'+str(img[y,x,2])+' h:'+str(hsv[y,x,0])+' s:'+str(hsv[y,x,1])+' v:'+str(hsv[y,x,2])
    print(mystr+'\r',end='')


    # Draw Rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (ix,iy,abs(ix-x),abs(iy-y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (ix,iy,abs(ix-x),abs(iy-y))
        rect_or_mask = 0
#        print " Now press the key 'n' a few times until no further change"
#        print " press Esc when done "

    # draw touchup curves

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print (str(x)+','+str(y))
        else:
            drawing = True
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

# Loading images
def load(name):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over,hsv

    print('in gc_int:'+name)
    if len(name) ==0:
        name ='../cpp/lena.jpg'
    img = cv2.imread(name)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.namedWindow('input', cv2.WINDOW_AUTOSIZE)
    #cv, 'CV_WINDOW_NORMAL', cv.CV_WINDOW_AUTOSIZE

    cv2.setMouseCallback('input',onmouse)
    cv2.moveWindow('input',img.shape[1]+10,90)
    while(1):
        cv2.imshow('input',img)
#plt.imshow(img)
   #     cv2.imshow('copy',img2)
        k = 0xFF & cv2.waitKey(1)

    # key bindings
        if k == 27:         # esc to exit
            break
        elif k == ord('0'): # BG drawing
           # print " mark background regions with left mouse button"
            value = DRAW_BG
    print('shapes')
    imgplot = plt.imshow(img)


imagedir='/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/Backend/code/classifier_stuff/images/cjdb/hat'
#imagedir='/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/Backend/code/fingerprint_stuff'

filelist=[]
Nfile=1
print('dir is:'+imagedir)
for dirName, subdirList, fileList in os.walk(imagedir, topdown=True):
    #        print('dir:'+dirName+' sublist:'+str(subdirList)+' filelist:'+str(fileList))
    for file in fileList:
        filelist.append(os.path.join(imagedir,file))
        #print('file:'+file)
print(str(len(filelist))+' files in dir '+imagedir)
print('file:'+filelist[Nfile])
load(filelist[Nfile])


