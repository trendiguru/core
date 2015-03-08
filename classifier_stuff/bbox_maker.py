#!/usr/bin/env python


import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys


#def grabcutInteractive(img):
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
n_rects=0
allRects=[]
rect = [0,0,1,1]
r=[]
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness
#img2=cv2.imread('~/Dropbox/projects/clothing_recognition/Images/test/images_003.jpeg')
img=[0]
img2=[0]
ix=0
iy=0

def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over
    global r,ww,hh
#    r=[1,2,3,4]
    # Draw Rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        rectangle = True
        ix,iy = x,y
        cv2.imshow('input',img)

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),BLUE,1)
            rect = (ix,iy,abs(ix-x),abs(iy-y))
            rect_or_mask = 0
            if x<ix:
                r[0]=x
                r[2]=ix-x
            else:
                r[0]=ix
                r[2]=x-ix
            if y<iy:
                r[1]=y
                r[3]=iy-y
            else:
                r[1]=iy
                r[3]=y-iy
            if r[0]<0:
                r[0]=0
            if r[0]+r[2]>ww:
                r[2]=ww-r[0]-1
            if r[1]<0:
                r[1]=0
            if r[1]+r[3]>hh:
                r[3]=hh-r[1]-1
    elif event == cv2.EVENT_LBUTTONUP:
        rectangle = False
        rect_over = True
        rect = (ix,iy,abs(ix-x),abs(iy-y))
        if x<ix:
            r[0]=x
            r[2]=ix-x
        else:
            r[0]=ix
            r[2]=x-ix
        if y<iy:
            r[1]=y
            r[3]=iy-y
        else:
            r[1]=iy
            r[3]=y-iy
        if r[0]<0:
            r[0]=0
        if r[0]+r[2]>ww:
            r[2]=ww-r[0]-1
        if r[1]<0:
            r[1]=0
        if r[1]+r[3]>hh:
            r[3]=hh-r[1]-1
        rect_or_mask = 0
        cv2.rectangle(img,(ix,iy),(x,y),BLUE,1)
        cv2.rectangle(img,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),RED,1)
        print('r=:'+str(r))
    # draw touchup curves

# Loading images
def bbox_maker(imgname,datafilename):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over
    global hh,ww,d

    global n_rects,r
    global allRects
    print " press 'n' if there is another instance "
    print " press 'r' to reset "
    print " press 'space' to goto next image "
    print " press Esc when done "


    n_rects=0
    allRects=[]
    r=[1,2,3,4]
    print('file in:'+imgname)
    if len(imgname) ==0:
        name ='../cpp/lena.jpg'
    img = cv2.imread(imgname)
    hh,ww,d=img.shape
    img2=img.copy()


# input and output windows
#    cv2.namedWindow('output')
    cv2.namedWindow('input')
    cv2.setMouseCallback('input',onmouse)
    cv2.moveWindow('input',img.shape[1]+10,90)

    print()
    print " Instructions :"
    print " Draw a rectangle around the object using right mouse button "
    while(1):

 #       cv2.imshow('output',output)
        cv2.imshow('input',img)
   #     cv2.imshow('copy',img2)
        k = 0xFF & cv2.waitKey(1)

    # key bindings
        if k == 27:         # esc to exit
            break
        elif k == ord(' '): # BG drawing
            if n_rects>0:
                if r != allRects[n_rects-1]:
                    if r[0]<0:
                        r[0]=0
                    if r[1]<0:
                        r[1]=0
                    if r[0]+r[2]>ww:
                        r[2]=ww-r[0]-1
                    if r[1]+r[3]>hh:
                          r[3]=hh-r[1]-1
                    cv2.rectangle(img,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),RED,1)
                    allRects.append(r)
                    n_rects=n_rects+1
            elif n_rects==0:
                n_rects=n_rects+1
                allRects.append(r)
            print('all rects '+str(allRects))
            f = open(datafilename,'a+')
            strn=imgname+'  '+str(n_rects)+'  '
            for r in allRects:
                strn=strn+str(r[0])+' '+str(r[1])+' '+str(r[2])+' '+str(r[3])+'   '
            strn=strn+'\n'
            f.write(strn)
            f.close() # you can omit in most cases as the destructor will call if
            print(strn)
            return
        elif k == ord('n'): # save
            cv2.rectangle(img,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),RED,1)
            n_rects=n_rects+1
            print('nrect '+str(n_rects))
 #           print('all '+str(allRects))
            allRects.append(r)

            print('rect '+str(r))
            print('all '+str(allRects))
#file format:
#img/img1.jpg  1  140 100 45 45
#img/img2.jpg  2  100 200 50 50   50 30 25 25

        elif k == ord('r'): # reset everything
            print "resetting \n"
            rect = (0,0,1,1)
            allRects=[]
            n_rects=0
