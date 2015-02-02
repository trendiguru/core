__author__ = 'jeremy'
#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import Tkinter
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import subprocess
#from Tkinter import *
import tkMessageBox
#import Image
#import ImageT

def onmouse(event,x,y,flags,param):
    global img,hsv,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

    print (str(x)+','+str(y)+' r:'+str(img[x,y,0])+' g:'+str(img[x,y,1])+' b:'+str(img[x,y,2])+' h:'+str(hsv[x,y,0])+' s:'+str(hsv[x,y,1])+' v:'+str(hsv[x,y,2]))


class simpleapp_tk(Tkinter.Tk):
    def __init__(self,parent):
        Tkinter.Tk.__init__(self,parent)
        self.parent = parent
        self.initialize()

    def onmouse(event,x,y,flags,param):
        global img,hsv,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

        print (str(x)+','+str(y)+' r:'+str(img[x,y,0])+' g:'+str(img[x,y,1])+' b:'+str(img[x,y,2])+' h:'+str(hsv[x,y,0])+' s:'+str(hsv[x,y,1])+' v:'+str(hsv[x,y,2]))


    def initialize(self):
        global img
        global filelist
        global Nfile
        global min
        global max
        global step
        global imagedir
        global p1,p2,p3,p4,p5,p6,p7,p8,p9,p10
        global blobParamsList
        global blobParamsMax
        global blobParamsMin
        global blobParamNames
        global threshParamsList
        global threshParamsMax
        global threshParamsMin
        global threshParamNames
        global Ncols
        global Imrow
        global blobVar
        global contVar
        global fvariable
        global thresholdType
        global Im2col #=Ncols/2
        global currentImg
        global originalImg
        global hsvT

        cv2.setMouseCallback('input',onmouse)
        print('mouse call back set')

        filelist=[]
        Nfile=0
        Nfile=0
        p1=100
        p1min=0
        p1max=255


#   BLOB PARAMS
#    params.minThreshold = p1  #0
#    params.maxThreshold = 230   #1
#    params.thresholdStep=10   #2
#    params.minDistBetweenBlobs = 10.0  #3
    #params.minRepeatability = 1      #4
 #   params.minArea = 100.0            #5
 #   params.maxArea = 5000.0          #6
#    params.filterByInertia = Falsee
#    params.filterByConvexity = False
#    params.filterByColor = False
#    params.filterByCircularity = False
#    params.filterByArea = True

        paramsStart=[0,  2000, 5,  0, 7,  1,    50000]
        paramsList=paramsStart
        paramsMin=  [0,    0,    1,   0 ,  1,   1,      0]
        paramsMax=  [255,10000,1000,1000,1000,1000000,100000]
        paramNames=['minThresh','maxThresh','threshStep','minDistbetBlobs','minRepeatability','minArea','maxArea']

        threshParamsList=[100,    10,             2]
        threshParamsMax=[255,     1000,          1000]
        threshParamsMin=[0,       0,             0 ]
        threshParamNames=['thresh','neighborhood','unknown']

        imagedir='/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/Backend/code/classifier_stuff/images/cjdb/hat'
        #imagedir='images/cjdb/hat'
        Imrow=4
        print('starting(initialize)')
        self.grid()

        filelist=[]
       # print('dir is:'+imagedir)
 #       command_string='ls -l '+imagedir
  #      print(command_string)
   #     subprocess.call(command_string, shell=True)
        for dirName, subdirList, fileList in os.walk(imagedir, topdown=True):
    #        print('dir:'+dirName+' sublist:'+str(subdirList)+' filelist:'+str(fileList))
            for file in fileList:
                filelist.append(os.path.join(imagedir,file))
                #print('file:'+file)
        print(str(len(filelist))+' files in dir '+imagedir)
        image = Image.open(filelist[Nfile])
        originalImg=cv2.imread(filelist[Nfile])
        currentImg=originalImg.copy()

        Ncols=8
        #row 0Enter text here.
        r=0
        #string putput
        self.labelVariable = Tkinter.StringVar()
        label = Tkinter.Label(self,textvariable=self.labelVariable,anchor="w",fg="white",bg="blue")
        label.grid(column=0,row=r,columnspan=Ncols,sticky='EW')
        self.labelVariable.set(u"Hello !")


        #row 1
        r=1
       #string input
        self.entryVariable = Tkinter.StringVar()
        self.entry = Tkinter.Entry(self,textvariable=self.entryVariable)
        self.entry.grid(column=0,row=r,sticky='EW',columnspan=2)
        self.entry.bind("<Return>", self.OnPressEnter)
        self.entryVariable.set(u"Enter text here.")

#file - next and previous file
        #button1
        button1 = Tkinter.Button(self,text=u"prevfile",command=self.OnFileButtonDownClick)
        button1.grid(column=1,row=r,columnspan=1)
        #button2
        button2 = Tkinter.Button(self,text=u"nextfile",command=self.OnFileButtonUpClick)
        button2.grid(column=2,row=r,columnspan=1)


#function to use - dropdown list with one choice allowed
        fvariable = Tkinter.StringVar(self)
        fvariable.set("blob") # default value
        w = Tkinter.OptionMenu(self, fvariable, "blob", "contour", "houghlines")
        w.grid(column=3,row=r,columnspan=1)
        v=fvariable.get()
        print('func:'+str(fvariable)+' v:'+str(v))


#hsv : dropdown list with one choice allowed
        hsvT = Tkinter.StringVar(self)
        hsvT.set("H") # default value
        w = Tkinter.OptionMenu(self, hsvT, "H", "S", "V")
        w.grid(column=4,row=r,columnspan=1)
        thresholdType=hsvT.get()
        v=hsvT.get()
        print('func:'+str(hsvT)+' v:'+str(v))

#hsv 'go' button
        print('hsv:'+str(v))
        button = Tkinter.Button(self,text='do HSV',command=lambda:self.Dohsv(currentImg))
        button.grid(column=5,row=r)

#reset button
        button = Tkinter.Button(self,text='reset',command=self.reset)
        button.grid(column=6,row=r)


        #row 2
        #button1
        #need lambda here due to something having to do w callback http://stackoverflow.com/questions/4370160/tkinter-button-command-being-called-automatically
        r=2
        i=0
        n=0

        colNum=0

       # w = Tkinter.Scale(self, from_=0, to=200, orient=Tkinter.HORIZONTAL, command=self.BlobSliders)
        w.grid(column=colNum,row=r)

#        button1 = Tkinter.Button(self,text=str(i)+'d',command=lambda:self.OnButtonClick(0))
#        button1.grid(column=colNum,row=r)
#        button2 = Tkinter.Button(self,text=str(i)+'u',command=lambda:self.OnButtonClick(1))
#        button2.grid(column=colNum+1,row=r)



        colNum=colNum+2
        i=i+1
        button3 = Tkinter.Button(self,text=str(i)+'d',command=lambda:self.OnButtonClick(2))
        button3.grid(column=colNum,row=r)
        button4 = Tkinter.Button(self,text=str(i)+'u',command=lambda:self.OnButtonClick(3))
        button4.grid(column=colNum+1,row=r)
        colNum=colNum+2
        i=i+1
        button5 = Tkinter.Button(self,text=str(i)+'d',command=lambda:self.OnButtonClick(4))
        button5.grid(column=colNum,row=r)
        button6 = Tkinter.Button(self,text=str(i)+'u',command=lambda:self.OnButtonClick(5))
        button6.grid(column=colNum+1,row=r)
        colNum=colNum+2
        i=i+1
        button = Tkinter.Button(self,text=str(i)+'d',command=lambda:self.OnButtonClick(6))
        button.grid(column=colNum,row=r)
        button = Tkinter.Button(self,text=str(i)+'u',command=lambda:self.OnButtonClick(7))
        button.grid(column=colNum+1,row=r)

#ROW 3
        colNum=0
        i=i+1
        r=3
        button = Tkinter.Button(self,text=str(i)+'d',command=lambda:self.OnButtonClick(8))
        button.grid(column=colNum,row=r)
        button = Tkinter.Button(self,text=str(i)+'u',command=lambda:self.OnButtonClick(9))
        button.grid(column=colNum+1,row=r)
        colNum=colNum+2
        i=i+1
        button = Tkinter.Button(self,text=str(i)+'d',command=lambda:self.OnButtonClick(10))
        button.grid(column=colNum,row=r)
        button = Tkinter.Button(self,text=str(i)+'u',command=lambda:self.OnButtonClick(11))
        button.grid(column=colNum+1,row=r)
        colNum=colNum+2
        i=i+1
        button = Tkinter.Button(self,text=str(i)+'d',command=lambda:self.OnButtonClick(12))
        button.grid(column=colNum,row=r)
        button = Tkinter.Button(self,text=str(i)+'u',command=lambda:self.OnButtonClick(13))
        button.grid(column=colNum+1,row=r)
        colNum=colNum+2
        i=i+1
        button = Tkinter.Button(self,text=str(i)+'d',command=lambda:self.OnButtonClick(14))
        button.grid(column=colNum,row=r)
        button = Tkinter.Button(self,text=str(i)+'u',command=lambda:self.OnButtonClick(15))
        button.grid(column=colNum+1,row=r)

#threshold params
#row 4
        r=4
        colNum=0
        print('func:'+str(fvariable)+' v:'+str(v))
        button = Tkinter.Button(self,text='t1d',command=lambda:self.OnThreshParamsButtonClick(0))
        button.grid(column=colNum,row=r)
        button = Tkinter.Button(self,text='t1u',command=lambda:self.OnThreshParamsButtonClick(1))
        button.grid(column=colNum+1,row=r)
        button = Tkinter.Button(self,text='t2d',command=lambda:self.OnThreshParamsButtonClick(2))
        button.grid(column=colNum+2,row=r)
        button = Tkinter.Button(self,text='t2u',command=lambda:self.OnThreshParamsButtonClick(3))
        button.grid(column=colNum+3,row=r)

#threshold type : dropdown list with one choice allowed
        threshT = Tkinter.StringVar(self)
        threshT.set("gaussian") # default value
        w = Tkinter.OptionMenu(self, threshT, "binary", "mean", "gaussian")
        w.grid(column=colNum+4,row=r,columnspan=1)
        thresholdType=threshT.get()

#do thresh
        button = Tkinter.Button(self,text='thresh',command=self.DoThreshold)
        button.grid(column=colNum+5,row=r)


        #image
        #ROW 5
        r=5
        Imrow=r
        Im2col=Ncols/2

        photo = ImageTk.PhotoImage(image)
        mod=image.mode
        print('orig mode'+str(mod))
      #  img = Tkinter.PhotoImage(file=filelist[0])
        label1 = Tkinter.Label(self, image=photo)
        label1.image = photo
        label1.grid(row = Imrow, column = 0, columnspan = Im2col, sticky=Tkinter.NW)

        photo = ImageTk.PhotoImage(image)
        mod=image.mode
        print('2nd img mode'+str(mod))
      #  img = Tkinter.PhotoImage(file=filelist[0])
        label1 = Tkinter.Label(self, image=photo)
        label1.image = photo
        label1.grid(row = Imrow, column = Im2col, columnspan = Im2col, sticky=Tkinter.NW)


 #       img = Tkinter.PhotoImage(file='output2.gif')
 #       label1 = Tkinter.Label(self, image=img)
 #       label1.image = img
#        fname = Tkinter.Canvas(bg='black',height=600,width=600)
#        fname.grid(column=0,row=2,columnspan=2)
        #fname.pack(side=TOP)

       #  fname.pack()
        self.grid_columnconfigure(0,weight=1)
        self.resizable(True,True)
        self.update()
        self.geometry(self.geometry())
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

    def reset(self):
        #assert isinstance(originalImg, object)
        global currentImg
        currentImg=originalImg.copy()
        self.displayImage(currentImg)


    def BlobSliders(self):
        global paramsList
        print('oh yeah:'+str(n)+'and '+str(w.get))
        #w = Tkinter.Scale(self, from_=0, to=200, orient=Tkinter.HORIZONTAL, command=lambda:self.BlobSliders(0))
        #w.grid(column=colNum,row=r)


    def OnButtonClick(self,n):
        global paramsList
        global paramsMax
        global paramsMin
        global paramNames
        global p1max
        global button1Val
        global p1
        global blobVar
        global contVar
        global fvariable
        global Im2col
        global Nfile
        global fvariable
        global currentImg
        global originalImg

        buttonNum=int(n/2)
        rem=n%2
        print('button '+str(n))
        print(paramNames[buttonNum]+')='+str(paramsList[buttonNum]))
        if rem==1:  #increment value of param
            if paramsList[buttonNum]<paramsMax[buttonNum]:
                paramsList[buttonNum]=paramsList[buttonNum]+1
        if rem==0: #decrement value of param
            if paramsList[buttonNum]>paramsMin[buttonNum]:
                paramsList[buttonNum]=paramsList[buttonNum]-1

        print('func:'+fvariable.get())
#        self.labelVariable.set( self.entryVariable.get()+" p1val="+str(p1) )
        self.labelVariable.set(paramNames[buttonNum]+"="+str(paramsList[buttonNum]) )
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

        f=fvariable.get()
        img=currentImg
        newImg = {
          'blob': lambda img: self.blob(),
          'contour': lambda img: self.contours(),
          'hough': lambda img: img - 2
        }[f](img)
 #       currentImg=newImg

        self.displayImage(newImg)
        use_visual_output=False
        if use_visual_output:
            print('blob display')
            cv2.imshow('orig', currentImg)
            #cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.waitKey(0)
            cv2.imshow('current', newImg)
            #cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.waitKey(0)

            cv2.destroyAllWindows()
        return currentImg

    def blur(self):
        blurred_image = cv2.GaussianBlur( img, (0,0), 1 )
        #blurred_image = img
        currentImg=blurred_image

    def Dohsv(self,img):
        #    img=cv2.imread(file)
        global currentImg
        global originalImg
        global newImg
        global hsvT
        v=hsvT.get()
        print('v='+str(v))
        img=currentImg

        print('size im:'+str(img.shape))
        if len(img.shape) == 2:  #image is a single plane
            print('you are trying to do something bad and wrong')
            img=originalImg

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        newimg = {
          'H': lambda img: hsv[:,:,0]*2,
          'S': lambda img: hsv[:,:,1],
          'V': lambda img: hsv[:,:,2]
        }[v](img)
        #img=hsv[:,:,0]*2  # to display hue in full grayscale range 0-255
        cv2.imwrite('output.jpg',newimg)
        self.displayImage(newimg)
        currentImg=newimg
        return(currentImg)

    def DoThreshold(self):
        global threshParamsList
        global thresholdType
        file=filelist[Nfile]
        print('dothresh on file:'+file)
        img=cv2.imread(file)
#        thresh = cv2.adaptiveThreshold(img,threshParamsList[0],255, cv2.THRESH_BINARY,threshParamsList[1],threshParamsList[2])
        if thresholdType==cv2.ADAPTIVE_THRESH_GAUSSIAN_C or thresholdType==cv2.ADAPTIVE_THRESH_MEAN_C:
            thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,threshParamsList[0],threshParamsList[1])
        else:
            ret,thresh = cv2.threshold(img,threshParamsList[0],255,cv2.THRESH_BINARY_INV)
        self.displayImage(thresh)
        return(thresh)

    def displayImage(self,img):
        global Im2col
        print('displayimage')

        #internal display
        r=cv2.imwrite('temp.jpg',img)
        tk_img = Image.open('temp.jpg')
        #tk_img=Image.fromarray(img) #.resize((100,360),Image.ANTIALIAS)
        tk_photo=ImageTk.PhotoImage(tk_img)
        label1 = Tkinter.Label(self, image=tk_photo)
        label1.image = tk_photo
        label1.grid(row = Imrow, column = Im2col, columnspan = Im2col, sticky=Tkinter.NW)
        self.update()

        #external display
        use_visual_output=False
        if use_visual_output:
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
#            cv2.imwrite('output.jpg',img)



    def OnThreshParamsButtonClick(self,n):
        global threshParamsList
        global threshParamsMax
        global threshParamsMin
        global threshParamNames
        global thresholdType

        buttonNum=int(n/2)
        rem=n%2
        print('button '+str(n))
        print(threshParamNames[buttonNum]+')='+str(threshParamsList[buttonNum]))
        if rem==1:  #increment value of param
            if threshParamsList[buttonNum]<threshParamsMax[buttonNum]:
                threshParamsList[buttonNum]=threshParamsList[buttonNum]+1
        if rem==0: #decrement value of param
            if threshParamsList[buttonNum]>threshParamsMin[buttonNum]:
                threshParamsList[buttonNum]=threshParamsList[buttonNum]-1

        print('tresholdType:'+thresholdType)
#        self.labelVariable.set( self.entryVariable.get()+" p1val="+str(p1) )
        self.labelVariable.set(threshParamNames[buttonNum]+"="+str(threshParamsList[buttonNum]) )
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)
        #doProcess(self)



    def OnFileButtonUpClick(self):
        global p1
        global p1max
        global button1Val
        global Nfile
        global currentImg
        global originalImg

        print('file button up')
        if Nfile<len(filelist):
            Nfile=Nfile+1
        print('Nfile:'+str(Nfile))
        self.labelVariable.set( self.entryVariable.get()+" p1val="+str(filelist[Nfile]) )
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

        originalImg=cv2.imread(filelist[Nfile])
        currentImg=originalImg.copy()
        img = Image.open(filelist[Nfile])
        photo2 = ImageTk.PhotoImage(img)
        label1 = Tkinter.Label(self, image=photo2)
        label1.image = photo2
        label1.grid(row = Imrow, column = 0, columnspan = Ncols, sticky=Tkinter.NW)
        self.update()


    def OnFileButtonDownClick(self):
        global p1
        global p1max
        global button1Val
        global Nfile
        global currentImg
        global originalImg

        print('file button down')
        if Nfile>0:
            Nfile=Nfile-1
        print('Nfile:'+str(Nfile))
        self.labelVariable.set( self.entryVariable.get()+" p1val="+str(filelist[Nfile]) )
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

        originalImg=cv2.imread(filelist[Nfile])
        currentImg=cv2.copy(originalImg)

        img = Image.open(filelist[Nfile])
        photo2 = ImageTk.PhotoImage(img)
        label1 = Tkinter.Label(self, image=photo2)
        label1.image = photo2
        label1.grid(row = Imrow, column = 0, columnspan = Ncols, sticky=Tkinter.NW)
        self.update()



    def OnPressEnter(self,event):
        self.labelVariable.set( self.entryVariable.get()+" (You pressed ENTER)" )
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

    def hist(img):
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


    def contours(img):
    #    im = cv2.imread('test.jpg')
        #imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        global threshParamsList

        print('doing contour')
        #ret,thresh = cv2.threshold(img,12,255,0)
        thresh = cv2.adaptiveThreshold(img,255,threshParamsList[0], cv2.THRESH_BINARY,threshParamsList[1],threshParamsList[2])
        use_visual_output=True
        if use_visual_output:
            print('contour display')
            cv2.imshow('thresh',thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(img, contours, -1, (255,255,255), 5)
        cv2.drawContours(thresh, contours, -1, (255,255,255), 5)


        for cnt in contours:
            epsilon = 0.1*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)

    #        if cv2.waitKey(0) & 0xFF == ord('q'):
     #           a=0

    def blob(img):
        global paramsList
        global Nfile
        global currentImg

        print('doing blob')
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = paramsList[0]
        params.maxThreshold = paramsList[1]
        params.thresholdStep= paramsList[2]
        params.minDistBetweenBlobs = paramsList[3]
        params.minRepeatability = paramsList[4]
        params.minArea = paramsList[5]
        params.maxArea = paramsList[6]
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByColor = False
        params.filterByCircularity = False
        params.filterByArea = True
        b = cv2.SimpleBlobDetector(params)

    #    blurred_image = cv2.GaussianBlur( img, (0,0), 1 )
    #    blurred_image = img
        #a=cv2.copy
        blurred_image=currentImg
        blob = b.detect(blurred_image)
     #   print(str(blob))
        for kp in blob:
            x = int(kp.pt[0])
            y = int(kp.pt[1])
            cv2.circle(blurred_image, (x, y), int(kp.size), (0, 255, 255))
    #        print('x:'+str(x)+' y:'+str(y))

        cv2.imwrite('output.jpg',blurred_image)

    #imshow(frame, cmap=cm.gray)
        use_visual_output=False
        if use_visual_output:
            print('blob display')
            cv2.imshow('img', blurred_image)
            #cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow('blurred',blurred_image)
    #    cv2.waitKey(0)
            cv2.imshow('cur',currentImg)
            cv2.waitKey(0)

        return(blurred_image)

    def hough(img):
        edges = cv2.Canny(blurred_image,50,150,apertureSize = 3)
        lines = cv2.HoughLines(edges,1,np.pi/180, 200)

        print(lines)
        if lines is not None:
            for rho,theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(blurred_image,(x1,y1),(x2,y2),(0,0,0),2)


    def doBlobDetect(self,img):

        global Nfile
        global fvariable
        global currentImg
        global originalImg

        #file=filelist[Nfile]
        #print('file:'+file)

        f=fvariable.get()
        result = {
          'blob': lambda img: blob(img),
          'contour': lambda img: contours(img),
          'hough': lambda img: img - 2
        }[f](img)

        img=cv2.imread('output.jpg')
        self.displayImage(img)
    #    img = Image.open('output.jpg')
    #    photo2 = ImageTk.PhotoImage(img)
    #    label1 = Tkinter.Label(self, image=photo2)
    #    label1.image = photo2
    #    label1.grid(row = Imrow, column = 0, columnspan = Ncols, sticky=Tkinter.NW)
    #    self.update()


global img
global min
global max
global step
global imagedir
global filelist

imagedir='/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/Backend/code/classifier_stuff/images/cjdb/hat'
#imagedir='images/cjdb/hat'
min=0
max=255
step=10
button1Val=0
button2Val=0
global p1,p2,p3,p4,p5,p6,p7,p8,p9,p10
global p1min
global p1max
global Nfile
global blobVar
global contVar
global fvariable
global thresholdType

global blobParamsList
global blobParamsMax
global blobParamsMin
global blobParamNames

global threshParamsList
global threshParamsMax
global threshParamsMin
global threshParamNames
global currentImg
global originalImg

Nfile=0
p1=100
p1min=0
p1max=255
global Ncols
Ncols=6
global Imrow
global Im2col #=Ncols/2

Imrow=4
if __name__ == "__main__":
    app = simpleapp_tk(None)
    app.title('my application')
    app.mainloop()

'''
n=0
        colNum=0
        buttons=[]
        for i in range(0,int(Ncols/2)):
            buttons.append(Tkinter.Button(self,text=str(i)+'d',command=lambda:self.OnButtonClick(n)))
            buttons[n].grid(column=colNum,row=r)
            buttons.append( Tkinter.Button(self,text=str(i)+'u',command=lambda:self.OnButtonClick(n+1)))
            buttons[n+1].grid(column=colNum+1,row=r)
            print('button:'+str(i)+' n:'+str(n))
            n=n+2
            colNum=colNum+2
        #row 3
        r=3
        colNum=0
        for i in range(0,int(Ncols/2)):
            button = Tkinter.Button(self,text=str(i)+'d',command=lambda:self.OnButtonClick(n))
            button.grid(column=colNum,row=r)
            button = Tkinter.Button(self,text=str(i)+'u',command=lambda:self.OnButtonClick(n+1))
            button.grid(column=colNum+1,row=r)
            print('button:'+str(i)+' n:'+str(n))
            n=n+2
            colNum=colNum+2

'''