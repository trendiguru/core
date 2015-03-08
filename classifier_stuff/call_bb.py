__author__ = 'jeremy'
#done 19.8.14
#basedir = "/home/jeremy/Dropbox/projects/clothing_recognition/Images/pants/dress pants/"
#datafilename = "dressPantsPositives.txt"

import numpy as np
import cv2
import dateutil
import pyparsing
import six
import os
import sys

import sqlite3
import scipy
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import bbox_maker
from bbox_maker import bbox_maker
import Tkinter, tkFileDialog

basedir = "/home/jeremy/Dropbox/projects/clothing_recognition/Images/shirts/BUTTONSHIRT/"
basedir = "classifier_stuff/dresses_111014"
datafilename = "bb_dresses_111014.txt"
datafilename = ''

redo_files=['/home/jeremy/Dropbox/projects/clothing_recognition/Images/shirts/BUTTONSHIRT/images_399.jpeg',
'/home/jeremy/Dropbox/projects/clothing_recognition/Images/shirts/BUTTONSHIRT/images_131.jpeg']
redo_files=[]


# ======== Select directory to bound
root = Tkinter.Tk()
basedir = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select a directory for adding bounding boxes')
if len(basedir ) > 0:
    print "You chose %s" % basedir

if datafilename == '':
    # ======== Select a file for opening:
    root = Tkinter.Tk()
    #root.withdraw()
    datafilename = tkFileDialog.askopenfile(parent=root,mode='rb',title='Choose a file to hold bb info (or existing)')

if datafilename != None:
    data = datafilename.read()
    datafilename.close()
    print "I got %d bytes from this file." % len(data)

if redo_files!=[]:
    for filename in redo_files:
        img=bbox_maker(filename,datafilename)

print('base directory:'+basedir)
for dirname, dirnames, filenames in os.walk(basedir):
    n = 0
    print("dir=" + dirname)
    for filename in filenames:
        fullname = os.path.join(dirname, filename)
        print fullname

        if (filename[len(filename)-3:len(filename)]=='jpg' or filename[len(filename)-4:len(filename)]=='jpeg') :
            print('hit '+filename)
            n = n + 1
            print("n=" + (str(n)))
            img=bbox_maker(fullname,datafilename)

