__author__ = 'jeremy'

import matplotlib as plt
import numpy as np
import cv2
import string
import logging
import time
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from Tkinter import Tk
from tkFileDialog import askopenfilename
#from matplotlib import style

def get_file():

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    print(filename)
    img_array=cv2.imread(filename)
   # cv2.imshow('image',img_array)
   # cv2.waitKey(100)
    return(img_array)



img_array=get_file()
print(str(img_array[10,1:]))
plt.figure('figure1')

imgplot = plt.imshow(img_array)
imgplot.set_cmap('spectral')
cv2.circle(img_array,(226,583),20,(30,20,10),thickness=5)
cv2.circle(img_array,(340,270),30,(30,20,10),thickness=5)
cv2.circle(img_array,(440,240),40,(30,20,10),thickness=5)
cv2.circle(img_array,(450,540),50,(30,20,10),thickness=5)
cv2.circle(img_array,(330,960),60,(30,20,10),thickness=5)

grid()

rows,cols,depth = img_array.shape
matrix = np.float32([[0.5,0.5,100],[0.5,0.7,200]])
matrix=cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
plt.figure('gif2')
#matrix=np.matrix('1 0 0;0 1 0; 0 0 1')
img_array2=img_array.copy()
img_array2=cv2.warpAffine(img_array,matrix,(cols,rows))
print(str(img_array2[10,1:]))
imgplot = plt.imshow(img_array2)
imgplot.set_cmap('spectral')

grid(True)


fig=plt.figure('figure3')
ax=fig.add_subplot(111,projection='3d')
ax.grid(True)
xsout=[318,177,221,450,660]
ysout=[634,340,242,450,820]
sizes=np.divide(ysout,2)
print(sizes)
ax.scatter([226,450,340,330,440],[583,540,270,960,240],xsout,s=sizes)
#style.use('dark_background)')
plt.show()