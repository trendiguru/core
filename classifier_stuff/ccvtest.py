import sys
from PIL import Image
from ccv import PY_CCV_IO_GRAY, DenseMatrix, ClassifierCascade, detect_objects
import cv2
import ccv
import os
import numpy as np


def face(matrix):
    cascade = ClassifierCascade()
    cascade.read('/home/jeremy/sw/ccv/samples/face')
#cascade.read(sys.argv[2])

    print ccv.detect_objects(matrix, cascade, 1)


def get_image(matrix,rows,cols,channels=3):
    blank_image = np.zeros((rows,cols,channels), np.uint8)

    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                blank_image[i,j,k]=matrix.nth_pixel(i,j,k)
    cv2.imshow('out',blank_image)
    cv2.waitKey(0)

    return(blank_image)

def filewrite():
#    img_arr=np.mgrid[0:3,0:5]
    r=np.arange(5)
#    img_arr_2d=np.array([r,r+1,r+2])
    n_cols=10
#    img_arr_2d=np.arange(n_cols)
#    for i in range(0,n_cols):
#        img_arr_2d[i]=r+i
#    img_arr=np.array([[img_arr_2d],[img_arr_2d+10],[img_arr_2d+20]])

#    img_arr=np.mgrid[0:5]
    height=10
    width=20
    blank_image = np.zeros((height,width,3), np.uint8)
    #This initialises an RGB-image that is just black. Now, for example, if you wanted to set the left half of the image to blue and the right half to green , you could do so easily:

    for i in range(0,height):
        for j in range(0,width):
            blank_image[i,j,0] = i+2*j      # (B, G, R)
            blank_image[i,j,1] = i+2*j+41      # (B, G, R)
            blank_image[i,j,2] = i+2*j+82      # (B, G, R)
    img_arr=blank_image

 #   print('im:'+str(img_arr))
    cv2.imwrite('testfile.jpg',img_arr)
    cv2.imshow('out',img_arr)
    cv2.waitKey(0)

#import constants
filewrite()
matrix = ccv.DenseMatrix()

a = os.getcwd()
print('cwd:'+str(a))
b= os.path.dirname(__file__)
print('project dir:'+str(b))
#cv2_im = cv2.imread('../images/female1/jpg')
#cv2_im = cv2.imread('/home/jeremy/tg1/images/female1.jpg')
cv2_im = cv2.imread('/home/jeremy/tg1/images/faces-300x300.jpg')
#cv2_im = cv2.imread('testfile.jpg')

#cv2_im = cv2.imread(sys.argv[1])
#gray_image = cv2.cvtColor(cv2/homeim, cv2.COLOR_BGR2GRAY)

pil_im = Image.fromarray(cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB))
#pil_im = Image.fromarray(cv2_im)

#pil_im = Image.open(sys.argv[1])
st=bytearray(pil_im.tostring())
st_cv=bytearray(cv2_im.tostring())
print('len:{0} w {1} h {2} mode {3}'.format(str(len(st)),str(pil_im.size[0]),str(pil_im.size[1]),str(pil_im.mode)))
#matrix.set_buf(st, pil_im.mode, pil_im.size[0], pil_im.size[1], PY_CCV_IO_GRAY)
print('cv len:{0} w {1} h {2} '.format(str(len(st_cv)),str(cv2_im.shape[0]),str(cv2_im.shape[1]),str(pil_im.mode)))
matrix.set_buf(st_cv,'RGB',cv2_im.shape[0], cv2_im.shape[1])
#matrix.set_buf(st, pil_im.mode, pil_im.size[0], pil_im.size[1])
#print matrix
a=matrix.first_pixel()
print('first pixel:'+str(a))
rows=cv2_im.shape[0]
cols=cv2_im.shape[1]
channels=3
get_image(matrix,rows,cols,channels)
face(matrix)
#print matrix._matrix
#        ccv_dense_matrix_t* image = 0;
# 6        ccv_read(argv[1], &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
# 7        ccv_write(image, argv[2], 0, CCV_IO_PNG_FILE, 0);
#ccv.ccv_write(pil_im,"testout.png", 0, CCV_IO_PNG_FILE, 0);

#matrix.set_file(sys.argv[1], PY_CCV_IO_GRAY)
