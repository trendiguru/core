__author__ = 'jeremy'

import scipy.io as sio
import cv2
import os
import numpy as np

def load_malldataset(filename='/data/jeremy/image_dbs/hls/mall_dataset/mall_gt.mat'):
    mat = sio.loadmat(filename)
    for k,v in mat.iteritems():
        print(k)

#    print(mat['__globals__'])
#    print(mat['__header__'])
#    print(mat['__version__'])
    #print(mat['count'])
    #raw_input('ret to cont')
    #print(mat['frame'])

    return mat['frame']


def inspect_matlab_dataset(filename='/data/jeremy/image_dbs/hls/mall_dataset/mall_gt.mat'):
    mat = sio.loadmat(filename)
    for k,v in mat.iteritems():
        print(k)
    raw_input('return to continue')
    return mat

def show_rects(n,head_positions):
    name = '/data/jeremy/image_dbs/hls/mall_dataset/frames/seq_%06d'%n+'.jpg'
    if not os.path.isfile(name):
        print('{} is not a file'.format(name))
        exit()
    print(name)
    img_arr = cv2.imread(name)
    xsize = 5
    im_height,im_width = img_arr.shape[0:2]
    for i in range(len(head_positions)):
        print head_positions[i]
        x=int(head_positions[i][0])
        y=int(head_positions[i][1])
#        y=im_height-y
        cv2.rectangle(img_arr,(x-xsize,y-xsize),(x+xsize,y+xsize),color=[255,0,100],thickness=2)
    cv2.imshow('img',img_arr)

    cv2.waitKey(0)
    return(img_arr)

if __name__=="__main__":
    f1 = inspect_matlab_dataset(filename='/data/jeremy/image_dbs/hls/mall_dataset/mall_feat.mat')

    frame = load_malldataset()
    print frame.shape
    n=1
    name = '/data/jeremy/image_dbs/hls/mall_dataset/frames/seq_%06d'%n+'.jpg'
    if not os.path.isfile(name):
        print('{} is not a file'.format(name))
        exit()
    print(name)
    img = cv2.imread(name)
    height , width , layers =  img.shape
    video = cv2.VideoWriter('video.xvid',-1,1,(width,height))


    for n in range(1,200):
        framedat = frame[0,n]
        head_outer = framedat[0,:][0]
        head_positions = head_outer[0]
        img=show_rects(n,head_positions)

        video.write(img)
        cv2.destroyAllWindows()
        video.release()