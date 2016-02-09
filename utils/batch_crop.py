__author__ = 'jeremy'
import os
import cv2
from trendi import Utils

dir = '/home/jr/Pictures/'

def crop(img_arr,bb):
    x=bb[0]
    y=bb[1]
    w=bb[2]
    h=bb[3]
    #roi_gray = gray[y:y + h,  x:x + w]
    cropped = img_arr[y:y+h,x:x+w]
    return cropped

if __name__=="__main__":

    file_list = Utils.get_directory_structure('/home/jr/Pictures/nugs')
    print('n_files:'+str(len(file_list)))
    #print('filelist:'+str(file_list))
    bb = [350,150,450,550]
    for file_n in file_list:
#        if not '_cropped' in file_n:
        if not 0 : #skip the skipping
            print('file:'+str(file_n))
            img_arr = cv2.imread(file_n)
            cropped = crop(img_arr,bb)
            name=file_n.split('.')[0]
            suffix=file_n.split('.')[1]
            name = name+'_cropped.'+suffix
            print('newname:'+str(name))
            cv2.imwrite(name,cropped)
        else:
            print('skipping '+str(file_n)+', already done')