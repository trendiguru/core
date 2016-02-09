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

def batch_crop(path):
    file_list = Utils.get_files_from_dir_and_subdirs(path)
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

def caffe_filelist(path):
    types = ['bubba_kush', 'hawg', 'hindu_kush', 'platinum_og2']
    file_list = Utils.get_files_from_dir_and_subdirs(path)
    print('n_files:'+str(len(file_list)))
    for file_n in file_list:
        print('file:'+str(file_n))
        if 'cropped' in file_n:
            print('got :'+str(file_n))
            mpath=os.path.split(file_n)[0]
            containing_folder=os.path.split(mpath)[0]
            print('mpath {0} containing {1}'.format(mpath,containing_folder))
#            tail=os.path.split[1]
            file_class = None
            for i in range(0,len(types)):
                if types[i] in file_n:
                    file_class = i
            if file_class is not None:
                print('{0} is class {1} :'.format(file_n,file_class))


if __name__=="__main__":
    caffe_filelist('/home/jr/Pictures/nugs')