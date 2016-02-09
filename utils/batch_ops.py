__author__ = 'jeremy'
import os
import cv2
from trendi import Utils

dir = '/home/jr/Pictures/'

'''
caffe image db usage
http://stackoverflow.com/questions/31427094/guide-to-use-convert-imageset-cpp

Labels: create a text file (e.g., /path/to/labels/train.txt) with a line per input image . For example:

img_0000.jpeg 1
img_0001.jpeg 0
img_0002.jpeg 0

In this example the first image is labeled 1 while the other two are labeled 0.
Convert the dataset

~$ GLOG_logtostderr=1 $CAFFE_ROOT/build/tools/convert_imageset \
    --resize_height=200 --resize_width=200 --shuffle  \
    /path/to/jpegs/ \
    /path/to/labels/train.txt \
    /path/to/lmdb/train_lmdb

Command line explained:

GLOG_logtostderr flag is set to 1 before calling convert_imageset indicates the logging mechanism to redirect log messages to stderr.
--resize_height and --resize_width resize all input images to same size 200x200.
--shuffle randomly change the order of images and does not preserve the order in the /path/to/labels/train.txt file.
Following are the path to the images folder, the labels text file and the output name. Note that the output name should not exist prior to calling convert_imageset otherwise you'll get a scary error message.

Other flags that might be useful:

--backend - allows you to choose between an lmdb dataset or levelDB.
--gray - convert all images to gray scale.
--help - shows some help, see all relevant flags under Flags from tools/convert_imageset.cpp
'''
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

def generate_pics()

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