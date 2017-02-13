__author__ = 'jeremy'


import csv
import os
import cv2

def read_csv(filename='/data/olympics/olympicsfull.csv'):
    filename = "olympicsfull.csv"
    unique_descs=[]
    with open(filename, "rb") as file:
        reader = csv.DictReader(file)
        for row in reader:
            print row
            filename = row['path']
            im = cv2.imread(filename)
            if im is None:
                print('couldnt read '+filename)
                continue
            bb = [row["boundingBoxX"],row["boundingBoxY"],row["boundingBoxWidth"],row["boundingBoxHight"]]
            bb_img = im[bb[0]:bb[0]+bb[2],bb[1]+bb[3]]
            savename = filename.replace('.jpg',str(bb[0])+'_'+str(bb[1])+'_'+str(bb[2])+'_'+str(bb[3])+'_')
            cv2.imwrite(savename,bb_img)

            lblname = row['description']+'_labels.txt'
            with open(lblname,'a') as fp:
                fp.write(savename,1)
                fp.close()

            if not row['description'] in unique_descs:
                unique_descs.append(row['description'])
                print unique_descs



def make_rcnn_trainfile(dir,filter='.jpg',trainfile='train.txt'):
    '''
    https://github.com/zeyuanxy/fast-rcnn/tree/master/help/train
    better yet ssee https://github.com/deboc/py-faster-rcnn/tree/master/help
    
    :param dir:
    :param filter:
    :param trainfile:
    :return:
    '''
    files = [f for f in os.listdir(dir) if filter in f]
    with open(trainfile,'w') as fp:
        for f in files:
            stripped = f.replace('.jpg','')
            fp.write(stripped+'\n')
        fp.close()


	    # Do awesome things with row["path"], row["boundingBoxX"], etc..."
		# DictReader autommatically turn the row into a dict.

