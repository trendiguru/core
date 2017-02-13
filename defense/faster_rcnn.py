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
            im_h,im_w=im.shape[0:2]
            bbx=int(row["boundingBoxX"])
            bby=int(row["boundingBoxY"])
            bbw=int(row["boundingBoxWidth"])
            bbh=int(row["boundingBoxHight"])
            x=max(0,bbx)
            y=max(0,bby)
            x2=min(im_h,bbx+bbw)
            y2=min(im_w,bby+bbh)
            bb = [x,y,x2,y2]
            print('im_w {} im_h {} bb {} x {} y {} w {} h {}'.format(im_w,im_h,bb,bbx,bby,bbw,bbh))
            bb_img = im[bb[0]:bb[2],bb[1]:bb[3]]
            savename = filename.replace('.jpg',str(bb[0])+'_'+str(bb[1])+'_'+str(bb[2])+'_'+str(bb[3])+'.jpg')
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

