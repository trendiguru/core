__author__ = 'jeremy'


import csv
import os
import cv2
import Image


def read_csv(csvfile='/data/olympics/olympicsfull.csv',imagedir='/data/olympics/olympics',visual_output=False,confidence_threshold=0.9,manual_verification=True):
    ''''
    ok the bbx, bby , bbwidth, bbight are in % of image dims, and bbwidth/hight are not width/hight but
    rather x2,y2 of the bb
    '''
    #filename = "olympicsfull.csv"
    unique_descs=[]
    all_bbs=[]
    if manual_verification:  #write a description line in verified objects file
        verified_objects_file = 'verified_objects.txt'
        with open(verified_objects_file,'a') as fp:
            line = '#filename\tdescription\tx\ty\tw\th\n'
            fp.write(line)
            fp.close()

    with open(csvfile, "rb") as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row['path']
            if float(row['confidence'])<confidence_threshold:
                print('too low confidence '+str(row['confidence']))
                continue
            if imagedir is not None:
                full_name = os.path.join(imagedir,filename)
            else:
                full_name = filename
            im = cv2.imread(full_name)
            if im is None:
                print('couldnt read '+filename)
                continue
            print row

            im_h,im_w=im.shape[0:2]
            factor = 1
            dx = int(float(im_w)/factor)
            dy = int(float(im_h)/factor)
            im = cv2.resize(im,(dx,dy))
            im_h,im_w=im.shape[0:2]
            bbx=int(row["boundingBoxX"])*im_w/100
            bby=int(row["boundingBoxY"])*im_h/100
            bbw=int(row["boundingBoxWidth"]) #* (im_w-bbx)/100
            bbh=int(row["boundingBoxHight"]) #* (im_h-bby)/100
            bbx2=int(row["boundingBoxWidth"])*im_w/100 #* (im_w-bbx)/100
            bby2=int(row["boundingBoxHight"])*im_h/100 #* (im_h-bby)/100
            x=max(0,bbx)
            y=max(0,bby)
            x2=min(im_h,bbx+bbw)
            y2=min(im_w,bby+bbh)
            bb = [x,y,bbx2-bbx,bby2-bby]
            all_bbs.append(bb)
            if bb[2]==0 or bb[3] == 0 :
                print('got 0 width or height')
                continue
            object = row['description']
            print('im_w {} im_h {} bb {} object {} bbx {} bby {}'.format(im_w,im_h,bb,object,row['boundingBoxX'],row['boundingBoxY']))
            bb_img = im[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
            savename = filename.replace('.jpg','_'+str(bb[0])+'_'+str(bb[1])+'_'+str(bb[2])+'_'+str(bb[3])+'.jpg')
            if visual_output:
#                cv2.imwrite(savename,bb_img)
                cv2.rectangle(im,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),color=[255,0,100],thickness=2)
                cv2.imshow('full',im)
                #cv2.waitKey(0)
                cv2.imshow('rect',bb_img)
                print('(a)ccept , any other key to not accept')
                k=cv2.waitKey(0)
            lblname = row['description']+'_labels.txt'
            if manual_verification:
                if k == ord('a'):
                    with open(verified_objects_file,'a') as fp:
                        line = filename+'\t'+row['description']+'\t'+str(bb[0])+'\t'+str(bb[1])+'\t'+str(bb[2])+'\t'+str(bb[3])+'\n'
                        fp.write(line)
                        fp.close()
            else:
                with open(lblname,'a') as fp:
                    line = savename+'\t'+'1'+'\n'
                    fp.write(line)
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

def check_verified(verified_objects_file='verified_objects.txt',imagedir='/data/olympics/olympics'):
    with open(verified_objects_file,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            if line[0]=='#':  #first line describes fields
                continue
            filename,object_type,x,y,w,h=line.split()
            x=int(x)
            y=int(y)
            w=int(w)
            h=int(h)
            print('file {} obj {} x {} y {} w {} h {}'.format(filename,object_type,x,y,w,h))
            fullname = os.path.join(imagedir,filename)
            im = cv2.imread(fullname)
            if im is None:
                print('couldnt read '+filename)
                continue
#                cv2.imwrite(savename,bb_img)
            cv2.rectangle(im,(x,y),(x+w,y+h),color=[255,0,100],thickness=2)
            cv2.imshow('full',im)
            #cv2.waitKey(0)
            k=cv2.waitKey(0)
