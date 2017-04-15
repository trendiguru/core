__author__ = 'jeremy'


import csv
import os
import cv2
import numpy as np
import copy

from trendi.defense import defense_client
from trendi import Utils
from trendi import constants

def read_csv(csvfile='/data/olympics/olympicsfull.csv',imagedir='/data/olympics/olympics',
             visual_output=False,confidence_threshold=0.9,manual_verification=True):
    ''''
    ok the bbx, bby , bbwidth, bbight are in % of image dims, and bbwidth/hight are not width/hight but
    rather x2,y2 of the bb
    '''
    #filename = "olympicsfull.csv"
    unique_descs=[]
    all_bbs=[]
    if manual_verification:  #write a description line in verified objects file
        verified_objects_file = 'verified_objects.txt'
        visual_output = True
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
            print('unique descriptions:'+str(unique_descs))
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

    print('unique descriptions:'+str(unique_descs))

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

def get_results_on_verified_objects(verified_objects_file='verified_objects.txt',in_docker=True,visual_output=False):
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
            bb_gt=[x,y,w,h]
            img = Utils.get_cv2_img_array("http://justvisual.cloudapp.net:8000/"+filename)
            imh,imw=img.shape[0:2]
            print('file {} obj {} x {} y {} w {} h {} imh {} imw {}'.format(filename,object_type,x,y,w,h,imh,imw))
            best_obj,iou = send_and_check(img,bb_gt,object_type,in_docker=in_docker,show_visual_output=visual_output)
            if best_obj:
                print('best fit {} iou {}'.format(best_obj,iou))
            else:
                print('no objects found')
        #split to 4 and check those, zoom in on them


def send_and_check(img,bb_gt,object_type,bb_to_analyze=None,show_visual_output=False,in_docker=True):
    if in_docker:
        defense_client.CLASSIFIER_ADDRESS = "http://hls_frcnn:8082/hls"
    else:
        defense_client.CLASSIFIER_ADDRESS = constants.FRCNN_CLASSIFIER_ADDRESS

    DEFENSE_CLASSES = ['bicycle', 'bus', 'car', 'motorbike', 'person','tvmonitor','train','bottle','chair']
    OLYMPICS_CLASSES = ['street_style-man', 'street_style-vehicle-private_car', 'street_style-vehicle-suv_jeep',
                        'street_style-vehicle-van', 'street_style-from_above-private_car', 'street_style-vehicle-pickup_truck',
                        'street_style-man-supermarket_cart', 'street_style-man-red_top', 'street_style-man-with_hat',
                        'street_style-man-blue_top', 'street_style-man-backpack', 'street_style-man-bag_in_hand']

    matches = {'street_style-man':['person'],
               'street_style-vehicle-private-car':['car'],
               'street_style-vehicle-suv_jeep':['bus','car'],
               'street_style-vehicle-van':['bus','car'],
               'street_style-from_above-private_car':['car'],
               'street_style-vehicle-pickup_truck':['bus','car'],
               'street_style-man-supermarket_cart':None,
               'street_style-man-red_top':['person'],
               'street_style-man-with_hat':['person'],
               'street_style-man-blue_top':['person'],
               'street_style-man-backpack':['person'],
               'street_style-man-bag_in_hand':['person']}

    if bb_to_analyze:
        retval = defense_client.detect(img,bb_to_analyze)
    else:
        retval = defense_client.detect(img)
    best_object = None
    print retval
    data = retval['data']
    if show_visual_output:
        cv2.rectangle(img,(bb_gt[0],bb_gt[1]),(bb_gt[0]+bb_gt[2],bb_gt[1]+bb_gt[3]),color=[100,255,100],thickness=2)
        cv2.putText(img,object_type,(bb_gt[0],max(0,bb_gt[1]-10)),cv2.FONT_HERSHEY_SIMPLEX,1,[100,255,100])
        cv2.imshow('img',img)
        cv2.waitKey(0)
    if retval == []:
        print('no objects detected')
    else:
        best_iou = 0
        for object in data:
            bb = object['bbox'] #we return x,y,w,h
            conf = object['confidence']
            iou = Utils.intersectionOverUnion(bb_gt,bb)
            found_object = object['object']
            if iou>best_iou:
                best_object = object
                best_iou = iou
            if show_visual_output:
                cv2.rectangle(img,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),color=[100,255,100],thickness=2)
                cv2.putText(img,found_object,cv2.FONT_HERSHEY_SIMPLEX,1,[100,255,100])
                cv2.imshow('img',img)
                cv2.waitKey(0)
#    raw_input('return to continue')
    print('best match found: {} {}'.format(best_iou,best_object))
    return(best_object,best_iou)

def get_zoomed_bb(img,bb_gt,percent_to_crop,show_visual_output=False):
    '''
    :param img:
    :param bb_gt:
    :param percent_to_crop: max crop (percent_to_crop=1) is just the bb, min crop (percent_to_crop=0) is no orig full image#
    :param show_visual_output:
    :return:
    '''
    image_h,image_w=img.shape[0:2]
    #max crops - top, left, bottom, right, maximum possible crop distance around bb
    max_crops = np.array([bb_gt[1],bb_gt[0],image_h-(bb_gt[1]+bb_gt[3]),image_w-(bb_gt[0]+bb_gt[2])])
    actual_crops = [int(max_crops[0]*percent_to_crop),
                    int(max_crops[1]*percent_to_crop),
                    int(max_crops[2]*(percent_to_crop)),
                    int(max_crops[3]*(percent_to_crop))]
    print('max crops '+str(max_crops)+'  top left  bottom right')
    print('actual crops '+str(actual_crops))
    #crop positions is y1 y2 x1 x2 (for use in crop operation that order makes sense)
    crop_positions=[actual_crops[0],image_h-actual_crops[2],actual_crops[1],image_w-actual_crops[3]]
    print('crop pos '+str(crop_positions)+' y1y2 x1x2')
    #new_bb is xywh
    new_bb = [crop_positions[2],crop_positions[0],crop_positions[3]-crop_positions[2],crop_positions[1]-crop_positions[0]]
    new_gt = [bb_gt[0]-new_bb[0],bb_gt[1]-new_bb[1],bb_gt[2],bb_gt[3]]
    print('new bb '+str(new_bb))
    if show_visual_output:
        cropped_img = img[crop_positions[0]:crop_positions[1],crop_positions[2]:crop_positions[3]]
        cv2.rectangle(img,(bb_gt[0],bb_gt[1]),(bb_gt[0]+bb_gt[2],bb_gt[1]+bb_gt[3]),color=[100,255,100],thickness=2)
        cv2.rectangle(img,(new_bb[0],new_bb[1]),(new_bb[0]+new_bb[2],new_bb[1]+new_bb[3]),color=[255,100,100],thickness=2)
        cv2.imshow('orig',img)
        cv2.rectangle(cropped_img,(new_gt[0],new_gt[1]),(new_gt[0]+new_gt[2],new_gt[1]+new_gt[3]),color=[100,255,100],thickness=2)
        cv2.imshow('cropped',cropped_img)
        cv2.waitKey(0)
    return new_bb

def tile_and_conquer(img,bb_gt,n,show_visual_output=False):
    '''
    divide into n subarrays , check each

    :param img:
    :param bb_gt:
    :return:
    '''
    image_width,image_height=img.shape[0:2]
    dx=image_height/n
    dy=image_width/n
    subarrays=[]
    orig_img=copy.copy(img)
    print('orig gt: '+str(bb_gt))
    if show_visual_output:
        cv2.rectangle(img,(bb_gt[0],bb_gt[1]),(bb_gt[0]+bb_gt[2],bb_gt[1]+bb_gt[3]),color=[255,0,100],thickness=2)
        cv2.imshow('origimage',img)
    for i in range(n):
        for j in range(n):
            top=dy*j
            left=dx*i
            subimage = orig_img[top:top+dy,left:left+dx] #use orig image to avoid bb that may get drawn
            subimage_w,subimage_h = subimage.shape[0:2]
            #the ground truth has now shifted
            new_gt = [bb_gt[0]-top,bb_gt[1]-left,bb_gt[2],bb_gt[3]]
            #check if gt is in the current subimage, only check image if it is
            #this will miss all the false pos but rght now lets just conc. on false neg
            if ((new_gt[0]<0 and new_gt[0]+new_gt[0]+new_gt[2]<0) or
                (new_gt[0]>0 and new_gt[0]+new_gt[0]+new_gt[3]<0)):
                pass
            print('new gt:{} subimage dims {} '.format(new_gt,subimage.shape))
            orig_subimage = copy.copy(subimage)
            if show_visual_output:
                cv2.rectangle(img,(bb_gt[0],bb_gt[1]),(bb_gt[0]+bb_gt[2],bb_gt[1]+bb_gt[3]),color=[255,0,100],thickness=2)
                cv2.imshow('subimage',subimage)
                cv2.waitKey(0)
            send_and_check(orig_subimage,new_gt)