import cv2
import os
import pwd
import numpy as np
import logging

from trendi import Utils
#print os.environ["USER"]
#print os.getuid() # numeric uid
#print pwd.getpwuid(os.getuid())
#print os.environ["OLDPWD"]
if not 'REDIS_HOST' in os.environ:
    os.environ['REDIS_HOST'] = 'localhost'
    os.environ['REDIS_PORT'] = '6379'
if not 'MONGO_HOST' in os.environ:
    os.environ['MONGO_HOST'] = 'localhost'
    os.environ['MONGO_PORT'] = '27017'
from trendi import constants

import paperdoll_parse_enqueue

def show_pd_results(file_base_name):
    mask_file=file_base_name+'.bmp'
    jpg_file=file_base_name+'.jpg'
    pose_file=file_base_name+'.pose'

    print('reading '+str(jpg_file))
    img_arr = cv2.imread(jpg_file)
 #   paperdoll_parse_enqueue.colorbars()
    if img_arr is None:
        logging.debug('couldnt open file '+jpg_file)
        return
    print('reading '+str(mask_file))
    mask_arr = cv2.imread(mask_file)
    mask_arr = mask_arr[:,:,0] # all channels are identical
    print np.shape(mask_arr)
    if mask_arr is not None:
        pass
#        paperdoll_parse_enqueue.show_parse(img_array = mask_arr)
    else:
        print('couldnt get png at '+str(mask_file))
        return
    mask_arr = mask_arr-1
    uniques = np.unique(mask_arr)
    print('uniques;'+str(uniques))
    mmax = np.amax(mask_arr)
    mmin = np.amin(mask_arr)
    print('min {} max {} '.format(mmin,mmax))

    h = np.shape(img_arr)[0]
    w = np.shape(img_arr)[1]
    max_width = 500
    if w > max_width:
        img_arr = cv2.resize(img_arr,(h*max_width/w,max_width))
        mask_arr = cv2.resize(mask_arr,(h*max_width/w,max_width))

    maxVal = 56  # 57 categories in paperdoll
    scaled = np.multiply(mask_arr, int(255 / maxVal))
#    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_HSV)
 #   h,w,d = img_arr.shape
  #  print('h {0} w {1} d{2} '.format(h,w,d))
    mask_with_labels = add_unique_colorbars(constants.fashionista_categories,uniques,scaled)
    both = np.concatenate((img_arr,mask_with_labels), axis=1)
#    cv2.imshow("orig", img_arr)
 #   cv2.imshow("dest", colored)
    cv2.imshow("both", both)
#    colorbars()
    cv2.waitKey(0)

def add_unique_colorbars(labels,uniques,img_arr):
    maxval = 56
    bar_height = 12
    bar_width = 70
    text_width = 100
    font = cv2.FONT_HERSHEY_SIMPLEX
    i =0
    h = np.shape(img_arr)[0]
    w = np.shape(img_arr)[1]
    y_labelpos = h-20
    x_labelpos = 20   #bottom left corner
    for unique in uniques:
        img_arr[y_labelpos - (i+1)*bar_height:y_labelpos-i*bar_height,x_labelpos:x_labelpos+bar_width] = int(unique*255/maxval)
        cv2.putText(img_arr,labels[unique],(x_labelpos, max(5,y_labelpos -i*bar_height)), font, 0.5,128,1,8) #cv2.LINE_AA)
        i = i + 1
    colored = cv2.applyColorMap(img_arr, cv2.COLORMAP_HSV)
#    cv2.imshow('labels',colored)
 #   cv2.waitKey(0)
    return colored

def colorbars(labels):
    maxval = 56
    bar_height = 12
    bar_width = 70
    text_width = 100
    new_img = np.zeros([len(labels)*bar_height,bar_width+text_width],np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(0,len(labels)):
        new_img[i*bar_height:(i+1)*bar_height,0:bar_width] = int(i*255/maxval)
        cv2.putText(new_img,labels[i],(5+bar_width, bar_height+i*bar_height), font, 0.5,128,1,8) #cv2.LINE_AA)
  #      cv2.imwrite('testvarout.jpg',new_img)
    #print(new_img)
    colored = cv2.applyColorMap(new_img, cv2.COLORMAP_HSV)
    cv2.imshow('labels',colored)
    cv2.waitKey(0)

#    show_parse(img_array=new_img+1)

def save_clothing_parts(mask_file_or_arr,image_file_or_arr,savedir=None,visual_output=True):
    if isinstance(mask_file_or_arr,basestring):
        mask_arr = cv2.imread(mask_file_or_arr)
    elif isinstance(mask_file_or_arr,np.ndarray):
        mask_arr = mask_file_or_arr
    else:
        print('not clear what input is')
        return
    if isinstance(image_file_or_arr,basestring):
        img_arr = cv2.imread(image_file_or_arr)
        filename = image_file_or_arr.replace('.jpg','.part')
    elif isinstance(image_file_or_arr,np.ndarray):
        img_arr = image_file_or_arr
        filename = 'part'
    else:
        print('not clear what input is')
        return
    print('maskshape {} imshape {}'.format(mask_arr.shape,img_arr.shape))
    for i in np.unique(mask_arr):
        item_img = (mask_arr==i) * img_arr
        item_filename = os.path.join(savedir,os.path.basename(filename) + str(i)+'.jpg')
        if visual_output:
            cv2.imshow('part'+str(i),item_img)

        if savedir:
            print('saving '+item_filename)
            res=cv2.imwrite(item_filename,item_img)
            print('res '+str(res))

    cv2.waitKey(0)

if __name__ == '__main__':

    print os.getuid() # numeric uid
    print pwd.getpwuid(os.getuid())
    path = '/home/jr/tg/pd_output'
    path = '/media/jr/Transcend/my_stuff/tg/tg_ultimate_image_db/ours/pd_output_brain1'
    path = '/home/jr/tg/pd_output/'
    imgpath = '/data/jeremy/image_dbs/tg/pixlevel/pixlevel_fullsize_train'
    labelpath = '/data/jeremy/image_dbs/tg/pixlevel/pixlevel_fullsize_labels_fashionista_augmented_categories'
    #take the file 'base' i.e. without extension
    savedir = '/data/jeremy/image_dbs/tg/pixlevel/separated_parts'
#    imgfiles = [f.split('.')[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    labelfiles = [f for f in os.listdir(labelpath) if 'png' in f]
    imgfiles = [f for f in os.listdir(imgpath) if 'jpg' in f]

    print('nfiles:'+str(len(labelfiles)))
#    fashionista_ordered_categories = constants.fashionista_categories
        #in case it changes in future - as of 2/16 this list goes a little something like this:
        #fashionista_categories = ['null','tights','shorts','blazer','t-shirt','bag','shoes','coat','skirt','purse',
    # 'boots',  'blouse','jacket','bra','dress','pants','sweater','shirt','jeans','leggings','scarf','hat',
            #              'top','cardigan','accessories','vest','sunglasses','belt','socks','glasses','intimate',
              #            'stockings','necklace','cape','jumper','sweatshirt','suit','bracelet','heels','wedges','ring',
                #          'flats','tie','romper','sandals','earrings','gloves','sneakers','clogs','watch','pumps','wallet',
                  #        'bodysuit','loafers','hair','skin']
#    colorbars(fashionista_ordered_categories)
    Utils.ensure_dir(savedir)
    for file in imgfiles:
        full_img = os.path.join(imgpath,file)
        full_lbl = os.path.join(labelpath,file.replace('.jpg','.png'))
        if not os.path.isfile(full_img):
            print('img {} is not there '.format(full_img))
            continue
        if not os.path.isfile(full_lbl):
            print('img {} is not there '.format(full_lbl))
            continue
        save_clothing_parts(full_lbl,full_img,savedir=savedir)

  #      raw_input('enter')
#        show_pd_results(fullpath)
#        raw_input('enter for next')

