"""
put image sets into our db systematically keeping as much relevant data as poss.
e.g. for deep fashion, i will save data like this:
{ u'image_height': 4275,
 u'image_width': 2606,
 u'items': [{u'bb': [x, y, w, h], u'category': u'dress','attributes': [['pleated', 'fabric'], ['print', 'texture']]}]
 u'url': u'https://tg-training.storage.googleapis.com/deep_fashion/category_and_attribute_prediction/img/folder/6966.jpg',
}
where the categories are from the hydra categories (constants.hydra_cats)

"""

__author__ = 'jeremy'
import cv2
import numpy as np
import os
import json

from trendi.yonatan import yonatan_constants
from trendi.classifier_stuff.caffe_nns import create_nn_imagelsts
from trendi import Utils
from trendi import constants

def deepfashion_to_db(attribute_file='/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/list_attr_img.txt',
                        bbox_file='/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/list_bbox.txt',
                        bucket='https://tg-training.storage.googleapis.com/deep_fashion/category_and_attribute_prediction/',
 #                       bucket = 'gs://tg-training/deep_fashion/',
                        use_visual_output=True):
    '''
    takes deepfashion lists of bbs and attrs, and images in bucket.
    puts bb, attr, and link to file on bucket into db
    :param attribute_file:
    :param bbox_file:
    :param bucket:
    :param use_visual_output:
    :return:
    '''

    with open(attribute_file,'r') as fp:
        attrlines = fp.readlines()
        attrlines = attrlines[2:] #1st line is # of files, 2nd line describes fields
        fp.close()
    with open(bbox_file,'r') as fp2:
        bboxlines = fp2.readlines()
        bboxlines = bboxlines[2:]  #1st line is # of files, 2nd line describes fields
        fp2.close()

    bbox_files = [bboxline.split()[0] for bboxline in bboxlines ]

#    hydra_cats_for_deepfashion_folders = create_nn_imagelsts.deepfashion_to_tg_hydra(folderpath='/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/img')
    hydra_cats_for_deepfashion_folders=json.load('/data/jeremy/image_dbs/labels/deepfashion_to_hydra_map.txt')

    hydra_cats_dirsonly = [dummy[0] for dummy in hydra_cats_for_deepfashion_folders]

#    print(hydra_cats_for_deepfashion_folders[0])
#    print(hydra_cats_dirsonly[0])
#    print(attrlines[0])
#    print(bboxlines[0])
    db = constants.db
#    cursor = db.training_images.find()
    for line in attrlines:
        bbox = None
        hydra_cat = None
        info_dict = {}
        info_dict['items'] = []
        #raw_input('ret to cont')
        attribute_list = []
        #print line
        path = line.split()[0]
        vals = [int(i)+1 for i in line.split()[1:]]  #the vals are -1, 1 so add 1 to get 0, 2
        non_zero_idx = np.nonzero(vals)
        print('nonzero idx:'+str(non_zero_idx))
        for i in range(len(non_zero_idx[0])):
            #print yonatan_constants.attribute_type_dict[str(non_zero_idx[0][i])]
            attribute_list.append(yonatan_constants.attribute_type_dict[str(non_zero_idx[0][i])])
        print('attributes:'+str(attribute_list))
        url = bucket+path
        info_dict['items'].append({'attributes':attribute_list})
        print('url:'+str(url))
        try:
            bbox_index = bbox_files.index(path)  #there is prob a better way to search here than building another list
            bbox = [int(x) for x in bboxlines[bbox_index].split()[1:]]
            print('bbox '+str(bbox)+' line '+str(bboxlines[bbox_index]))
            #deepfashion bbox is x1 x2 y1 y2, convert to x y w h
            bbox_tg = [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
            info_dict['items'][0]['bb']=bbox_tg
        except ValueError:
            print(path+' is not in bboxfile list')
        try:
            folder_only = path.replace('img/','') #the paths from attr_file (and bbfile) have img/folder not just folder
            folder_only = os.path.dirname(folder_only)
            hydra_category_index = hydra_cats_dirsonly.index(folder_only)
            hydra_cat = hydra_cats_for_deepfashion_folders[hydra_category_index][1]
            print('hydracat '+str(hydra_cat)+' line '+str(hydra_cats_for_deepfashion_folders[hydra_category_index]))
            info_dict['items'][0]['category']=hydra_cat
        except ValueError:
            print(folder_only+' is not in hydracat list')
        img_arr = Utils.get_cv2_img_array(url)
        if img_arr is None:
            print('WARNING could not get '+url)
        else:
            h,w = img_arr.shape[0:2]
            if use_visual_output:
                if bbox is not None:
                    cv2.rectangle(img_arr,(bbox_tg[0],bbox_tg[1]),(bbox_tg[0]+bbox_tg[2],bbox_tg[1]+bbox_tg[3]),color=[255,0,0],thickness=5)
                cv2.imshow('deepfashion',img_arr)
                cv2.waitKey()
            info_dict['image_width'] = w
            info_dict['image_height'] = h

        info_dict['url'] = url

#        info_dict['items'] = items
        print('db entry:'+str(info_dict))
        ack = db.training_images_deepfashion.insert_one(info_dict)
        print('ack:'+str(ack.acknowledged))

def nextfun():
    pass

'''
for deep fashion, i will save data like this:
where the categories are from the hydra categories (constants.hydra_cats)

{ 'image_source':'deep_fashion',
 u'image_height': 4275,
 u'image_width': 2606,
 u'items': [{u'bb': [669, 726, 1453, 2295], u'category': u'dress'}],
 u'url': u'https://tg-training.storage.googleapis.com/deep_fashion/category_and_attribute_prediction/img/folder/6966.jpg',
}


fyi , filipino data is like this

 {u'a': 100,
 u'already_seen_image_level': 3,
 u'image_height': 4275,
 u'image_width': 2606,
 u'items': [{u'category': u'dress'},
  {u'category': u'eyewear'},
  {u'category': u'footwear'},
  {u'bb': [669, 726, 1453, 2295], u'category': u'dress'},
  {u'category': u'eyewear'},
  {u'category': u'footwear'}],
 u'mask_url': u'https://tg-training.storage.googleapis.com/tamara_berg_street2shop_dataset/masks/photo_16966.txt',
 u'selections': [u'true', u'false', u'false'],
 u'url': u'https://tg-training.storage.googleapis.com/tamara_berg_street2shop_dataset/images/photo_16966.jpg',
 u'user_name': [u'nobody',
  u'jeremy',
  u'nobody',
  u'jeremy',
  u'philipines4']}




'''
