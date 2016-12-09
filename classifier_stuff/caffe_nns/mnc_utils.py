__author__ = 'jeremy'
import cPickle
import os
import numpy as np


#jeremy attempts to make .pkl files for use with MNC
#see https://github.com/daijifeng001/MNC/issues/5
#...yes, you need to implement MNC/pascal_voc_det.py
##in gt_overlap, it is 'overlap between classes'.#
##For the mask_max, this is because, I want to pass the mask before re-size to 21x21 to top layers. At this stage, mask size will be num_mask x 1 x mask_height x mask_width.#
##But different masks have different h/w, so I generate a blob according to the max value (num_mask x 1 x max(mask_height) x max(mask_width)) over each mask and pad with 0.


def checkout_pkl_file(thefile):
    if os.path.exists(thefile):
        with open(thefile, 'rb') as fid:
            db = cPickle.load(fid)
    for l in db:
        print l


def checkout_roi_pkl_file(thefile):
 #   {'boxes': array([[265, 144, 290, 213]], dtype=uint16), 'gt_overlaps': <1x21 sparse matrix of type '<type 'numpy.float32'>'
 #  with 1 stored elements in Compressed Sparse Row format>, 'gt_classes': array([3], dtype=int32), 'flipped': False}

    if os.path.exists(thefile):
        with open(thefile, 'rb') as fid:
            roi_db = cPickle.load(fid)
    count=0
    for l in roi_db:
        for k,v in l.iteritems():
            print k
            if not k in ['boxes','gt_overlaps','gt_classes','flipped']:
                print('got unexpected key :'+str(k))
        boxes=l['boxes']
        gt_overlaps=l['gt_overlaps']
        gt_classes=l['gt_classes']
        flipped=l['flipped']
        print('boxes:'+str(boxes))
        print('gt overlaps:'+str(gt_overlaps))
        print('gt_classes:'+str(gt_classes))
        print('flipped:'+str(flipped))
        count += 1
    print('count {}'.format(count))

def checkout_mask_pkl_file(thefile):
 #   {'boxes': array([[265, 144, 290, 213]], dtype=uint16), 'gt_overlaps': <1x21 sparse matrix of type '<type 'numpy.float32'>'
 #  with 1 stored elements in Compressed Sparse Row format>, 'gt_classes': array([3], dtype=int32), 'flipped': False}

    if os.path.exists(thefile):
        with open(thefile, 'rb') as fid:
            db = cPickle.load(fid)
    count=0
    for l in db:
        for k,v in l.iteritems():
            print k
            if not k in ['mask_max','gt_masks','flipped']:
                print('got unexpected key :'+str(k))

        flipped=l['flipped']
        mask_max=l['mask_max']
        gt_masks=l['gt_masks']
        print('mask_max:'+str(mask_max)+ ' flipped '+str(flipped))
        print('len gt masks:'+str(len(gt_masks))+' size0 '+str(gt_masks[0].shape))
        max0=0
        max1=0
        for m in gt_masks:
            gt0 = m.shape[0]
            max0=gt0 if gt0>max0 else max0
            gt1 = m.shape[1]
            max1=gt1 if gt1>max1 else max1
        count += 1
        print('calculated max {},{}'.format(max0,max1))
    print('count {}'.format(count))