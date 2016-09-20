#!/usr/bin/python

# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Standard module
import os
import argparse
import time
import cv2
import numpy as np
# User-defined module
# import _init_paths
from . import mnc_init_path
import caffe
#from mnc_config import cfg
from mnc_config import cfg
from transform.bbox_transform import clip_boxes
from utils2.blob import prep_im_for_blob, im_list_to_blob
from transform.mask_transform import gpu_mask_voting
from utils2.vis_seg import _convert_pred_to_image, _get_voc_color_map
import matplotlib.pyplot as plt
import Image
import urllib
import time

#print('this has to be run from /root/MNC')

# VOC 20 classes
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

mnc_root = '/root/MNC'



def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    if url.count('jpg') > 1:
        return None
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if image.size == 0:
        return None
    new_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return new_image


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='MNC demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=mnc_root+'/models/VGG16/mnc_5stage/test.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=mnc_root+'/data/mnc_model/mnc_model.caffemodel.h5', type=str)

    args = parser.parse_args()
    return args


def prepare_mnc_args(im, net):
    # Prepare image data blob
    blobs = {'data': None}
    processed_ims = []
    im, im_scale_factors = \
        prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.TEST.SCALES[0], cfg.TRAIN.MAX_SIZE)
    processed_ims.append(im)
    blobs['data'] = im_list_to_blob(processed_ims)
    # Prepare image info blob
    im_scales = [np.array(im_scale_factors)]
    assert len(im_scales) == 1, 'Only single-image batch implemented'
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)
    # Reshape network inputs and do forward
    net.blobs['data'].reshape(*blobs['data'].shape)
    net.blobs['im_info'].reshape(*blobs['im_info'].shape)
    forward_kwargs = {
        'data': blobs['data'].astype(np.float32, copy=False),
        'im_info': blobs['im_info'].astype(np.float32, copy=False)
    }
    return forward_kwargs, im_scales


def im_detect(im, net):
    forward_kwargs, im_scales = prepare_mnc_args(im, net)
    blobs_out = net.forward(**forward_kwargs)
    # output we need to collect:
    # 1. output from phase1'
    rois_phase1 = net.blobs['rois'].data.copy()
    masks_phase1 = net.blobs['mask_proposal'].data[...]
    scores_phase1 = net.blobs['seg_cls_prob'].data[...]
    # 2. output from phase2
    rois_phase2 = net.blobs['rois_ext'].data[...]
    masks_phase2 = net.blobs['mask_proposal_ext'].data[...]
    scores_phase2 = net.blobs['seg_cls_prob_ext'].data[...]
    # Boxes are in resized space, we un-scale them back
    rois_phase1 = rois_phase1[:, 1:5] / im_scales[0]
    rois_phase2 = rois_phase2[:, 1:5] / im_scales[0]
    rois_phase1, _ = clip_boxes(rois_phase1, im.shape)
    rois_phase2, _ = clip_boxes(rois_phase2, im.shape)
    # concatenate two stages to get final network output
    masks = np.concatenate((masks_phase1, masks_phase2), axis=0)
    boxes = np.concatenate((rois_phase1, rois_phase2), axis=0)
    scores = np.concatenate((scores_phase1, scores_phase2), axis=0)
    return boxes, masks, scores


def get_vis_dict(result_box, result_mask, img_name, cls_names, vis_thresh=0.5):
    box_for_img = []
    mask_for_img = []
    cls_for_img = []
    for cls_ind, cls_name in enumerate(cls_names):
        det_for_img = result_box[cls_ind]
        seg_for_img = result_mask[cls_ind]
        keep_inds = np.where(det_for_img[:, -1] >= vis_thresh)[0]
        for keep in keep_inds:
            box_for_img.append(det_for_img[keep])
            mask_for_img.append(seg_for_img[keep][0])
            cls_for_img.append(cls_ind + 1)
    res_dict = {'image_name': img_name,
                'cls_name': cls_for_img,
                'boxes': box_for_img,
                'masks': mask_for_img}
    return res_dict

def mnc_pixlevel_detect(url_or_np_array):
    demo_dir = './'
    start = time.time()
    if isinstance(url_or_np_array,basestring):
        im = url_to_image(url_or_np_array)
        im_name = url_or_np_array.split('/')[-1]
    else:
        im = url_or_np_array
        im_name=int(time.time())+'.jpg'
    #resize to max dim of max_dim
    max_dim = 400
    h,w = im.shape[0:2]
    compress_factor = float(max(h,w))/max_dim
    new_h = int(float(h)/compress_factor)
    new_w = int(float(w)/compress_factor)
    im = cv2.resize(im,(new_w,new_h))
    actual_new_h,actual_new_w = im.shape[0:2]
    print('old w,h {}x{}, planned {}x{}, actual {}x{}'.format(w,h,new_w,new_h,actual_new_w,actual_new_h))

    gt_image = os.path.join(demo_dir, im_name)
    print gt_image
    cv2.imwrite(gt_image,im)

    boxes, masks, seg_scores = im_detect(im, net)
    end = time.time()
    print 'im_detect time %f' % (end-start)
    start = time.time()
    result_mask, result_box = gpu_mask_voting(masks, boxes, seg_scores, len(CLASSES) + 1,
                                               100, im.shape[1], im.shape[0])
    end = time.time()

    return result_mask,result_box,im,im_name

#load net
#args = parse_args()
test_prototxt = mnc_root+'/models/VGG16/mnc_5stage/test.prototxt'
test_model = mnc_root+'/data/mnc_model/mnc_model.caffemodel.h5'
print('ok computer 0')
caffe.set_mode_gpu()
print('ok computer 1')
caffe.set_device(0)
print('ok computer 2')
net = caffe.Net(test_prototxt, test_model, caffe.TEST)
print('ok computer 3')
