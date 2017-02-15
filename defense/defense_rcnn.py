__author__ = 'whoever made rcnn and yonatan and now jeremy'
#!/usr/bin/env python

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import matplotlib as mpl
mpl.use('Agg')


#where does system think this link is - at link location or where link points to
import os
base_dir = os.path.dirname(os.path.realpath(__file__))
print('current_dir is '+str(base_dir))

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import string
import json
import numpy as np
import caffe, os, sys, cv2
import argparse
import requests
import random

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

DEFENSE_CLASSES = ('__background__', 'bicycle', 'bus', 'car', 'motorbike', 'person')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

gpu_id = 1
cpu_mode = False
demo_net = 'vgg16'

cfg.TEST.HAS_RPN = True  # Use RPN for proposals

prototxt = os.path.join(cfg.MODELS_DIR, NETS[demo_net][0],
                        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                          NETS[demo_net][1])

if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\nDid you run ./data/script/'
                   'fetch_faster_rcnn_models.sh?').format(caffemodel))

if cpu_mode:
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    cfg.GPU_ID = gpu_id
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

print '\n\nLoaded network {:s}'.format(caffemodel)

def detect_frcnn(url_or_np_array,save_data=1,filename=None):
    print "detect_frcnn started"
    # check if i get a url (= string) or np.ndarray
    if filename:
        full_image = cv2.imread(filename)
        url = 'from_file'
    elif isinstance(url_or_np_array, basestring):
        response = requests.get(url_or_np_array)  # download
        full_image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
        filename=url_or_np_array.replace('https://','').replace('http://','').replace('/','_')
        url = url_or_np_array
    elif type(url_or_np_array) == np.ndarray:
        full_image = url_or_np_array
        n_chars=6
        filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n_chars))+'.jpg'
        url = 'from_img_arr'
    else:
        return None

    if full_image is None:
        print "not a good image"
        return None

    #demo(full_image)
    detections,img_arr = do_detect_frcnn(full_image)
    if save_data:
        save_path='./'
        imgname=os.path.join(save_path,filename)
        if imgname[:-4] != '.jpg':
            imgname = imgname + '.jpg'
        cv2.imwrite(imgname,full_image)
        detections.append(filename)
        detections.append(url)
        textfile = os.path.join(save_path,'output.txt')
        with open(textfile,'a') as fp:
            json.dump(detections,fp,indent=4)
            fp.close()
        print('wrote image to {} and output text to {}'.format(imgname,textfile))
    return detections

def do_detect_frcnn(img_arr,conf_thresh=0.8,NMS_THRESH=0.3):
    """Detect object classes in an image using pre-computed object proposals."""

    person_bbox = []
    relevant_bboxes = []

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, img_arr)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
#        vis_detections(im, cls, dets, thresh=CONF_THRESH)

        class_name = cls

        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= conf_thresh)[0]
        if len(inds) == 0:
            continue

        for i in inds:
            bbox = [int(a) for a in dets[i, :4]]
            score = dets[i, -1]

            print "class name: {0}, score: {1}".format(class_name, score)

            cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(int(bbox[0]), int(bbox[1] + 18)), font, 1,(0,255,0),2,cv2.LINE_AA)

            if class_name in ['person', 'bicycle',  'boat', 'bus', 'car',  'motorbike']:
                print('class {} bbox {} '.format(class_name,bbox))
                relevant_bboxes.append([class_name,bbox])
        # person_bbox = person_bbox.tolist()

    print("relevant boxes: {}".format(relevant_bboxes))
    print cv2.imwrite("/data/yonatan/linked_to_web/testing_2.jpg", im)
    return relevant_bboxes