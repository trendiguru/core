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
from sklearn import mixture


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

def detect_frcnn(url_or_np_array,save_data=True,filename=None):
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
        print "not a good image"#
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
    return detections #

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

        inds = np.where(dets[:, -1] >= conf_thresh)[0]
        if len(inds) == 0:
            continue

        for i in inds:
            bbox = [int(a) for a in dets[i, :4]]
            score = dets[i, -1]
            print "class name: {0}, score: {1}".format(class_name, score)

#        """Draw detected bounding boxes."""
#            cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),3)
#            font = cv2.FONT_HERSHEY_SIMPLEX
 #           cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(int(bbox[0]), int(bbox[1] + 18)), font, 1,(0,255,0),2,cv2.LINE_AA)

            if class_name in ['person', 'bicycle',  'boat', 'bus', 'car',  'motorbike']:
                print('class {} bbox {} '.format(class_name,bbox))
                relevant_bboxes.append([class_name,bbox])
            margin_percent = 0.3  #remove this percent of orig. box size
            top_x,top_y,top_w,top_h = [bbox[0],bbox[1],bbox[2],int(bbox[3]/2)]

            extra_pixels_h = int(margin_percent*top_h/2)
            extra_pixels_w = int(margin_percent*top_w/2)
            top_bb_smallified = [top_x+extra_pixels_w,top_y+extra_pixels_h,int(top_w*(1-margin_percent)),int(top_w*(1-margin_percent))]
            print('topbb {} {} {} {} small {} percent {}'.format(top_x,top_y,top_w,top_h,top_bb_smallified,margin_percent))
            cropped_arr = img_arr[top_bb_smallified[1]:top_bb_smallified[1]+top_bb_smallified[3],
                          top_bb_smallified[0]:top_bb_smallified[0]+top_bb_smallified[2]]

            colors = dominant_colors(cropped_arr)
#            bottom_bb = [bbox[0],bbox[1]+bbox[3]/2,bbox[2],int(bbox[3]/2)]

            print colors
        # person_bbox = person_bbox.tolist()


    print("answer:".format(relevant_bboxes))
    return relevant_bboxes

def dominant_colors(img_arr,n_components=2):
    '''
    :param img_arr: this is a subimage (orig image cropped to a bb)
    :return:
    '''
    hsv = cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,0]
    hist = np.bincount(hue.ravel(),minlength=180) #hue goes to max 180
    print('hist:'+str(hist))
    gmix = mixture.GMM(n_components=n_components, covariance_type='full')
    gmix.fit(hist)
    print gmix
    print('covars:'+str(gmix.covars_))
    print('means:'+str(gmix.means_))

    avg_sat = np.mean(hsv[:,:,1])
    avg_val = np.mean(hsv[:,:,2])
    print('avg sat {} avg val {}'.format(avg_sat,avg_val))

    if avg_sat < 127 or avg_val < 127:
        return None
##	colors = ['r' if i==0 else 'g' for i in gmix.predict(samples)]
#	ax = plt.gca()
#	ax.scatter(samples[:,0], samples[:,1], c=colors, alpha=0.8)
#	plt.show()
    relevant_colors = []
    relevant_covars = []
    relevant_color_names = []
    color_bounds = [n*32+16 for n in range(8)]
    color_names = ['red','yellow','green','aqua','blue','purple']
    for c in range(n_components):
        mean =gmix.means_[c]
        covar =gmix.covars_[c]
        if mean < 0 or mean > 180 or covar > 180:
            continue
        color = [mean>color_bounds[i] and mean<color_bounds[i+1] for i in range(len(color_bounds))]
        color_ind = np.nonzero(color)[0]
        color_name = color_names[color_ind]
        relevant_colors.append[mean]
        relevant_covars.append[covar]
        relevant_color_names.append(color_name)
    return None



if __name__ == "__main__":
    url = 'http://media.gettyimages.com/videos/market-street-teeming-with-people-and-group-of-security-officers-in-video-id123273695?s=640x640'
    detect_frcnn(url)