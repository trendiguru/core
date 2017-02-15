#!/usr/bin/env python

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import matplotlib as mpl
mpl.use('Agg')
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import caffe, os, sys, cv2
import argparse
import requests

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


def theDetector(url_or_np_array):

    print "Starting the Demo!"
    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        response = requests.get(url_or_np_array)  # download
        full_image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    elif type(url_or_np_array) == np.ndarray:
        full_image = url_or_np_array
    else:
        return None

    if full_image is None:
        print "not a good image"
        return None

    #demo(full_image)
    demo("/data/yonatan/linked_to_web/testing_2.jpg", full_image, 1)


def demo(image_name, image_data=0, link=0):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im = cv2.imread(im_file)

    if link:
        im = image_data
    else:
        #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
        im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
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
        thresh = 0.6

        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            if class_name == 'person':
                # print int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                # print bbox
                bagpack, handbag, hat = dummy_function(im, bbox)

                shirt_color = find_if_shirt_blue_or_red(im, bbox)

            if bagpack:
                class_name += "_with_bagpack"

            if handbag:
                class_name += "_with_handbag"

            if hat:
                class_name += "_with_hat"

            if shirt_color == "red":
                class_name += "_with_red_top"
            elif shirt_color == "blue":
                class_name += "_with_blue_top"
            else:
                class_name += "_without_red_or_blue_top"

            print "class name: {0}, score: {1}".format(class_name, score)

            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1] + 18)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    print cv2.imwrite("/data/yonatan/linked_to_web/testing_2.jpg", im)


def find_if_shirt_blue_or_red(image, bbox):

    threshold = 0.33

    # define BGR boundaries
    lower_red = np.array([0, 0, 112], dtype="uint8")
    upper_red = np.array([97, 105, 255], dtype="uint8")

    lower_blue = np.array([102, 0, 0], dtype="uint8")
    upper_blue = np.array([255, 104, 123], dtype="uint8")

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    person = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    shirt_bbox = person[w/3:2*w/3, 2*h/9:4*h/9]

    mask_red = cv2.inRange(shirt_bbox, lower_red, upper_red)

    if cv2.countNonZero(mask_red) / float(h * w) > threshold:
        return "red"

    mask_blue = cv2.inRange(shirt_bbox, lower_blue, upper_blue)

    if cv2.countNonZero(mask_blue) / float(h * w) > threshold:
        return "blue"

    return None


def dummy_function(image, bbox):


    # bagpack, handbag, hat are binaries
    return bagpack, handbag, hat
