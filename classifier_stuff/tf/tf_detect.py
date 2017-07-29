__author__ = 'jeremy'

# coding: utf-8

import os
import sys
import time
tensordir = '/data/jeremy/tensorflow/models/object_detection'
pardir = '/data/jeremy/tensorflow/models/'
os.chdir(tensordir)
print('cwd '+str(os.getcwd()))
sys.path.append(pardir)
#sys.path.append(".")
#sys.path.append(str(os.getcwd()))
#sys.path.append("..")
#print sys.path

import matplotlib
import sys
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
import cv2
import copy

from trendi.utils import imutils
from trendi import constants


MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
#MODEL_NAME = 'rfcn_resnet101_coco_11_06_2017'
#MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

get_model=False
if get_model:
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
        print('got model '+file_name)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


sess = None

def do_detect():
      global sess
    #use the ConfigProto to determine if GPU is running
#      with tf.Session(graph=detection_graph,config=tf.ConfigProto(log_device_placement=True)) as sess:
      with tf.Session(graph=detection_graph) as sess:
            for image_path in TEST_IMAGE_PATHS:
              start_time = time.time()
              image = Image.open(image_path)
              # the array based representation of the image will be used later in order to prepare the
              # result image with boxes and labels on it.
              image_np = load_image_into_numpy_array(image)

              # new_x=int(image_np.shape[1]/factor)
              # new_y=int(image_np.shape[0]/factor)
              # image_np = cv2.resize(image_np,(new_x,newy))
              # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
              image_np_expanded = np.expand_dims(image_np, axis=0)
              image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
              boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
              scores = detection_graph.get_tensor_by_name('detection_scores:0')
              classes = detection_graph.get_tensor_by_name('detection_classes:0')
              num_detections = detection_graph.get_tensor_by_name('num_detections:0')
              # Actual detection.
              (boxes, scores, classes, num_detections) = sess.run(
                  [boxes, scores, classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
              # Visualization of the results of a detection.
              vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=8)
              print('im shape '+str(image_np.shape))
              print('elapsed time '+str(time.time()-start_time))
              savename =  os.path.basename(image_path).strip('.jpg')+MODEL_NAME+'out.jpg'
              print('saving '+savename)
              cv2.imwrite(savename,image_np)
              visual_output=False
              if visual_output:
                  plt.figure(figsize=IMAGE_SIZE)
                  plt.imshow(image_np)
                  plt.savefig(savename)


def analyze_image(image_path,label_conversion=constants.tfcc2tg_map,thresh = 0.5):
  with tf.Session(graph=detection_graph) as sess:
      print('starting image analyse')
      start_time = time.time()
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      bgr_img = copy.deepcopy(image_np)
      print('origshape {}'.format(bgr_img.shape))
#      bgr_img  = bgr_img[...,::-1]
#      cv2.imshow('orig',orig)
#      cv2.waitKey(0)
   #   orig  = orig[...,[2,1,0]])
   #   orig  = np.array(orig[...,[2,1,0]]) #or [...,::-1]
      bgr_img = cv2.cvtColor(bgr_img,cv2.COLOR_RGB2BGR)
      print('origshape {}'.format(bgr_img.shape))
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      print('elapsed time '+str(time.time()-start_time))
      save_output = True
      if save_output:
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=1)
      print('im shape '+str(image_np.shape))
      savename =  os.path.basename(image_path).strip('.jpg')+MODEL_NAME+'out.jpg'
      print('saving '+savename)
      cv2.imwrite(savename,image_np)
      visual_output=False
      if visual_output:
          plt.figure(figsize=IMAGE_SIZE)
          plt.imshow(image_np)
          plt.savefig(savename)
      print('box {} score {} class {}'.format(boxes.shape,scores.shape,classes.shape))
      boxes_thresholded=[]
      scores_thresholded=[]
      class_names_thresholded=[]
      relevant_boxes = []
      for i in range(len(boxes[0])):
          if scores[0][i]<thresh:
              continue
          else:
              score = scores[0][i]
              bbox_tf = boxes[0][i][:]
              bb_x1y1x2y2 = imutils.tf_to_x1y1x2y2(bbox_tf,image_np.shape[0:2])
              bbox_xywh = imutils.x1y1x2y2_to_xywh(bb_x1y1x2y2)
              classno=int(classes[0][i])
              classname = category_index[classno]['name']
              if classname in label_conversion:
                  bgr_img = imutils.bb_with_text(bgr_img,bbox_xywh,classname+str(score))
                  scores_thresholded.append(score)
                  boxes_thresholded.append(bb_x1y1x2y2)
                  print('classno '+str(classname)+' convert to '+str(label_conversion[classname]))
                  print('bbtf {} x1y1x2y2 {} xywh {} size {}'.format(bbox_tf,bb_x1y1x2y2,bbox_xywh,image_np.shape[0:2]))
                  class_names_thresholded.append(label_conversion[classname])
                  item = {'object':classname,'bbox':bb_x1y1x2y2,
                          'confidence':round(float(score),3)}
                  relevant_boxes.append(item)


              else:
                  #or throw out ....
                pass
#                class_names_thresholded.append(classname)
      visual_output=True
      if visual_output:
          cv2.imshow('ours',bgr_img)
          cv2.waitKey(0)
      print('boxes '+str(boxes_thresholded))
      print('scores '+str(scores_thresholded))
      print('classes '+str(class_names_thresholded))
      print('numdet '+str(num_detections))
      #https://www.tensorflow.org/versions/r0.12/api_docs/python/image/working_with_bounding_boxes
 #The coordinates of the each bounding box in boxes are encoded as [y_min, x_min, y_max, x_max]. The bounding box coordinates are floats in [0.0, 1.0] relative to the width and height of the underlying image.3
      return(relevant_boxes)

with detection_graph.as_default():
  gpu = False
  gpu_n = 1
  if gpu:
    with tf.device('/gpu:'+str(gpu_n)):
#    with tf.device('/gpu:0'):
      print('using gpu'+str(gpu_n))
      do_detect()
  else:
    do_detect()
