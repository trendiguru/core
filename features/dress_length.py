
import numpy as np
import caffe
import cv2
import skimage
from ..yonatan import yonatan_classifier
from .. import Utils

MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_50_dress_length/ResNet-50-deploy.prototxt"
PRETRAINED = "/home/yonatan/dressLength_caffemodels/caffe_resnet50_snapshot_dress_length_3_categories_iter_10000.caffemodel"

classifier = None


def load(gpu_device=None):
    if gpu_device is None:
        caffe.set_device(int(gpu_device))
    caffe.set_mode_gpu()
    image_dims = [224, 224]
    mean, input_scale = np.array([120, 120, 120]), None
    channel_swap = [2, 1, 0]
    raw_scale = 255.0

    # Make classifier
    classifier = yonatan_classifier.Classifier(MODLE_FILE, PRETRAINED, image_dims=image_dims, mean=mean,
                                               input_scale=input_scale, raw_scale=raw_scale, channel_swap=channel_swap)

    print "Done initializing!"


def distance(v1, v2):
    if len(v1) != 3 or len(v2) != 3:
        return None
    v1 = np.array(v1) if isinstance(v1, list) else v1
    v2 = np.array(v2) if isinstance(v2, list) else v2
    return np.linalg.norm(v1 - v2)


def execute(image_or_url):

    print "Length classification started!"
    if isinstance(image_or_url, basestring):
        image = Utils.get_cv2_img_array(image_or_url)
    elif type(image_or_url) == np.ndarray:
        image = image_or_url
    else:
        return None

    image_for_caffe = [skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)]

    if image_for_caffe is None:
        return None

    # Classify
    predictions = classifier.predict(image_for_caffe)
    pred = predictions[0].tolist() if isinstance(predictions[0], np.ndarray) else predictions[0]
    return pred
