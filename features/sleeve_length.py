
import numpy as np
import caffe
import cv2
import skimage
from ..yonatan import yonatan_classifier
from .. import Utils

MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_50_dress_sleeve/ResNet-50-deploy.prototxt"
PRETRAINED = "/home/yonatan/resnet50_caffemodels/caffe_resnet50_snapshot_50_sgd_iter_10000.caffemodel"
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
    if len(v1) != 8 or len(v2) != 8:
        return None
    return np.linalg.norm(v1 - v2)


def execute(image_or_url):

    print "Sleeve classification started!"
    print "image_or_url is {0}".format(image_or_url)
    if isinstance(image_or_url, str):
        image = Utils.get_cv2_img_array(image_or_url)
        if image is None:
            return None
    else:
        image = image_or_url

    image_for_caffe = [skimage.img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)]

    if image_for_caffe is None:
        return None

    # Classify
    predictions = classifier.predict(image_for_caffe)

    return predictions[0]
