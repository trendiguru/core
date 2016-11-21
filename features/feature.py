import numpy as np
import caffe
import cv2
import skimage
from ..yonatan import yonatan_classifier
from .. import Utils


class Feature(object):

    def __init__(self, name, MODEL_FILE, PRETRAINED, gpu_device=None):

        self.name = name

        if gpu_device is None:
            caffe.set_device(int(gpu_device))
        caffe.set_mode_gpu()
        image_dims = [224, 224]
        mean, input_scale = np.array([104.0, 116.7, 122.7]), None
        channel_swap = [2, 1, 0]
        raw_scale = 255.0

        # Make classifier
        self.classifier = yonatan_classifier.Classifier(MODEL_FILE, PRETRAINED, image_dims=image_dims, mean=mean,
                                                   input_scale=input_scale, raw_scale=raw_scale,
                                                   channel_swap=channel_swap)

        print "Done initializing!"

    def distance(self, cats_num, v1, v2):

        categories_num = int(cats_num)
        if len(v1) != categories_num or len(v2) != categories_num:
            return None
        v1 = np.array(v1) if isinstance(v1, list) else v1
        v2 = np.array(v2) if isinstance(v2, list) else v2
        return np.linalg.norm(v1 - v2)

    def execute(self, image_or_url):

        print "{0} classification started!".format(self.name)
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
        predictions = self.classifier.predict(image_for_caffe)
        pred = predictions[0].tolist() if isinstance(predictions[0], np.ndarray) else predictions[0]
        return pred
