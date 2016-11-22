# Necessary to load and execute net
import traceback
try:
    import caffe
    import cv2
    import skimage
    from ..yonatan import yonatan_classifier
    from .. import Utils
    can_load = True
except:
    import_error = traceback.format_exc()
    can_load = False
# Necessary for both load and distance
import numpy as np
from .config import FEATURES

"""
This class can be used both to load a feature classifier net and to get distances between two results of such calssification.
These use cases happen in different environments, therefore import logic above allows not all packages to be installed.
"""
class Feature(object):
    def __init__(self, name, model_file=None, pretrained=None, gpu_device=None, features_config=FEATURES):
        self.name = name
        self.model_file = model_file or features_config[name]["MODEL_FILE"]
        self.pretrained = pretrained or features_config[name]["PRETRAINED"]
        self.labels = features_config[name]["labels"]
    
    
    def load(self)
        if not can_load:
            raise ImportError(import_error)
        if gpu_device is not None:
            caffe.set_device(int(gpu_device))
        caffe.set_mode_gpu()
        image_dims = [224, 224]
        mean, input_scale = np.array([104.0, 116.7, 122.7]), None
        channel_swap = [2, 1, 0]
        raw_scale = 255.0

        # Make classifier
        self.classifier = yonatan_classifier.Classifier(self.model_file, 
                                                        self.pretrained, 
                                                        image_dims=image_dims, 
                                                        mean=mean,
                                                        input_scale=input_scale, 
                                                        raw_scale=raw_scale,
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
