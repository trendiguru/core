__author__ = 'liorsabag'

"""
Usage: Usually, just call pd_py()
"""

import matlab
from matlab.engine import start_matlab
import sys
import os
import logging
import random
import string
import numpy as np
import cv2

from .. import Utils
from .. import constants

default_root = os.getcwd()
pd_pipeline_file = 'data/paperdoll_pipeline.mat'
ENG = None


def init_pd_eng(root=default_root):
    global ENG
    if not ENG:
        eng = start_matlab('-nodesktop -nojvm')
        eng.eval("load '{0}' 'config';".format(os.path.join(root, pd_pipeline_file)), nargout=0)
        eng.eval("addpath(genpath('{0}'))".format(root), nargout=0)
        eng.eval("config{1}.scale = 200; config{1}.model.thresh = -2;", nargout=0)
        ENG = eng
    return eng


def raw_parse(filename, _eng=None):
    eng = _eng or init_pd_eng()
    eng.eval("mask = []; label_names = []; pose = [];", nargout=0)
    try:
        eng.eval("image_array = imread('{0}');".format(filename), nargout=0)
    except:
        print sys.exc_info()
        return
    eng.eval("input_sample = struct('image', image_array);", nargout=0)
    eng.eval("result = feature_calculator.apply(config, input_sample);", nargout=0)
    try:
        eng.eval("mask = imdecode(result.final_labeling, 'png'); mask = mask-1;", nargout=0)
        mask = eng.eval("mask;", nargout=1)
        label_names = eng.eval("result.refined_labels;", nargout=1)
        pose = eng.eval("result.pose;", nargout=1)
    except:
        print sys.exc_info()
        return
    return mask, label_names, pose


def parse(img_url_or_cv2_array, _eng=None, filename=None):
    img = Utils.get_cv2_img_array(img_url_or_cv2_array)
    filename = filename or rand_string()
    img_ok = image_big_enough(img)
    if img_ok and cv2.imwrite(filename + '.jpg', img):
        mask, label_dict, pose = raw_parse(filename + '.jpg', _eng=_eng)
        mask_np = np.array(mask, dtype=np.uint8)
        pose_np = np.array(pose, dtype=np.uint8)

        return mask_np, label_dict, pose_np, filename
    else:
        if img_ok:
            logging.debug("problem writing " + str(filename) + " parse()")
        return


def image_big_enough(img_array):
    if img_array is None:
        logging.debug("input image is empty")
        return False
    width, height = img_array.shape[0:2]
    if width < constants.minimum_im_width or height < constants.minimum_im_height:
        logging.debug('image dimensions too small')
        return False
    else:
        return True


def rand_string():
    return ''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(32)])