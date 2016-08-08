__author__ = 'liorsabag'

"""
Usage: Usually, just call pd_py()
"""

import matlab
from matlab.engine import start_matlab
import sys
import os

default_root = os.getcwd()
pd_pipeline_file = 'data/paperdoll_pipeline.mat'
eng = None

def init_pd(root=default_root):
    global eng
    eng = start_matlab('-nodesktop -nojvm')
    eng.eval("load '{0}' 'config';".format(os.path.join(root, pd_pipeline_file)), nargout=0)
    eng.eval("addpath(genpath('{0}'))".format(root), nargout=0)
    eng.eval("config{1}.scale = 200; config{1}.model.thresh = -2;", nargout=0)


def pd_py(filename):
    if not eng:
        init_pd()
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
