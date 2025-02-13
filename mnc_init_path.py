import os.path
import sys

"""
Add lib paths and caffe path to system search path
"""


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

#cur_dir = "/root/MNC/"

# Add caffe python to PYTHONPATH
caffe_path = "/root/MNC/caffe-mnc/python"
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = "/root/MNC/lib"
add_path(lib_path)
