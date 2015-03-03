__author__ = 'liorsabag'
import csv
import gzip
import json
import numpy
import requests
from cv2 import imread, imdecode
import logging


def get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array):
    # first check if we have a numpy array
    if isinstance(url_or_path_to_image_file_or_cv2_image_array, numpy.ndarray):
        img_array = url_or_path_to_image_file_or_cv2_image_array
    # otherwise it's probably a string, check what kind
    elif isinstance(url_or_path_to_image_file_or_cv2_image_array, basestring):
        if "://" in url_or_path_to_image_file_or_cv2_image_array:
            img_url = url_or_path_to_image_file_or_cv2_image_array
            img_array = imdecode(numpy.asarray(bytearray(requests.get(img_url).content)), 1)
        else:
            img_path = url_or_path_to_image_file_or_cv2_image_array
            img_array = imread(img_path)
    # After we're done with all the above, this should be true - final check that we're outputting a good array
    if not (isinstance(img_array, numpy.ndarray) and isinstance(img_array[0][0], numpy.ndarray)):
        logging.warning("Bad image - check url/path/array")
    return img_array


class GZipCSVReader:
    def __init__(self, filename):
        self.gzfile = gzip.open(filename)
        self.reader = csv.DictReader(self.gzfile)

    def next(self):
        return self.reader.next()

    def close(self):
        self.gzfile.close()

    def __iter__(self):
        return self.reader.__iter__()


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)