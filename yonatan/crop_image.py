#!/usr/bin/env python

import numpy as np
import skimage.io
import os
import caffe
from .. import background_removal, Utils, constants
import cv2
import sys
import glob
import time
import skimage
import urllib
import pymongo
import argparse
import sys
import requests
import yonatan_constants


def crop(argv):

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="the argument should be a link to an image"
    )
    args = parser.parse_args()

    url_or_np_array = args.input_file

    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        # full_image = url_to_image(url_or_np_array)
        response = requests.get(url_or_np_array)  # download
        image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    else:
        return None

    # checks if the face coordinates are inside the image
    if image is None:
        print "not a good image"
        return None

    h, w = image.shape[:2]

    print "height: {0}, width: {1}".format(h, w)

    crop_img = image[0:h, w / 2:w]  # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

    cv2.imwrite("cropped_image.png", image)


if __name__ == '__main__':
    crop(sys.argv)
