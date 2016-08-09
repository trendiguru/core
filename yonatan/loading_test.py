#!/usr/bin/env python

import caffe
import numpy as np
from .. import background_removal, Utils, constants
import cv2
import os
import sys
import argparse
import glob
import time
import skimage
from PIL import Image
from . import gender_detector
import random
import matplotlib.pyplot as plt
import yonatan_classifier
from itertools import combinations, product


# to load the saved arrays: #
array_strapless = np.load('array_strapless.npy')
array_spaghetti_straps = np.load('array_spaghetti_straps.npy')
array_regular_straps = np.load('array_regular_straps.npy')
array_sleeveless = np.load('array_sleeveless.npy')
array_cap_sleeve = np.load('array_cap_sleeve.npy')
array_short_sleeve = np.load('array_short_sleeve.npy')
array_midi_sleeve = np.load('array_midi_sleeve.npy')
array_long_sleeve = np.load('array_long_sleeve.npy')


# the order of the coralations being printed is:
# 0 and 1, 0 and 2, ...
# 1 and 2, 1 and 3, ...
# 2 and 3, 2 and 4, ...
# ...
print "\ncorrelation between:\n0 and 1, 0 and 2, ...\n1 and 2, 1 and 3, ...\n2 and 3, 2 and 4, ...\n"
for comb in combinations([array_strapless, array_spaghetti_straps,array_regular_straps,
                          array_sleeveless, array_cap_sleeve, array_short_sleeve,
                          array_midi_sleeve, array_long_sleeve], 2):
    print np.min([np.linalg.norm(a-b) for a,b in product(*comb)])

print "mean array_strapless: {0}\nmean array_spaghetti: {1}\n" \
      "mean array_regular: {2}\nmean array_sleeveless: {3}\n" \
      "mean array_cap_sleeve: {4}\nmean array_short_sleeve: {5}\n" \
      "mean array_midi_sleeve: {6}\nmean array_long_sleeve: {7}\n".format(
      np.mean(array_strapless, 0), np.mean(array_spaghetti_straps, 0), np.mean(array_regular_straps, 0), np.mean(array_sleeveless, 0),
      np.mean(array_cap_sleeve, 0), np.mean(array_short_sleeve, 0), np.mean(array_midi_sleeve, 0), np.mean(array_long_sleeve, 0))

print "variance array_strapless: {0}\nvariance array_spaghetti: {1}\n" \
      "variance array_regular: {2}\nvariance array_sleeveless: {3}\n" \
      "variance array_cap_sleeve: {4}\nvariance array_short_sleeve: {5}\n" \
      "variance array_midi_sleeve: {6}\nvariance array_long_sleeve: {7}\n".format(
      np.var(array_strapless, 0), np.var(array_spaghetti_straps, 0), np.var(array_regular_straps, 0), np.var(array_sleeveless, 0),
      np.var(array_cap_sleeve, 0), np.var(array_short_sleeve, 0), np.var(array_midi_sleeve, 0), np.var(array_long_sleeve, 0))
