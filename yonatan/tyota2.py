#!/usr/bin/env python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil
import yonatan_constants
from .. import background_removal, utils, constants
import sys

test_text_file = open("test_dir.txt", "w")

counter = 0


dir_path = '/home/yonatan/test_dir'

for root, dirs, files in os.walk(dir_path):
    for file in files:
        test_text_file.write(root + "/" + file + " 0\n")


        counter += 1
        print counter
