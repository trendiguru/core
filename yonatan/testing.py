#!/usr/bin/env python
import string
from Tkinter import Tk
from tkFileDialog import askopenfilename
import collections
import os

import cv2
import numpy as np

def get_image():
    Tk().withdraw()
    filename = askopenfilename()
    big_image = cv2.imread(filename)
    return big_image

base_image = get_image()
print base_image
#face_image = core/background_removal.find_face_cascade(base_image)
#print face_image