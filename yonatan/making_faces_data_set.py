#!/usr/bin/env python

import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
import os
from PIL import Image

width = 115
height = 115

sets = {'train', 'test'}

for set in sets:
    if set == 'train':
        mypath_male = '/home/yonatan/train_set/male'
        mypath_female = '/home/yonatan/train_set/female'
    else:
        mypath_male = '/home/yonatan/test_set/male'
        mypath_female = '/home/yonatan/test_set/female'

    for root, dirs, files in os.walk(mypath_male):
        for file in files:
            if file.endswith(".jpg"):
                # Open the image file.
                img = Image.open(os.path.join(root, file))

                # Resize it.
                img = img.resize((width, height), Image.BILINEAR)

                # Save it back to disk.
                img.save(os.path.join(root, file + 'resized'))


    for root, dirs, files in os.walk(mypath_female):
        for file in files:
            if file.endswith(".jpg"):
                # Open the image file.
                img = Image.open(os.path.join(root, file))

                # Resize it.
                img = img.resize((width, height), Image.BILINEAR)

                # Save it back to disk.
                img.save(os.path.join(root, file + 'resized'))
