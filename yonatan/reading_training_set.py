#!/usr/bin/env python

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

width = 100
height = 100

mypath = '../../male'
f = []
breaker = False
text_file = open("train.txt", "w")
for root, dirs, files in os.walk(mypath):
    for file in files:
        if file.endswith(".jpg"):
            text_file.write(root + "/" + file + " 0\n")
            img = Image.open(root + "/" + file)
            iar = np.asarray(img)
            f.append(iar)
text_file.flush()
#text_file.close()

mypath = '../../female'
f = []
breaker = False
text_file = open("train.txt", "a")
for root, dirs, files in os.walk(mypath):
    for file in files:
        if file.endswith(".jpg"):
            text_file.write(root + "/" + file + " 1\n")
            img = Image.open(root + "/" + file)
            iar = np.asarray(img)
            f.append(iar)
text_file.flush()
#text_file.close()



'''
             #breaker  = True
             #break
    if breaker:
        break

        if file.endswith("Aaron_Eckhart_0001.jpg"):
             # Open the image file.
         img = Image.open(os.path.join(root, file))
 
             # Resize it.
             img = img.resize((width, height), Image.BILINEAR)
 
             # Save it back to disk.
             img.save(os.path.join(root, 'resized-' + file))
'''