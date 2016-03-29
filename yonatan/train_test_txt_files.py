#!/usr/bin/env python

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

width = 100
height = 100

mypath = '/home/yonatan/Documents/Trendi/male-female/train/male'
f = []
breaker = False
text_file = open("train.txt", "w")
for root, dirs, files in os.walk(mypath):
    for file in files:
        if file.endswith(".jpg"):
	    text_file.write(file + " 0\n")

