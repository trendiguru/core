#!/usr/bin/env python

#mini - 0
#maxi = 1

from .. import constants
import pymongo
import os
import numpy as np
import matplotlib.pyplot as plt


db = constants.db

mini_dresses = db["mini"]
maxi_dresses = db["maxi"]

mini_text_file = open("mini_1900_dresses_with_faces.txt", "w")
maxi_text_file = open("maxi_1900_dresses_with_faces.txt", "w")

counter = 0

for doc in mini_dresses.find().limit(1900):
    mini_text_file.write(doc['image_url'] + ' 0' + '\n')
    counter += 1
    print counter


for doc in maxi_dresses.find().limit(1900):
    maxi_text_file.write(doc['image_url'] + ' 1' + '\n')
    counter += 1
    print counter

mini_text_file.flush()
maxi_text_file.flush()
