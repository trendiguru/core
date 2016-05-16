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

mini_text_file = open("mini_900_dresses.txt", "w")
maxi_text_file = open("maxi_900_dresses.txt", "w")

counter = 0

for doc in mini_dresses.find().limit(900):
    mini_text_file.write(doc['image_url'] + ' 0' + '\n')
    print counter
    counter += 1

for doc in maxi_dresses.find().limit(900):
    maxi_text_file.write(doc['image_url'] + ' 1' + '\n')
    print counter
    counter += 1

mini_text_file.flush()
maxi_text_file.flush()
