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
maxi_dresses = db.maxi

text_file = open("mini_maxi_1800_dresses.txt", "w")

counter = 0

for doc in mini_dresses.find().limit(900):
    text_file.write(doc['image_url'] + ' 0' + '\n')
    print doc['image_url']
    counter += 1
    if counter % 100 == 0:
        print "after {0} docs".format(counter)

text_file.flush()
