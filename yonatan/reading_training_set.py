#!/usr/bin/env python

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sets = {'train', 'test'}
total_gender_pics = 0
total_gender_train_pics = 1500
total_gender_test_pics = 250

for set in sets:
    if set == 'train':
        total_gender_pics = total_gender_train_pics
        mypath_male = '/home/yonatan/train_set/male'
        mypath_female = '/home/yonatan/train_set/female'
    else:
        total_gender_pics = total_gender_test_pics
        mypath_male = '/home/yonatan/test_set/male'
        mypath_female = '/home/yonatan/test_set/female'

    breaker = False
    male_count = 0
    text_file = open(set + ".txt", "w")
    for root, dirs, files in os.walk(mypath_male):
        if breaker:
            break
        for file in files:
            if male_count >= total_gender_pics:
                breaker = True
                break
            if file.endswith(".jpg"):
                text_file.write(root + "/" + file + " 0\n")
                male_count += 1
    text_file.flush()
    #text_file.close()

    breaker = False
    female_count = 0
    text_file = open(set + ".txt", "a")
    for root, dirs, files in os.walk(mypath_female):
        if breaker:
            break
        for file in files:
            if female_count >= total_gender_pics:
                breaker = True
                break
            if file.endswith(".jpg"):
                text_file.write(root + "/" + file + " 1\n")
                female_count += 1
    text_file.flush()
    #text_file.close()




'''
width = 100
height = 100
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
'''
    mypath = '/home/yonatan/test_set/female'
    f = []
    breaker = False
    text_file = open("test.txt", "a")
    for root, dirs, files in os.walk(mypath):
        for file in files:
            if file.endswith(".jpg"):
                text_file.write(root + "/" + file + " 1\n")
                #img = Image.open(root + "/" + file)
                #iar = np.asarray(img)
                #f.append(iar)
    text_file.flush()
    #text_file.close()
'''