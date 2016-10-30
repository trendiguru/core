#!/usr/bin/env python

import pymongo
import re
import cv2
import requests
import numpy as np
import os
import random
from ..utils import imutils


db = pymongo.MongoClient().mydb

## regex:
# startswith - ^
# is there - .

limit = 100


##---- Casual: 0 ----##
regx_casual = re.compile("/*Casual", re.IGNORECASE)
casual_male_num = db.amazon_US_Male.count({'tree': regx_casual})  # = 117405
casual_female_num = db.amazon_US_Female.count({'tree': regx_casual})  # = 205237

casual_male = db.amazon_US_Male.find({'tree': regx_casual})
casual_female = db.amazon_US_Female.find({'tree': regx_casual})

casual_txt_file = open("/home/yonatan/style_classifier/casual_txt_file.txt", "w")

list_to_iter = range(1, casual_male_num)
random.shuffle(list_to_iter)
counter = 1

for i in list_to_iter:

    if counter > limit:
        break

    link_to_image = casual_male[i]['images']['XLarge']
    response = requests.get(link_to_image)  # download
    if not response:
        print 'Fail'
        continue
    image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    if image is None:
        print "not a good image"
        continue

    resized_image = imutils.resize_keep_aspect(image, output_size=(224, 224))

    image_file_name = 'casual_male_' + str(i) + '.jpg'
    cv2.imwrite(os.path.join("/home/yonatan/style_classifier/casual", image_file_name), resized_image)

    casual_txt_file.write(os.path.join("/home/yonatan/style_classifier/casual", image_file_name) + " 0" + "\n")

    counter += 1
    print "counter: {0}, i = {1}".format(counter, i)


list_to_iter = range(1, casual_female_num)
random.shuffle(list_to_iter)
counter = 0

for i in list_to_iter:

    if counter > limit:
        break

    link_to_image = casual_female[i]['images']['XLarge']
    response = requests.get(link_to_image)  # download
    if not response:
        print 'Fail'
        continue
    image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    if image is None:
        print "not a good image"
        continue
        
    resized_image = imutils.resize_keep_aspect(image, output_size=(224, 224))

    image_file_name = 'casual_female_' + str(i) + '.jpg'
    cv2.imwrite(os.path.join("/home/yonatan/style_classifier/casual", image_file_name), resized_image)

    casual_txt_file.write(os.path.join("/home/yonatan/style_classifier/casual", image_file_name) + " 0" + "\n")

    counter += 1
    print "counter: {0}, i = {1}".format(counter, i)


##---- Prom & Homecoming: 1 ----##
regx_prom = re.compile("/*Prom & Homecoming", re.IGNORECASE)
prom_female_num = db.amazon_US_Female.count({'tree': regx_prom})  # = 48754

prom_female = db.amazon_US_Female.find({'tree': regx_prom})

prom_txt_file = open("/home/yonatan/style_classifier/prom_txt_file.txt", "w")

list_to_iter = range(1, casual_female_num)
random.shuffle(list_to_iter)
counter = 0

for i in list_to_iter:

    if counter > limit:
        break

    link_to_image = prom_female[i]['images']['XLarge']
    response = requests.get(link_to_image)  # download
    if not response:
        print 'Fail'
        continue
    image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    if image is None:
        print "not a good image"
        continue
        
    resized_image = imutils.resize_keep_aspect(image, output_size=(224, 224))

    image_file_name = 'prom_female_' + str(i) + '.jpg'
    cv2.imwrite(os.path.join("/home/yonatan/style_classifier/prom", image_file_name), resized_image)

    prom_txt_file.write(os.path.join("/home/yonatan/style_classifier/prom", image_file_name) + " 1" + "\n")

    counter += 1
    print "counter: {0}, i = {1}".format(counter, i)


##---- Tuxedos & Suits: 2 ----##
regx_tux = re.compile("/*Tuxedos", re.IGNORECASE)
regx_suit = re.compile("/*Departments->Men->Clothing->Suits & Sport Coats->Suits", re.IGNORECASE)
tux_male_num = db.amazon_US_Male.count({"tree": regx_tux})  # = 2363
suit_male_num = db.amazon_US_Male.count({"tree": regx_suit})  # = 6202

tux_male = db.amazon_US_Male.find({"tree": regx_tux})  # = 2363
suit_male = db.amazon_US_Male.find({"tree": regx_suit})  # = 6202

tuxedos_txt_file = open("/home/yonatan/style_classifier/tuxedos_txt_file.txt", "w")
suits_txt_file = open("/home/yonatan/style_classifier/suits_txt_file.txt", "w")

list_to_iter = range(1, casual_female_num)
random.shuffle(list_to_iter)
counter = 0

for i in list_to_iter:

    if counter > limit:
        break

    link_to_image = tux_male[i]['images']['XLarge']
    response = requests.get(link_to_image)  # download
    if not response:
        print 'Fail'
        continue
    image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    if image is None:
        print "not a good image"
        continue
        
    resized_image = imutils.resize_keep_aspect(image, output_size=(224, 224))

    image_file_name = 'tux_male_' + str(i) + '.jpg'
    cv2.imwrite(os.path.join("/home/yonatan/style_classifier/tuxedos_and_suits", image_file_name), resized_image)

    tuxedos_txt_file.write(os.path.join("/home/yonatan/style_classifier/tuxedos_and_suits", image_file_name) + " 2" + "\n")

    counter += 1
    print "counter: {0}, i = {1}".format(counter, i)


list_to_iter = range(1, casual_female_num)
random.shuffle(list_to_iter)
counter = 0

for i in list_to_iter:

    if counter > limit:
        break

    link_to_image = suit_male[i]['images']['XLarge']
    response = requests.get(link_to_image)  # download
    if not response:
        print 'Fail'
        continue
    image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    if image is None:
        print "not a good image"
        continue

    resized_image = imutils.resize_keep_aspect(image, output_size=(224, 224))

    image_file_name = 'suit_male_' + str(i) + '.jpg'
    cv2.imwrite(os.path.join("/home/yonatan/style_classifier/tuxedos_and_suits", image_file_name), resized_image)

    suits_txt_file.write(
        os.path.join("/home/yonatan/style_classifier/tuxedos_and_suits", image_file_name) + " 2" + "\n")

    counter += 1
    print "counter: {0}, i = {1}".format(counter, i)


##---- Bride: 3 ----##
regx_bride = re.compile("/*Departments->Women->Clothing->Dresses->Wedding Party->Wedding Dresses", re.IGNORECASE)
bride_female_num = db.amazon_US_Female.count({"tree": regx_bride})  # = 16174

bride_female = db.amazon_US_Female.find({"tree": regx_bride})  # = 16174

bride_dress_txt_file = open("/home/yonatan/style_classifier/bride_dress_txt_file.txt", "w")

list_to_iter = range(1, casual_female_num)
random.shuffle(list_to_iter)
counter = 0

for i in list_to_iter:

    if counter > limit:
        break

    link_to_image = bride_female[i]['images']['XLarge']
    response = requests.get(link_to_image)  # download
    if not response:
        print 'Fail'
        continue
    image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    if image is None:
        print "not a good image"
        continue

    resized_image = imutils.resize_keep_aspect(image, output_size=(224, 224))

    image_file_name = 'bride_female_' + str(i) + '.jpg'
    cv2.imwrite(os.path.join("/home/yonatan/style_classifier/bride_dress", image_file_name), resized_image)

    bride_dress_txt_file.write(
        os.path.join("/home/yonatan/style_classifier/bride_dress", image_file_name) + " 3" + "\n")

    counter += 1
    print "counter: {0}, i = {1}".format(counter, i)


##---- Active: 4 ----##
regx_active = re.compile("/*active", re.IGNORECASE)
# regx3 = re.compile("/*sport", re.IGNORECASE)
# db.amazon_US_Male.count({"tree": {'$in': [regx2, regx3]}})
# db.amazon_US_Female.count({"tree": {'$in': [regx2, regx3]}})
active_male_num = db.amazon_US_Male.count({"tree": regx_active})  # = 239458
active_female_num = db.amazon_US_Female.count({"tree": regx_active})  # = 176763

active_male = db.amazon_US_Male.find({"tree": regx_active})
active_female = db.amazon_US_Female.find({"tree": regx_active})

active_txt_file = open("/home/yonatan/style_classifier/active_txt_file.txt", "w")

list_to_iter = range(1, casual_female_num)
random.shuffle(list_to_iter)
counter = 0

for i in list_to_iter:

    if counter > limit:
        break

    link_to_image = active_male[i]['images']['XLarge']
    response = requests.get(link_to_image)  # download
    if not response:
        print 'Fail'
        continue
    image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    if image is None:
        print "not a good image"
        continue
        
    resized_image = imutils.resize_keep_aspect(image, output_size=(224, 224))

    image_file_name = 'active_male_' + str(i) + '.jpg'
    cv2.imwrite(os.path.join("/home/yonatan/style_classifier/active", image_file_name), resized_image)

    active_txt_file.write(os.path.join("/home/yonatan/style_classifier/active", image_file_name) + " 4" + "\n")

    counter += 1
    print "counter: {0}, i = {1}".format(counter, i)


list_to_iter = range(1, casual_female_num)
random.shuffle(list_to_iter)
counter = 0

for i in list_to_iter:

    if counter > limit:
        break

    link_to_image = active_female[i]['images']['XLarge']
    response = requests.get(link_to_image)  # download
    if not response:
        print 'Fail'
        continue
    image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    if image is None:
        print "not a good image"
        continue
        
    resized_image = imutils.resize_keep_aspect(image, output_size=(224, 224))

    image_file_name = 'active_female_' + str(i) + '.jpg'
    cv2.imwrite(os.path.join("/home/yonatan/style_classifier/active", image_file_name), resized_image)

    active_txt_file.write(os.path.join("/home/yonatan/style_classifier/active", image_file_name) + " 4" + "\n")

    counter += 1
    print "counter: {0}, i = {1}".format(counter, i)


##---- Swim: 5 ----##
regx_swim = re.compile("/*swim", re.IGNORECASE)
swim_male_num = db.amazon_US_Male.count({'tree': regx_swim})  # = 25116
swim_female_num = db.amazon_US_Female.count({"tree": regx_swim})  # = 170063

swim_male = db.amazon_US_Male.find({'tree': regx_swim})
swim_female = db.amazon_US_Female.find({'tree': regx_swim})

swim_txt_file = open("/home/yonatan/style_classifier/swim_txt_file.txt", "w")

list_to_iter = range(1, casual_female_num)
random.shuffle(list_to_iter)
counter = 0

for i in list_to_iter:

    if counter > limit:
        break

    link_to_image = swim_male[i]['images']['XLarge']
    response = requests.get(link_to_image)  # download
    if not response:
        print 'Fail'
        continue
    image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    if image is None:
        print "not a good image"
        continue
        
    resized_image = imutils.resize_keep_aspect(image, output_size=(224, 224))

    image_file_name = 'swim_male_' + str(i) + '.jpg'
    cv2.imwrite(os.path.join("/home/yonatan/style_classifier/swim", image_file_name), resized_image)

    swim_txt_file.write(os.path.join("/home/yonatan/style_classifier/swim", image_file_name) + " 5" + "\n")

    counter += 1
    print "counter: {0}, i = {1}".format(counter, i)


list_to_iter = range(1, casual_female_num)
random.shuffle(list_to_iter)
counter = 0

for i in list_to_iter:

    if counter > limit:
        break

    link_to_image = swim_female[i]['images']['XLarge']
    response = requests.get(link_to_image)  # download
    if not response:
        print 'Fail'
        continue
    image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    if image is None:
        print "not a good image"
        continue
        
    resized_image = imutils.resize_keep_aspect(image, output_size=(224, 224))

    image_file_name = 'swim_female_' + str(i) + '.jpg'
    cv2.imwrite(os.path.join("/home/yonatan/style_classifier/swim", image_file_name), resized_image)

    swim_txt_file.write(os.path.join("/home/yonatan/style_classifier/swim", image_file_name) + " 5" + "\n")

    counter += 1
    print "counter: {0}, i = {1}".format(counter, i)



