#!/usr/bin/env python

import pymongo
import re
import cv2
import requests
import numpy as np
import os

db = pymongo.MongoClient().mydb

## regex:
# startswith - ^
# is there - .


##---- Casual: 0 ----##
regx_casual = re.compile("/*Casual", re.IGNORECASE)
casual_male_num = db.amazon_US_Male.count({'tree': regx_casual})  # = 117405
casual_female_num = db.amazon_US_Female.count({'tree': regx_casual})  # = 205237

casual_male = db.amazon_US_Male.find({'tree': regx_casual})
casual_female = db.amazon_US_Female.find({'tree': regx_casual})

casual_txt_file = open("/home/yonatan/style_classifier/casual_txt_file.txt", "w")

for i in range(1, casual_male_num):
    link_to_image = casual_male[i]['images']['XLarge']
    response = requests.get(link_to_image)  # download
    if not response:
        print 'Fail'
        continue
    image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    if image is None:
        print "not a good image"
        continue

    image_file_name = 'casual_male_' + str(i) + '.jpg'
    cv2.imwrite(os.path.join("/home/yonatan/style_classifier/casual", image_file_name), image)

    casual_txt_file.write(os.path.join("/home/yonatan/style_classifier/casual", image_file_name) + " 0" + "\n")

for i in range(1, casual_female_num):
    link_to_image = casual_female[i]['images']['XLarge']
    response = requests.get(link_to_image)  # download
    if not response:
        print 'Fail'
        continue
    image = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    if image is None:
        print "not a good image"
        continue

    image_file_name = 'casual_female_' + str(i) + '.jpg'
    cv2.imwrite(os.path.join("/home/yonatan/style_classifier/casual", image_file_name), image)

    casual_txt_file.write(os.path.join("/home/yonatan/style_classifier/casual", image_file_name) + " 0" + "\n")

#
# ##---- Prom & Homecoming: 1 ----##
# regx_prom = re.compile("/*Prom & Homecoming", re.IGNORECASE)
# prom_female_num = db.amazon_US_Female.count({'tree': regx_prom})  # = 48754
#
# prom_female = db.amazon_US_Female.find({'tree': regx_prom})
#
# prom_txt_file = open("/home/yonatan/style_classifier/prom_txt_file.txt", "w")
#
#
# ##---- Tuxedos & Suits: 2 ----##
# regx_tux = re.compile("/*Tuxedos", re.IGNORECASE)
# regx_suit = re.compile("/*Departments->Men->Clothing->Suits & Sport Coats->Suits", re.IGNORECASE)
# tux_male_num = db.amazon_US_Male.count({"tree": regx_tux})  # = 2363
# suit_male_num = db.amazon_US_Male.count({"tree": regx_suit})  # = 6202
#
# tux_male = db.amazon_US_Male.find({"tree": regx_tux})  # = 2363
# suit_male = db.amazon_US_Male.find({"tree": regx_suit})  # = 6202
#
# tuxedos_txt_file = open("/home/yonatan/style_classifier/tuxedos_txt_file.txt", "w")
# suits_txt_file = open("/home/yonatan/style_classifier/suits_txt_file.txt", "w")
#
#
# ##---- Bride: 3 ----##
# regx_bride = re.compile("/*Departments->Women->Clothing->Dresses->Wedding Party->Wedding Dresses", re.IGNORECASE)
# bride_female_num = db.amazon_US_Female.count({"tree": regx_bride})  # = 16174
#
# bride_female = db.amazon_US_Female.find({"tree": regx_bride})  # = 16174
#
# bride_dress_txt_file = open("/home/yonatan/style_classifier/bride_dress_txt_file.txt", "w")
#
#
# ##---- Sport: 4 ----##
# regx_active = re.compile("/*active", re.IGNORECASE)
# # regx3 = re.compile("/*sport", re.IGNORECASE)
# # db.amazon_US_Male.count({"tree": {'$in': [regx2, regx3]}})
# # db.amazon_US_Female.count({"tree": {'$in': [regx2, regx3]}})
# active_male_num = db.amazon_US_Male.count({"tree": regx_active})  # = 239458
# active_female_num = db.amazon_US_Female.count({"tree": regx_active})  # = 176763
#
# active_male = db.amazon_US_Male.find({"tree": regx_active})
# active_female = db.amazon_US_Female.find({"tree": regx_active})
#
# active_txt_file = open("/home/yonatan/style_classifier/active_txt_file.txt", "w")
#
#
# ##---- Swim: 5 ----##
# regx_swim = re.compile("/*swim", re.IGNORECASE)
# swim_male_num = db.amazon_US_Male.count({'tree': regx_swim})  # = 25116
# swim_female_num = db.amazon_US_Female.count({"tree": regx_swim})  # = 170063
#
# swim_male = db.amazon_US_Male.find({'tree': regx_swim})
# swim_female = db.amazon_US_Female.find({'tree': regx_swim})
#
# swim_txt_file = open("/home/yonatan/style_classifier/swim_txt_file.txt", "w")