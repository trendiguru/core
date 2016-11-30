#!/usr/bin/env python

import pymongo
import re
import cv2
import requests
import numpy as np
import os
import random
from ..utils import imutils
import yonatan_constants


def style_classifier_1():

    db = pymongo.MongoClient().mydb

    ## regex:
    # startswith - ^
    # is there - .

    limit = 7500


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
    error_counter = 0

    for i in list_to_iter:

        if counter > limit:
            break

        try:
            link_to_image = casual_male[i]['images']['XLarge']
        except:
            print "link ain't good"
            error_counter += 1
            continue

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

        print "counter: {0}, i = {1}, error_counter = {2}".format(counter, i, error_counter)
        counter += 1


    list_to_iter = range(1, casual_female_num)
    random.shuffle(list_to_iter)
    counter = 1
    error_counter = 0

    for i in list_to_iter:

        if counter > limit:
            break

        try:
            link_to_image = casual_female[i]['images']['XLarge']
        except:
            print "link ain't good"
            error_counter += 1
            continue

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

        print "counter: {0}, i = {1}, error_counter = {2}".format(counter, i, error_counter)
        counter += 1

    casual_txt_file.close()


    ##---- Prom & Homecoming: 1 ----##
    regx_prom = re.compile("/*Prom & Homecoming", re.IGNORECASE)
    prom_female_num = db.amazon_US_Female.count({'tree': regx_prom})  # = 48754

    prom_female = db.amazon_US_Female.find({'tree': regx_prom})

    prom_txt_file = open("/home/yonatan/style_classifier/prom_txt_file.txt", "w")

    list_to_iter = range(1, prom_female_num)
    random.shuffle(list_to_iter)
    counter = 1
    error_counter = 0

    for i in list_to_iter:

        if counter > limit:
            break

        try:
            link_to_image = prom_female[i]['images']['XLarge']
        except:
            print "link ain't good"
            error_counter += 1
            continue

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

        print "counter: {0}, i = {1}, error_counter = {2}".format(counter, i, error_counter)
        counter += 1

    prom_txt_file.close()


    ##---- Tuxedos & Suits: 2 ----##
    regx_tux = re.compile("/*Tuxedos", re.IGNORECASE)
    regx_suit = re.compile("/*Departments->Men->Clothing->Suits & Sport Coats->Suits", re.IGNORECASE)
    tux_male_num = db.amazon_US_Male.count({"tree": regx_tux})  # = 2363
    suit_male_num = db.amazon_US_Male.count({"tree": regx_suit})  # = 6202

    tux_male = db.amazon_US_Male.find({"tree": regx_tux})  # = 2363
    suit_male = db.amazon_US_Male.find({"tree": regx_suit})  # = 6202

    tuxedos_txt_file = open("/home/yonatan/style_classifier/tuxedos_txt_file.txt", "w")
    suits_txt_file = open("/home/yonatan/style_classifier/suits_txt_file.txt", "w")

    list_to_iter = range(1, tux_male_num)
    random.shuffle(list_to_iter)
    counter = 1
    error_counter = 0

    for i in list_to_iter:

        if counter > limit:
            break

        try:
            link_to_image = tux_male[i]['images']['XLarge']
        except:
            print "link ain't good"
            error_counter += 1
            continue

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

        print "counter: {0}, i = {1}, error_counter = {2}".format(counter, i, error_counter)
        counter += 1

    tuxedos_txt_file.close()


    list_to_iter = range(1, suit_male_num)
    random.shuffle(list_to_iter)
    counter = 1
    error_counter = 0

    for i in list_to_iter:

        if counter > limit:
            break

        try:
            link_to_image = suit_male[i]['images']['XLarge']
        except:
            print "link ain't good"
            error_counter += 1
            continue

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

        suits_txt_file.write(os.path.join("/home/yonatan/style_classifier/tuxedos_and_suits", image_file_name) + " 2" + "\n")

        print "counter: {0}, i = {1}, error_counter = {2}".format(counter, i, error_counter)
        counter += 1

    suits_txt_file.close()


    ##---- Bride: 3 ----##
    regx_bride = re.compile("/*Departments->Women->Clothing->Dresses->Wedding Party->Wedding Dresses", re.IGNORECASE)
    bride_female_num = db.amazon_US_Female.count({"tree": regx_bride})  # = 16174

    bride_female = db.amazon_US_Female.find({"tree": regx_bride})  # = 16174

    bride_dress_txt_file = open("/home/yonatan/style_classifier/bride_dress_txt_file.txt", "w")

    list_to_iter = range(1, bride_female_num)
    random.shuffle(list_to_iter)
    counter = 1
    error_counter = 0

    for i in list_to_iter:

        if counter > limit:
            break

        try:
            link_to_image = bride_female[i]['images']['XLarge']
        except:
            print "link ain't good"
            error_counter += 1
            continue

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

        bride_dress_txt_file.write(os.path.join("/home/yonatan/style_classifier/bride_dress", image_file_name) + " 3" + "\n")

        print "counter: {0}, i = {1}, error_counter = {2}".format(counter, i, error_counter)
        counter += 1

    bride_dress_txt_file.close()


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

    list_to_iter = range(1, active_male_num)
    random.shuffle(list_to_iter)
    counter = 1
    error_counter = 0

    for i in list_to_iter:

        if counter > limit:
            break

        try:
            link_to_image = active_male[i]['images']['XLarge']
        except:
            print "link ain't good"
            error_counter += 1
            continue

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

        print "counter: {0}, i = {1}, error_counter = {2}".format(counter, i, error_counter)
        counter += 1

    list_to_iter = range(1, active_female_num)
    random.shuffle(list_to_iter)
    counter = 1
    error_counter = 0

    for i in list_to_iter:

        if counter > limit:
            break

        try:
            link_to_image = active_female[i]['images']['XLarge']
        except:
            print "link ain't good"
            error_counter += 1
            continue

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

        print "counter: {0}, i = {1}, error_counter = {2}".format(counter, i, error_counter)
        counter += 1

    active_txt_file.close()


    ##---- Swim: 5 ----##
    regx_swim = re.compile("/*swim", re.IGNORECASE)
    swim_male_num = db.amazon_US_Male.count({'tree': regx_swim})  # = 25116
    swim_female_num = db.amazon_US_Female.count({"tree": regx_swim})  # = 170063

    swim_male = db.amazon_US_Male.find({'tree': regx_swim})
    swim_female = db.amazon_US_Female.find({'tree': regx_swim})

    swim_txt_file = open("/home/yonatan/style_classifier/swim_txt_file.txt", "w")

    list_to_iter = range(1, swim_male_num)
    random.shuffle(list_to_iter)
    counter = 1
    error_counter = 0

    for i in list_to_iter:

        if counter > limit:
            break

        try:
            link_to_image = swim_male[i]['images']['XLarge']
        except:
            print "link ain't good"
            error_counter += 1
            continue

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

        print "counter: {0}, i = {1}, error_counter = {2}".format(counter, i, error_counter)
        counter += 1

    list_to_iter = range(1, swim_female_num)
    random.shuffle(list_to_iter)
    counter = 1
    error_counter = 0

    for i in list_to_iter:

        if counter > limit:
            break

        try:
            link_to_image = swim_female[i]['images']['XLarge']
        except:
            print "link ain't good"
            error_counter += 1
            continue

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

        print "counter: {0}, i = {1}, error_counter = {2}".format(counter, i, error_counter)
        counter += 1

    swim_txt_file.close()


def resize_save():

    #############

    dictionary = yonatan_constants.style_dict

    error_counter = 0
    counter_dot = 0

    source_dir = "/home/yonatan/test_can_delete"

    for root, dirs, files in os.walk(source_dir):

        counter = 0

        for file in files:

            if "._" in file:
                counter_dot += 1
                continue

            counter += 1
            print counter

            try:
                image_array = cv2.imread(os.path.join(root, file))
                resized_image = imutils.resize_keep_aspect(image_array, output_size=(224, 224))

                image_file_name = 'style-resized_' + str(counter) + '.jpg'

                cv2.imwrite(os.path.join(root, image_file_name), resized_image)

                os.remove(os.path.join(root, file))

            except:
                print "something ain't good"
                error_counter += 1
                continue

    print "number of errors: {0}, number of '._: {1}".format(error_counter, counter_dot)

    # ## erase useless ##
    #
    # dictionary = yonatan_constants.style_dict
    #
    # error_counter = 0
    #
    # for key, value in dictionary.iteritems():
    #     source_dir = '/home/yonatan/style_classifier/style_second_try/style_images/' + key
    #
    #     if os.path.isdir(source_dir):
    #         if not os.listdir(source_dir):
    #             print '\nfolder is empty ' + key
    #             break
    #     else:
    #         print '\nfolder doesn\'t exist ' + key
    #         break
    #
    #     for root, dirs, files in os.walk(source_dir):
    #
    #         counter = 0
    #
    #         for file in files:
    #
    #             counter += 1
    #             print counter
    #
    #             if file.startswith("images") or "._" in file:
    #                 try:
    #                     os.remove(os.path.join(root, file))
    #                 except:
    #                     print "something ain't good"
    #                     error_counter += 1
    #                     continue
    #
    # print "number of errors: {0}".format(error_counter)


def collar_classifier():

    dictionary = yonatan_constants.collar_basic_dict

    error_counter = 0

    for key, value in dictionary.iteritems():
        source_dir = '/home/yonatan/collar_classifier/collar_images/' + key

        if os.path.isdir(source_dir):
            if not os.listdir(source_dir):
                print '\nfolder is empty ' + key
                break
        else:
            print '\nfolder doesn\'t exist ' + key
            break

        for root, dirs, files in os.walk(source_dir):

            counter = 0

            for file in files:

                if "._" in file:
                    continue

                counter += 1
                print counter

                try:
                    image_array = cv2.imread(os.path.join(root, file))
                    resized_image = imutils.resize_keep_aspect(image_array, output_size=(224, 224))

                    image_file_name = 'collar-' + key + '_' + str(counter) + '.jpg'

                    cv2.imwrite(os.path.join(root, image_file_name), resized_image)

                except:
                    print "something ain't good"
                    error_counter += 1
                    continue

    print "number of errors: {0}".format(error_counter)


if __name__ == '__main__':
    # style_classifier_1()
    resize_save()
    # collar_classifier()
