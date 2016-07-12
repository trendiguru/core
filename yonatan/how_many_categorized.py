#!/usr/bin/env python

import numpy as np
import skimage.io
import os
import caffe
from .. import background_removal, Utils, constants
import cv2
import sys
import argparse
import glob
import time
import skimage
import urllib
import pymongo


def how_many(argv):

    db = constants.db

    # dress sleeve #
    if argv == 'dress_sleeve':
        dress_sleeve_dict = {
        'strapless' : db.yonatan_dresses.count({'sleeve_length': ['true', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}),
        'spaghetti_straps' : db.yonatan_dresses.count({'sleeve_length': ['false', 'true', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}),
        'straps' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'true', 'false', 'false', 'false', 'false', 'false', 'false']}),
        'sleeveless' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'false', 'true', 'false', 'false', 'false', 'false', 'false']}),
        'cap_sleeve' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'false', 'false', 'true', 'false', 'false', 'false', 'false']}),
        'short_sleeve' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'true', 'false', 'false', 'false']}),
        'midi_sleeve' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'true', 'false', 'false']}),
        'long_sleeve' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'true', 'false']}),
        'asymmetry' : db.yonatan_dresses.count({'sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'true']})
        }

        sum_of_all = dress_sleeve_dict['strapless'] + dress_sleeve_dict['spaghetti_straps'] + dress_sleeve_dict['straps'] + \
                     dress_sleeve_dict['sleeveless'] + dress_sleeve_dict['cap_sleeve'] + dress_sleeve_dict['short_sleeve'] + \
                     dress_sleeve_dict['midi_sleeve'] + dress_sleeve_dict['long_sleeve'] + dress_sleeve_dict['asymmetry']

        sum_of_all_already_seen = db.yonatan_dresses.count({'already_seen_dress_sleeve': True})

        deleted = sum_of_all_already_seen - sum_of_all

        for key, value in dress_sleeve_dict.iteritems():
            print '{0}: {1}, percent: {2}'.format(key, value, round(float(value) / sum_of_all, 2))


    # dress length #
    elif argv == 'dress_length':
        dress_length_dict = {
        'mini_length' : db.yonatan_dresses.count({'dress_length': ['true', 'false', 'false', 'false', 'false', 'false']}),
        'above_knee' : db.yonatan_dresses.count({'dress_length': ['false', 'true', 'false', 'false', 'false', 'false']}),
        'knee_length' : db.yonatan_dresses.count({'dress_length': ['false', 'false', 'true', 'false', 'false', 'false']}),
        'tea_length' : db.yonatan_dresses.count({'dress_length': ['false', 'false', 'false', 'true', 'false', 'false']}),
        'ankle_length' : db.yonatan_dresses.count({'dress_length': ['false', 'false', 'false', 'false', 'true', 'false']}),
        'floor_length' : db.yonatan_dresses.count({'dress_length': ['false', 'false', 'false', 'false', 'false', 'true']})
        }

        sum_of_all = dress_length_dict['mini_length'] + dress_length_dict['above_knee'] + dress_length_dict['knee_length'] + \
                     dress_length_dict['tea_length'] + dress_length_dict['ankle_length'] + dress_length_dict['floor_length']

        sum_of_all_already_seen = db.yonatan_dresses.count({'already_seen_dress_length': True})

        deleted = sum_of_all_already_seen - sum_of_all

        for key, value in dress_length_dict.iteritems():
            print '{0}: {1}, percent: {2}'.format(key, value, round(float(value) / sum_of_all, 2))


    # men shirt sleeve #
    elif argv == 'men_shirt_sleeve':
        men_shirt_sleeve_dict = {
        'straps' : db.yonatan_men_shirts.count({'shirt_sleeve_length': ['true', 'false', 'false', 'false', 'false']}),
        'sleeveless' : db.yonatan_men_shirts.count({'shirt_sleeve_length': ['false', 'true', 'false', 'false', 'false']}),
        'short_sleeve' : db.yonatan_men_shirts.count({'shirt_sleeve_length': ['false', 'false', 'true', 'false', 'false']}),
        'midi_sleeve' : db.yonatan_men_shirts.count({'shirt_sleeve_length': ['false', 'false', 'false', 'true', 'false']}),
        'long_sleeve' : db.yonatan_men_shirts.count({'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'true']})
        }

        sum_of_all = men_shirt_sleeve_dict['straps'] + men_shirt_sleeve_dict['sleeveless'] + men_shirt_sleeve_dict['short_sleeve'] + \
                     men_shirt_sleeve_dict['midi_sleeve'] + men_shirt_sleeve_dict['long_sleeve']

        sum_of_all_already_seen = db.yonatan_men_shirts.count({'already_seen_men_shirt_sleeve': True})

        deleted = sum_of_all_already_seen - sum_of_all

        for key, value in men_shirt_sleeve_dict.iteritems():
            print '{0}: {1}, percent: {2}'.format(key, value, round(float(value) / sum_of_all, 2))


    # pants length #
    elif argv == 'pants_length':
        pants_length_dict = {
        'bermuda' : db.yonatan_pants.count({'pants_length': ['true', 'false', 'false', 'false']}),
        'knee' : db.yonatan_pants.count({'pants_length': ['false', 'true', 'false', 'false']}),
        'capri' : db.yonatan_pants.count({'pants_length': ['false', 'false', 'true', 'false']}),
        'floor' : db.yonatan_pants.count({'pants_length': ['false', 'false', 'false', 'true']})
        }

        sum_of_all = pants_length_dict['bermuda'] + pants_length_dict['knee'] + \
                     pants_length_dict['capri'] + pants_length_dict['floor']

        sum_of_all_already_seen = db.yonatan_pants.count({'already_seen_pants_length': True})

        deleted = sum_of_all_already_seen - sum_of_all

        for key, value in pants_length_dict.iteritems():
            print '{0}: {1}, percent: {2}'.format(key, value, round(float(value) / sum_of_all, 2))


    # women shirt sleeve #
    elif argv == 'women_shirt_sleeve':
        women_shirt_sleeve_dict = {
        'strapless' : db.yonatan_women_shirts.count({'shirt_sleeve_length': ['true', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}),
        'spaghetti_straps' : db.yonatan_women_shirts.count({'shirt_sleeve_length': ['false', 'true', 'false', 'false', 'false', 'false', 'false', 'false', 'false']}),
        'straps' : db.yonatan_women_shirts.count({'shirt_sleeve_length': ['false', 'false', 'true', 'false', 'false', 'false', 'false', 'false', 'false']}),
        'sleeveless' : db.yonatan_women_shirts.count({'shirt_sleeve_length': ['false', 'false', 'false', 'true', 'false', 'false', 'false', 'false', 'false']}),
        'cap_sleeve' : db.yonatan_women_shirts.count({'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'true', 'false', 'false', 'false', 'false']}),
        'short_sleeve' : db.yonatan_women_shirts.count({'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'true', 'false', 'false', 'false']}),
        'midi_sleeve' : db.yonatan_women_shirts.count({'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'true', 'false', 'false']}),
        'long_sleeve' : db.yonatan_women_shirts.count({'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'true', 'false']}),
        'asymmetry' : db.yonatan_women_shirts.count({'shirt_sleeve_length': ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'true']})
        }

        sum_of_all = women_shirt_sleeve_dict['strapless'] + women_shirt_sleeve_dict['spaghetti_straps'] + women_shirt_sleeve_dict['straps'] + \
                     women_shirt_sleeve_dict['sleeveless'] + women_shirt_sleeve_dict['cap_sleeve'] + women_shirt_sleeve_dict['short_sleeve'] + \
                     women_shirt_sleeve_dict['midi_sleeve'] + women_shirt_sleeve_dict['long_sleeve'] + women_shirt_sleeve_dict['asymmetry']

        sum_of_all_already_seen = db.yonatan_women_shirts.count({'already_seen_women_shirt_sleeve': True})

        deleted = sum_of_all_already_seen - sum_of_all

        for key, value in women_shirt_sleeve_dict.iteritems():
            print '{0}: {1}, percent: {2}'.format(key, value, round(float(value) / sum_of_all, 2))

    else:
        print "wrong input!"
        print "the argument should be one of those:\n{0}\n{1}\n{2}\n{3}/\n{4}".format('dress_sleeve',
                                                                                     'dress_length', 'men_shirt_sleeve', 'pants_length', 'women_shirt_sleeve')
        return

    print 'sum of all: {0}'.format(sum_of_all)
    print 'deleted: {0}'.format(deleted)

if __name__ == '__main__':
    how_many(sys.argv)
